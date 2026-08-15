"""Stratified pose-only download of real EgoVerse episodes from R2.

Per episode we pull ONLY: zarr.json (attrs: task_name, embodiment, fps),
left/right.obs_ee_pose, obs_head_pose (absent for some vendors), annotations
— ~400 KB/episode vs GBs with video. Credentials come from ~/.egoverse_env
(written by fetch_r2_creds.py); nothing is printed.

Usage: python experiments/download_real_sample.py [--per-lab 60] [--out data/real]
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config

# mecka nests one level deeper (flagship/freeform protocol prefixes);
# lightwheel is raw .mcap, not zarr — out of scope for the pose pipeline
LABS = ["aria", "microagi", "mecka/flagship", "scale"]
WANT_PREFIXES = (
    "zarr.json",
    "left.obs_ee_pose/",
    "right.obs_ee_pose/",
    "obs_head_pose/",
    "annotations/",
)


def client():
    env = dict(
        l.split("=", 1)
        for l in Path.home().joinpath(".egoverse_env").read_text().splitlines()
        if "=" in l
    )
    return (
        boto3.client(
            "s3",
            endpoint_url=env["AWS_ENDPOINT_URL_S3"],
            aws_access_key_id=env["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(max_pool_connections=32, retries={"max_attempts": 5}),
        ),
        env.get("BUCKET", "rldb"),
    )


def list_episodes(s3, bucket: str, lab: str, cap: int = 4000) -> list[str]:
    eps, token = [], None
    while len(eps) < cap:
        kw = {"Bucket": bucket, "Prefix": f"processed_v3/{lab}/", "Delimiter": "/", "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        eps += [p["Prefix"] for p in r.get("CommonPrefixes", []) if p["Prefix"].endswith(".zarr/")]
        token = r.get("NextContinuationToken")
        if not token:
            break
    return eps


def episode_keys(s3, bucket: str, ep_prefix: str) -> list[tuple[str, int]]:
    keys = []
    for want in WANT_PREFIXES:
        r = s3.list_objects_v2(Bucket=bucket, Prefix=ep_prefix + want, MaxKeys=200)
        keys += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
    return keys


def download_episode(s3, bucket: str, ep_prefix: str, out_root: Path) -> int:
    lab = ep_prefix.split("/")[1]
    ep_name = ep_prefix.rstrip("/").split("/")[-1]
    dest = out_root / lab / ep_name
    total = 0
    for key, size in episode_keys(s3, bucket, ep_prefix):
        rel = key[len(ep_prefix):]
        target = dest / rel
        if target.exists() and target.stat().st_size == size:
            total += size
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(target))
        total += size
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lab", type=int, default=60)
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s3, bucket = client()
    out_root = Path(args.out)
    rng = random.Random(args.seed)

    chosen: list[str] = []
    for lab in LABS:
        eps = list_episodes(s3, bucket, lab)
        take = min(args.per_lab, len(eps))
        chosen += rng.sample(eps, take) if take else []
        print(f"{lab}: {len(eps)} listed (first page cap 4000), sampling {take}", flush=True)

    done = grand = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(download_episode, s3, bucket, ep, out_root): ep for ep in chosen}
        for fut in as_completed(futs):
            try:
                grand += fut.result()
            except Exception as e:  # noqa: BLE001 — skip broken episodes, report at end
                print(f"FAILED {futs[fut]}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(chosen)} episodes, {grand/1e6:.0f} MB", flush=True)
    print(f"DONE: {done}/{len(chosen)} episodes, {grand/1e6:.0f} MB -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
