"""Exchange the hackathon AWS keys (from the EgoVerse README) for R2 read
credentials via Secrets Manager — a boto3 port of EgoVerse's setup_secret.sh
(no AWS CLI on this machine). Keys are parsed from the README at runtime and
never printed; output goes to ~/.egoverse_env, chmod 600, same as upstream.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import boto3

README = Path.home() / "Desktop/hackathon/EgoVerse/README.md"
ENV_FILE = Path.home() / ".egoverse_env"
REGION = "us-east-2"


def parse_readme_keys() -> tuple[str, str]:
    text = README.read_text()
    ak = re.search(r"AccessKeyId:\s*(\S+)", text)
    sk = re.search(r"SecretAccessKey:\s*(\S+)", text)
    if not ak or not sk:
        sys.exit("could not find AWS keys in EgoVerse README")
    return ak.group(1), sk.group(1)


def main() -> None:
    ak, sk = parse_readme_keys()
    sm = boto3.client(
        "secretsmanager",
        region_name=REGION,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
    )
    payload = None
    for name in ("r2/rldb/credentials", "r2/rldb/public/credentials"):
        try:
            payload = json.loads(sm.get_secret_value(SecretId=name)["SecretString"])
            print(f"got R2 secret: {name} (keys: {sorted(payload)})")
            break
        except Exception as e:  # noqa: BLE001 — report and try the public secret
            print(f"{name}: {type(e).__name__}: {str(e)[:100]}")
    if payload is None:
        sys.exit("no R2 secret accessible")

    lines = [
        f"R2_ACCESS_KEY_ID={payload['access_key_id']}",
        f"R2_SECRET_ACCESS_KEY={payload['secret_access_key']}",
        f"AWS_ENDPOINT_URL_S3={payload['endpoint_url']}",
        f"AWS_DEFAULT_REGION={REGION}",
        f"BUCKET={payload.get('bucket', 'rldb')}",
    ]
    if payload.get("session_token"):
        lines.append(f"R2_SESSION_TOKEN={payload['session_token']}")
    ENV_FILE.write_text("\n".join(lines) + "\n")
    os.chmod(ENV_FILE, 0o600)
    print(f"wrote {ENV_FILE} (0600); endpoint host: {payload['endpoint_url'].split('//')[-1][:20]}...")

    # also try the read-only DB secret for episode metadata (task labels)
    for db_name in ("rds/appdb/appuser-readonly", "rds/appdb/appuser"):
        try:
            db = json.loads(sm.get_secret_value(SecretId=db_name)["SecretString"])
            with ENV_FILE.open("a") as f:
                f.write(f"DB_SECRET_JSON={json.dumps(db)}\n")
            print(f"got DB secret: {db_name} (keys: {sorted(db)})")
            break
        except Exception as e:  # noqa: BLE001
            print(f"{db_name}: {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
