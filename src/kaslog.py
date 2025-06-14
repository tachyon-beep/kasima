"""Simple CLI for verifying germination logs."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def verify(file: Path) -> bool:
    prev = ""
    with file.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            root = obj.get("root")
            event = obj.get("event")
            calc = sha256((prev + json.dumps(event)).encode()).hexdigest()
            if calc != root:
                return False
            prev = root
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verify", choices=["verify"])
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    ok = verify(args.file)
    if not ok:
        raise SystemExit("log verification failed")
    print("log verified")


if __name__ == "__main__":
    main()
