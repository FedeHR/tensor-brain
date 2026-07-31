"""Read and verify immutable PVSG snapshot files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON document."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty rows from one UTF-8 JSON Lines file."""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    """Hash a file without loading it completely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
