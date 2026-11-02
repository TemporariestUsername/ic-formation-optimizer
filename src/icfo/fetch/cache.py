"""On-disk JSON cache for API responses.

Definitions are large (~20 MB) and change only with game patches, so they are
cached by the server-reported checksum. User data is cached with a timestamp so
repeated optimizer runs don't call the game servers again.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path(os.environ.get("ICFO_CACHE_DIR", ".icfo-cache"))


class Cache:
    def __init__(self, root: Path | str = DEFAULT_CACHE_DIR):
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def load(self, name: str, max_age_s: float | None = None) -> Any | None:
        path = self._path(name)
        if not path.exists():
            return None
        if max_age_s is not None and time.time() - path.stat().st_mtime > max_age_s:
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def save(self, name: str, data: Any) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(name)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
        return path
