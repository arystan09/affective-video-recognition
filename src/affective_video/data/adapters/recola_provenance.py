"""Stable source hashes with explicit cache invalidation and containment."""

import hashlib
import json
from pathlib import Path


def local_source(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"source path must remain relative and inside release root: {relative}")
    return path


class ChecksumCache:
    """Cache by relative path, size, mtime_ns, ctime_ns; --rehash defeats stat reuse."""

    def __init__(self, root: Path, cache_path: Path | None = None, *, rehash: bool = False):
        self.root, self.cache_path, self.rehash = root.resolve(), cache_path, rehash
        self.entries: dict = {}
        if cache_path and cache_path.exists():
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            if saved.get("root") == str(self.root):
                self.entries = saved.get("entries", {})

    def checksum(self, relative: str) -> str:
        path = local_source(self.root, relative)
        before = path.stat()
        fingerprint = [before.st_size, before.st_mtime_ns, before.st_ctime_ns]
        cached = self.entries.get(relative)
        if not self.rehash and cached and cached["fingerprint"] == fingerprint:
            return str(cached["sha256"])
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if fingerprint != [after.st_size, after.st_mtime_ns, after.st_ctime_ns]:
            raise ValueError(f"source changed while hashing: {relative}")
        result = digest.hexdigest()
        self.entries[relative] = {"fingerprint": fingerprint, "sha256": result}
        return result

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps({"root": str(self.root), "entries": self.entries}, indent=2),
                encoding="utf-8",
            )
