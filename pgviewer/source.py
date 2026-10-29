"""입력 추상화: PGDATA 폴더 또는 PGDATA 가 들어 있는 zip.

원본은 읽기만 한다. 폴더는 'rb' 로만 열고, zip 은 압축을 풀지 않고 메모리에서 읽는다.
경로는 PGDATA 기준 상대경로이고 구분자는 '/' 다.
"""

from __future__ import annotations

import hashlib
import os
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path


class SourceError(Exception):
    pass


def _is_pgdata_dir(p: Path) -> bool:
    return (p / "PG_VERSION").is_file() and (p / "global" / "pg_control").is_file()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Source:
    kind = ""

    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.pgdata = ""  # 입력 안에서 PGDATA 의 위치 (표시용)

    def exists(self, rel: str) -> bool:
        raise NotImplementedError

    def size(self, rel: str) -> int:
        raise NotImplementedError

    def read(self, rel: str, offset: int = 0, length: int = -1) -> bytes:
        raise NotImplementedError

    def listdir(self, rel: str) -> list[str]:
        raise NotImplementedError

    def input_hash(self) -> dict:
        raise NotImplementedError

    def read_outside(self, rel: str) -> bytes | None:
        """PGDATA 밖(입력 루트 기준) 파일. 없으면 None."""
        return None

    def close(self) -> None:
        pass


class DirSource(Source):
    kind = "dir"

    def __init__(self, input_path: Path, root: Path):
        super().__init__(input_path)
        self.root = root
        self.pgdata = str(root)

    def _p(self, rel: str) -> Path:
        parts = [x for x in rel.replace("\\", "/").split("/") if x]
        if any(x == ".." or ":" in x for x in parts):
            raise FileNotFoundError(f"PGDATA 밖을 가리키는 경로: {rel}")
        return self.root.joinpath(*parts)

    def exists(self, rel):
        return self._p(rel).is_file()

    def size(self, rel):
        return self._p(rel).stat().st_size

    def read(self, rel, offset=0, length=-1):
        with open(self._p(rel), "rb") as f:
            f.seek(offset)
            return f.read(length)

    def listdir(self, rel):
        p = self._p(rel)
        return sorted(os.listdir(p)) if p.is_dir() else []

    def input_hash(self):
        # 폴더는 파일별 SHA-256 목록(매니페스트)의 SHA-256 을 대표값으로 쓴다
        lines = []
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                fp = Path(dirpath) / name
                rel = fp.relative_to(self.root).as_posix()
                lines.append(f"{sha256_file(fp)}  {rel}")
        lines.sort(key=lambda s: s[66:])
        manifest = "\n".join(lines) + "\n"
        return {
            "algorithm": "sha256",
            "kind": "manifest",
            "value": hashlib.sha256(manifest.encode()).hexdigest(),
            "files": len(lines),
        }


class ZipSource(Source):
    kind = "zip"
    CACHE_BYTES = 512 << 20

    def __init__(self, input_path: Path, zf: zipfile.ZipFile, prefix: str):
        super().__init__(input_path)
        self.zf = zf
        self.prefix = prefix
        self.pgdata = prefix or "/"
        self._names = {i.filename: i for i in zf.infolist()}
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_bytes = 0
        self._lock = threading.Lock()

    def _load(self, name: str) -> bytes:
        with self._lock:
            data = self._cache.get(name)
            if data is not None:
                self._cache.move_to_end(name)
                return data
        if name not in self._names:
            raise FileNotFoundError(name)
        data = self.zf.read(name)
        with self._lock:
            self._cache[name] = data
            self._cache_bytes += len(data)
            while self._cache_bytes > self.CACHE_BYTES and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= len(old)
        return data

    def exists(self, rel):
        return (self.prefix + rel) in self._names

    def size(self, rel):
        return self._names[self.prefix + rel].file_size

    def read(self, rel, offset=0, length=-1):
        data = self._load(self.prefix + rel)
        return data[offset:] if length < 0 else data[offset:offset + length]

    def listdir(self, rel):
        base = self.prefix + rel.rstrip("/") + "/"
        out = set()
        for n in self._names:
            if n.startswith(base) and n != base:
                out.add(n[len(base):].split("/", 1)[0])
        return sorted(x for x in out if x)

    def read_outside(self, rel):
        return self.zf.read(rel) if rel in self._names else None

    def input_hash(self):
        return {"algorithm": "sha256", "kind": "file", "value": sha256_file(self.input_path)}

    def close(self):
        self.zf.close()


def _zip_pgdata_prefixes(names: set[str]) -> list[str]:
    out = []
    for n in names:
        if n == "PG_VERSION" or n.endswith("/PG_VERSION"):
            prefix = n[: -len("PG_VERSION")]
            if prefix + "global/pg_control" in names:
                out.append(prefix)
    return sorted(out, key=len)


def open_source(path: str | os.PathLike) -> Source:
    p = Path(path)
    if not p.exists():
        raise SourceError(f"경로가 없습니다: {p}")
    if p.is_file():
        if not zipfile.is_zipfile(p):
            raise SourceError("파일 입력은 zip 만 지원합니다")
        zf = zipfile.ZipFile(p, "r")
        prefixes = _zip_pgdata_prefixes(set(zf.namelist()))
        if not prefixes:
            zf.close()
            raise SourceError("zip 안에서 PGDATA(PG_VERSION + global/pg_control)를 찾지 못했습니다")
        if len(prefixes) > 1:
            zf.close()
            raise SourceError(f"zip 안에 PGDATA 가 여러 개입니다: {prefixes}")
        return ZipSource(p, zf, prefixes[0])
    if _is_pgdata_dir(p):
        return DirSource(p, p)
    found = []
    base_depth = len(p.parts)
    for dirpath, dirs, _files in os.walk(p):
        d = Path(dirpath)
        if len(d.parts) - base_depth >= 4:
            dirs[:] = []
        if _is_pgdata_dir(d):
            found.append(d)
            dirs[:] = []
    if not found:
        raise SourceError("폴더 안에서 PGDATA(PG_VERSION + global/pg_control)를 찾지 못했습니다")
    if len(found) > 1:
        raise SourceError(f"PGDATA 가 여러 개입니다: {[str(x) for x in found]}")
    return DirSource(p, found[0])
