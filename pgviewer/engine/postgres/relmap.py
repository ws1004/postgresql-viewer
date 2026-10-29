"""pg_filenode.map (RelMapFile) 해석 — 매핑 카탈로그의 OID → 파일 번호.

출처: src/backend/utils/cache/relmapper.c
  int32 magic (0x592717), int32 num_mappings, RelMapping mappings[MAX_MAPPINGS], pg_crc32c crc
  RelMapping = { Oid mapoid; RelFileNumber mapfilenumber; }
PG 16 부터 MAX_MAPPINGS 64 · 파일 524 바이트, 그 이전은 62 · 512 바이트(패딩 포함).
"""

from __future__ import annotations

import struct

from .crc32c import crc32c

MAGIC = 0x592717


def parse_relmap(data: bytes) -> dict:
    magic, num = struct.unpack_from("<ii", data, 0)
    if magic != MAGIC:
        raise ValueError(f"pg_filenode.map magic 불일치: {magic:#x}")
    if len(data) == 524:
        max_mappings = 64
    elif len(data) == 512:
        max_mappings = 62
    else:
        raise ValueError(f"pg_filenode.map 크기 미지원: {len(data)}")
    if not 0 <= num <= max_mappings:
        raise ValueError(f"num_mappings 범위 밖: {num}")
    mappings = {}
    for i in range(num):
        oid, filenode = struct.unpack_from("<II", data, 8 + i * 8)
        mappings[oid] = filenode
    crc_off = 8 + max_mappings * 8
    stored = struct.unpack_from("<I", data, crc_off)[0]
    return {"mappings": mappings, "crc_ok": stored == crc32c(data[:crc_off])}
