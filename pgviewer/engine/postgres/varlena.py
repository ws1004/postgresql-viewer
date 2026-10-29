"""가변 길이 값(varlena) 헤더 · TOAST 포인터 · 압축 해제(pglz, lz4).

출처 (PostgreSQL 17, little-endian):
  src/include/varatt.h        VARATT_IS_1B / 1B_E / 4B_U / 4B_C, varatt_external
  src/common/pg_lzcompress.c  pglz_decompress
  LZ4 block format            https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
"""

from __future__ import annotations

import struct

VARTAG_INDIRECT = 1
VARTAG_EXPANDED_RO = 2
VARTAG_EXPANDED_RW = 3
VARTAG_ONDISK = 18
# 태그별 포인터 크기 (VARTAG_SIZE). ONDISK = sizeof(varatt_external) = 16
_VARTAG_SIZE = {VARTAG_INDIRECT: 8, VARTAG_EXPANDED_RO: 8, VARTAG_EXPANDED_RW: 8, VARTAG_ONDISK: 16}

COMPRESS_PGLZ = 0
COMPRESS_LZ4 = 1
COMPRESS_NAMES = {COMPRESS_PGLZ: "pglz", COMPRESS_LZ4: "lz4"}


class VarlenaError(Exception):
    pass


def varlena_size(buf: bytes, off: int) -> int:
    """off 에서 시작하는 varlena 의 전체 길이(헤더 포함)."""
    b0 = buf[off]
    if b0 == 0x01:  # 1B_E: TOAST 포인터
        tag = buf[off + 1]
        if tag not in _VARTAG_SIZE:
            raise VarlenaError(f"알 수 없는 vartag {tag}")
        return 2 + _VARTAG_SIZE[tag]
    if b0 & 0x01:  # 1B: 짧은 헤더
        return (b0 >> 1) & 0x7F
    return (struct.unpack_from("<I", buf, off)[0] >> 2) & 0x3FFFFFFF


def varlena_kind(raw: bytes) -> str:
    b0 = raw[0]
    if b0 == 0x01:
        return "external"
    if b0 & 0x01:
        return "short"
    return "compressed" if (b0 & 0x03) == 0x02 else "plain"


def parse_external(raw: bytes) -> dict:
    tag = raw[1]
    if tag != VARTAG_ONDISK:
        raise VarlenaError(f"디스크에 있을 수 없는 TOAST 태그 {tag}")
    rawsize, extinfo, valueid, toastrelid = struct.unpack_from("<iIII", raw, 2)
    extsize = extinfo & 0x3FFFFFFF
    return {
        "rawsize": rawsize,          # 원래 크기 (헤더 4바이트 포함)
        "extsize": extsize,          # 저장된 크기 (헤더 제외)
        "method": extinfo >> 30,
        "compressed": extsize < rawsize - 4,  # VARATT_EXTERNAL_IS_COMPRESSED
        "valueid": valueid,
        "toastrelid": toastrelid,
    }


def decompress(method: int, data: bytes, rawsize: int) -> bytes:
    if method == COMPRESS_PGLZ:
        return pglz_decompress(data, rawsize)
    if method == COMPRESS_LZ4:
        return lz4_decompress(data, rawsize)
    raise VarlenaError(f"알 수 없는 압축 방식 {method}")


def pglz_decompress(src: bytes, rawsize: int) -> bytes:
    dst = bytearray()
    sp, end = 0, len(src)
    while sp < end and len(dst) < rawsize:
        ctrl = src[sp]
        sp += 1
        for _ in range(8):
            if sp >= end or len(dst) >= rawsize:
                break
            if ctrl & 1:
                if sp + 1 >= end:
                    raise VarlenaError("pglz: 태그가 잘렸습니다")
                length = (src[sp] & 0x0F) + 3
                offset = ((src[sp] & 0xF0) << 4) | src[sp + 1]
                sp += 2
                if length == 18:
                    if sp >= end:
                        raise VarlenaError("pglz: 길이 확장 바이트가 잘렸습니다")
                    length += src[sp]
                    sp += 1
                if offset == 0 or offset > len(dst):
                    raise VarlenaError("pglz: 잘못된 역참조 오프셋")
                length = min(length, rawsize - len(dst))
                start = len(dst) - offset
                if offset >= length:
                    dst += dst[start:start + length]
                else:
                    for i in range(length):  # 겹치는 복사 (반복 패턴)
                        dst.append(dst[start + i])
            else:
                dst.append(src[sp])
                sp += 1
            ctrl >>= 1
    if len(dst) != rawsize:
        raise VarlenaError(f"pglz: 해제 크기 {len(dst)} != 기대 {rawsize}")
    return bytes(dst)


def lz4_decompress(src: bytes, rawsize: int) -> bytes:
    dst = bytearray()
    i, n = 0, len(src)

    def ext_len(v: int) -> int:
        nonlocal i
        while True:
            if i >= n:
                raise VarlenaError("lz4: 길이 확장이 잘렸습니다")
            b = src[i]
            i += 1
            v += b
            if b != 255:
                return v

    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            lit = ext_len(lit)
        if i + lit > n or len(dst) + lit > rawsize:
            raise VarlenaError("lz4: 리터럴이 입력 또는 기대 크기를 넘습니다")
        dst += src[i:i + lit]
        i += lit
        if i >= n:
            break
        if i + 2 > n:
            raise VarlenaError("lz4: 오프셋이 잘렸습니다")
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        if offset == 0 or offset > len(dst):
            raise VarlenaError("lz4: 잘못된 오프셋")
        ml = token & 15
        if ml == 15:
            ml = ext_len(ml)
        ml += 4
        if len(dst) + ml > rawsize:
            raise VarlenaError("lz4: 해제 크기가 기대값을 넘습니다")
        start = len(dst) - offset
        if offset >= ml:
            dst += dst[start:start + ml]
        else:
            for k in range(ml):
                dst.append(dst[start + k])
    if len(dst) != rawsize:
        raise VarlenaError(f"lz4: 해제 크기 {len(dst)} != 기대 {rawsize}")
    return bytes(dst)


def detoast_inline(raw: bytes) -> bytes:
    """외부 TOAST 가 아닌 varlena 의 내용(헤더 뒤)을 돌려준다. 압축돼 있으면 해제한다."""
    kind = varlena_kind(raw)
    if kind == "short":
        return raw[1:]
    if kind == "plain":
        return raw[4:]
    if kind == "compressed":
        tcinfo = struct.unpack_from("<I", raw, 4)[0]
        return decompress(tcinfo >> 30, raw[8:], tcinfo & 0x3FFFFFFF)
    raise VarlenaError("외부 TOAST 값은 detoast_inline 으로 풀 수 없습니다")


def detoast_external_data(ext: dict, data: bytes) -> bytes:
    """TOAST 청크를 이어 붙인 data 를 원래 값으로 되돌린다."""
    if len(data) != ext["extsize"]:
        raise VarlenaError(f"TOAST 청크 합계 {len(data)} != extsize {ext['extsize']}")
    if ext["compressed"]:
        tcinfo = struct.unpack_from("<I", data, 0)[0]
        return decompress(tcinfo >> 30, data[4:], tcinfo & 0x3FFFFFFF)
    return data
