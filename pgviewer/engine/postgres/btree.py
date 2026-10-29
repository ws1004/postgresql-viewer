"""B-tree 인덱스 리프 페이지 — 인덱스 키와 그 키가 가리키는 힙 위치(TID).

행이 지워지고 힙에서 정리돼도 인덱스 엔트리는 인덱스 VACUUM 전까지 남는다. 그 키 값이 흔적이 된다.

출처 (PostgreSQL 17):
  src/include/access/nbtree.h   BTPageOpaqueData(16바이트, special 영역), BTP_LEAF, P_FIRSTDATAKEY,
                                포스팅 리스트(BT_IS_POSTING, BT_OFFSET_MASK)
  src/include/access/itup.h     IndexTupleData (t_tid 6 + t_info 2), INDEX_SIZE_MASK, INDEX_NULL_MASK,
                                INDEX_ALT_TID_MASK, 데이터 시작 = MAXALIGN(8 [+ 비트맵 4])
"""

from __future__ import annotations

import struct

from .heap import LP_DEAD, LP_NORMAL, line_pointers, page_problem, parse_page_header

BTP_LEAF, BTP_ROOT, BTP_DELETED, BTP_META, BTP_HALF_DEAD = 0x01, 0x02, 0x04, 0x08, 0x10
INDEX_SIZE_MASK, INDEX_ALT_TID_MASK, INDEX_VAR_MASK, INDEX_NULL_MASK = 0x1FFF, 0x2000, 0x4000, 0x8000
BT_OFFSET_MASK, BT_IS_POSTING = 0x0FFF, 0x2000


def leaf_items(page: bytes, blcksz: int):
    """리프 페이지의 데이터 아이템: (lp, killed, 키 바이트가 담긴 튜플, 데이터 시작, hasnulls, [힙 TID…])."""
    h = parse_page_header(page)
    if page_problem(h, blcksz) or blcksz - h.special != 16:
        return
    _prev, nxt, _level, flags, _cycle = struct.unpack_from("<IIIHH", page, h.special)
    if not flags & BTP_LEAF or flags & (BTP_DELETED | BTP_META | BTP_HALF_DEAD):
        return
    first = 1 if nxt == 0 else 2  # 가장 오른쪽 페이지가 아니면 1번은 high key
    for lp, off, lpflags, length in line_pointers(page, h):
        if lp < first or lpflags not in (LP_NORMAL, LP_DEAD) or length < 8 or off + length > blcksz:
            continue
        tup = page[off:off + length]
        bi_hi, bi_lo, posid, tinfo = struct.unpack_from("<HHHH", tup, 0)
        size = tinfo & INDEX_SIZE_MASK
        if size < 8 or size > length:
            continue
        tup = tup[:size]
        hasnulls = bool(tinfo & INDEX_NULL_MASK)
        data_off = 16 if hasnulls else 8  # MAXALIGN(IndexTupleData [+ IndexAttributeBitMapData])
        block = (bi_hi << 16) | bi_lo
        if tinfo & INDEX_ALT_TID_MASK:
            if not posid & BT_IS_POSTING:
                continue  # pivot 튜플 (리프에서는 high key 뿐)
            n = posid & BT_OFFSET_MASK
            pstart = block  # 포스팅 리스트 시작 오프셋
            if pstart + 6 * n > len(tup):
                continue
            tids = []
            for k in range(n):
                h1, l1, p1 = struct.unpack_from("<HHH", tup, pstart + 6 * k)
                tids.append(((h1 << 16) | l1, p1))
            key = tup[:pstart]
        else:
            tids = [(block, posid)]
            key = tup
        yield lp, lpflags == LP_DEAD, key, data_off, hasnulls, tids
