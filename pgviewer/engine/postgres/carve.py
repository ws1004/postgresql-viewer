"""힙 페이지의 빈 공간 카빙 — 라인 포인터가 가리키지 않는 영역에서 옛 튜플을 찾는다.

VACUUM/정리(pruning)는 라인 포인터를 UNUSED/DEAD 로 바꾸고 남은 튜플을 페이지 끝쪽으로 모으지만
(PageRepairFragmentation), 비워진 바이트를 0 으로 지우지는 않는다. 그래서 pd_lower ~ pd_upper 사이와
살아 있는 튜플 사이의 틈에 옛 튜플이 통째로 또는 일부 남는다.

후보 조건 (모두 만족해야 한다 — 오탐을 줄이기 위해 엄격하게):
  - 오프셋이 MAXALIGN(8) 경계 (튜플은 항상 MAXALIGN 위치에 놓인다)
  - t_hoff = MAXALIGN(23 + NULL 비트맵) 이고 natts 가 1 ~ 릴레이션 컬럼 수
  - xmin 이 3 이상이고 nextXid 보다 작다 (또는 frozen), xmax 는 0 이거나 nextXid 보다 작다
  - 옛 형식 플래그(HASOID_OLD, MOVED_IN/OFF)가 없다
  - ctid 의 라인 포인터 번호가 1 ~ 291 (MaxHeapTuplesPerPage, 8KB)
  - 모든 컬럼이 오류 없이 분해되고 틈 안에서 끝난다
"""

from __future__ import annotations

from .heap import (HEAP_HASNULL, HEAP_HASOID_OLD, HEAP_MOVED_IN, HEAP_MOVED_OFF, HEAP_XMAX_IS_MULTI,
                   HEAP_XMIN_FROZEN, LP_NORMAL, PAGE_HEADER_SIZE, TUPLE_HEADER_SIZE, line_pointers,
                   parse_tuple_header)

MAX_TUPLES_PER_PAGE = 291


def _maxalign(v: int) -> int:
    return (v + 7) & ~7


def free_gaps(page: bytes, h, blcksz: int) -> list[tuple[int, int]]:
    """라인 포인터가 가리키지 않는 [시작, 끝) 구간들."""
    used = [(0, max(h.lower, PAGE_HEADER_SIZE)), (h.special, blcksz)]
    for _lp, off, flags, length in line_pointers(page, h):
        if flags == LP_NORMAL and length and off + length <= blcksz:
            used.append((off, off + length))
    used.sort()
    gaps, cur = [], 0
    for s, e in used:
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e)
    if cur < blcksz:
        gaps.append((cur, blcksz))
    return [(s, e) for s, e in gaps if e - s >= TUPLE_HEADER_SIZE + 1]


def plausible_header(page: bytes, off: int, natts_max: int, next_xid: int | None):
    th = parse_tuple_header(page[off:off + TUPLE_HEADER_SIZE])
    natts = th.natts
    if not 1 <= natts <= natts_max:
        return None
    bitmap = (natts + 7) // 8 if th.infomask & HEAP_HASNULL else 0
    if th.hoff != _maxalign(TUPLE_HEADER_SIZE + bitmap):
        return None
    if th.infomask & (HEAP_HASOID_OLD | HEAP_MOVED_IN | HEAP_MOVED_OFF):
        return None
    frozen = th.infomask & HEAP_XMIN_FROZEN == HEAP_XMIN_FROZEN
    limit = next_xid if next_xid else 1 << 32
    if not frozen and not 3 <= th.xmin < limit:
        return None
    if th.xmax and not th.infomask & HEAP_XMAX_IS_MULTI and not th.xmax < limit:
        return None
    if not 1 <= th.ctid_pos <= MAX_TUPLES_PER_PAGE:
        return None
    return th


def carve_page(page: bytes, h, blcksz: int, natts_max: int, next_xid: int | None, try_deform):
    """try_deform(bytes, header) → 끝 오프셋(튜플 길이) 또는 None.
    돌려주는 값: [(페이지 내 오프셋, 튜플 바이트, 헤더)]"""
    out = []
    for gs, ge in free_gaps(page, h, blcksz):
        off = _maxalign(gs)
        while off + TUPLE_HEADER_SIZE < ge:
            th = plausible_header(page, off, natts_max, next_xid)
            if th is not None:
                length = try_deform(page[off:ge], th)
                if length:
                    out.append((off, page[off:off + length], th))
                    off = _maxalign(off + length)
                    continue
            off += 8
    return out
