"""힙 페이지 · 라인 포인터 · 튜플 헤더 · 행 상태 판정.

출처 (PostgreSQL 17):
  src/include/storage/bufpage.h   PageHeaderData (24바이트)
  src/include/storage/itemid.h    ItemIdData (lp_off:15, lp_flags:2, lp_len:15)
  src/include/access/htup_details.h  HeapTupleHeaderData (고정부 23바이트) 와 infomask 비트
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

PAGE_HEADER = struct.Struct("<IIHHHHHHI")
PAGE_HEADER_SIZE = 24
TUPLE_HEADER = struct.Struct("<IIIHHHHHB")
TUPLE_HEADER_SIZE = 23

LP_UNUSED, LP_NORMAL, LP_REDIRECT, LP_DEAD = 0, 1, 2, 3
LP_FLAG_NAMES = {0: "unused", 1: "normal", 2: "redirect", 3: "dead"}

# t_infomask
HEAP_HASNULL = 0x0001
HEAP_HASVARWIDTH = 0x0002
HEAP_HASEXTERNAL = 0x0004
HEAP_HASOID_OLD = 0x0008
HEAP_XMAX_KEYSHR_LOCK = 0x0010
HEAP_COMBOCID = 0x0020
HEAP_XMAX_EXCL_LOCK = 0x0040
HEAP_XMAX_LOCK_ONLY = 0x0080
HEAP_XMIN_COMMITTED = 0x0100
HEAP_XMIN_INVALID = 0x0200
HEAP_XMIN_FROZEN = HEAP_XMIN_COMMITTED | HEAP_XMIN_INVALID
HEAP_XMAX_COMMITTED = 0x0400
HEAP_XMAX_INVALID = 0x0800
HEAP_XMAX_IS_MULTI = 0x1000
HEAP_UPDATED = 0x2000
HEAP_MOVED_OFF = 0x4000
HEAP_MOVED_IN = 0x8000
HEAP_LOCK_MASK = HEAP_XMAX_EXCL_LOCK | HEAP_XMAX_KEYSHR_LOCK

# t_infomask2
HEAP_NATTS_MASK = 0x07FF
HEAP_KEYS_UPDATED = 0x2000
HEAP_HOT_UPDATED = 0x4000
HEAP_ONLY_TUPLE = 0x8000

INFOMASK_NAMES = [
    (HEAP_HASNULL, "HASNULL"), (HEAP_HASVARWIDTH, "HASVARWIDTH"), (HEAP_HASEXTERNAL, "HASEXTERNAL"),
    (HEAP_HASOID_OLD, "HASOID_OLD"), (HEAP_XMAX_KEYSHR_LOCK, "XMAX_KEYSHR_LOCK"),
    (HEAP_COMBOCID, "COMBOCID"), (HEAP_XMAX_EXCL_LOCK, "XMAX_EXCL_LOCK"),
    (HEAP_XMAX_LOCK_ONLY, "XMAX_LOCK_ONLY"), (HEAP_XMIN_COMMITTED, "XMIN_COMMITTED"),
    (HEAP_XMIN_INVALID, "XMIN_INVALID"), (HEAP_XMAX_COMMITTED, "XMAX_COMMITTED"),
    (HEAP_XMAX_INVALID, "XMAX_INVALID"), (HEAP_XMAX_IS_MULTI, "XMAX_IS_MULTI"),
    (HEAP_UPDATED, "UPDATED"), (HEAP_MOVED_OFF, "MOVED_OFF"), (HEAP_MOVED_IN, "MOVED_IN"),
]
INFOMASK2_NAMES = [(HEAP_KEYS_UPDATED, "KEYS_UPDATED"), (HEAP_HOT_UPDATED, "HOT_UPDATED"),
                   (HEAP_ONLY_TUPLE, "HEAP_ONLY_TUPLE")]


def flag_names(value: int, table) -> list[str]:
    return [name for bit, name in table if value & bit]


@dataclass(slots=True)
class PageHeader:
    lsn: int
    checksum: int
    flags: int
    lower: int
    upper: int
    special: int
    pagesize: int
    layout_version: int
    prune_xid: int

    @property
    def n_items(self) -> int:
        return max(0, (self.lower - PAGE_HEADER_SIZE) // 4)


def parse_page_header(page: bytes) -> PageHeader:
    hi, lo, checksum, flags, lower, upper, special, psv, prune = PAGE_HEADER.unpack_from(page, 0)
    return PageHeader((hi << 32) | lo, checksum, flags, lower, upper, special,
                      psv & 0xFF00, psv & 0x00FF, prune)


def page_problem(h: PageHeader, blcksz: int) -> str | None:
    """정상 힙 페이지가 아니면 이유를 돌려준다 (PageIsVerified 의 헤더 검사와 같은 조건)."""
    if h.upper == 0:
        return "new"  # 할당만 되고 초기화되지 않은 페이지
    if h.pagesize != blcksz or h.layout_version != 4:
        return f"pagesize/version {h.pagesize}/{h.layout_version}"
    if not (PAGE_HEADER_SIZE <= h.lower <= h.upper <= h.special <= blcksz):
        return f"lower/upper/special {h.lower}/{h.upper}/{h.special}"
    return None


def line_pointers(page: bytes, h: PageHeader):
    """(lp 번호(1부터), lp_off, lp_flags, lp_len)."""
    for i in range(h.n_items):
        v = struct.unpack_from("<I", page, PAGE_HEADER_SIZE + i * 4)[0]
        yield i + 1, v & 0x7FFF, (v >> 15) & 0x3, (v >> 17) & 0x7FFF


@dataclass(slots=True)
class TupleHeader:
    xmin: int
    xmax: int
    cid: int
    ctid_block: int
    ctid_pos: int
    infomask2: int
    infomask: int
    hoff: int

    @property
    def natts(self) -> int:
        return self.infomask2 & HEAP_NATTS_MASK

    def to_dict(self) -> dict:
        return {
            "xmin": self.xmin, "xmax": self.xmax, "cid": self.cid,
            "ctid": [self.ctid_block, self.ctid_pos],
            "infomask": self.infomask, "infomask_flags": flag_names(self.infomask, INFOMASK_NAMES),
            "infomask2": self.infomask2, "infomask2_flags": flag_names(self.infomask2, INFOMASK2_NAMES),
            "natts": self.natts, "hoff": self.hoff,
        }


def parse_tuple_header(tup: bytes) -> TupleHeader:
    xmin, xmax, cid, bi_hi, bi_lo, pos, im2, im, hoff = TUPLE_HEADER.unpack_from(tup, 0)
    return TupleHeader(xmin, xmax, cid, (bi_hi << 16) | bi_lo, pos, im2, im, hoff)


# ---------------------------------------------------------------- 행 상태

def xmin_status(h: TupleHeader, clog) -> str:
    if h.infomask & HEAP_XMIN_FROZEN == HEAP_XMIN_FROZEN:
        return "frozen"
    if h.infomask & HEAP_XMIN_COMMITTED:
        return "committed"
    if h.infomask & HEAP_XMIN_INVALID:
        return "aborted"
    return clog.status(h.xmin)


def xmax_locked_only(infomask: int) -> bool:
    # HEAP_XMAX_IS_LOCKED_ONLY()
    return bool(infomask & HEAP_XMAX_LOCK_ONLY) or (
        infomask & (HEAP_XMAX_IS_MULTI | HEAP_LOCK_MASK)) == HEAP_XMAX_EXCL_LOCK


def xmax_status(h: TupleHeader, clog) -> str:
    if h.xmax == 0 or h.infomask & HEAP_XMAX_INVALID:
        return "none"
    if xmax_locked_only(h.infomask):
        return "lock_only"
    if h.infomask & HEAP_XMAX_IS_MULTI:
        return "multi"  # MultiXact 해석은 미지원
    if h.infomask & HEAP_XMAX_COMMITTED:
        return "committed"
    return clog.status(h.xmax)


# 행 상태: live / deleted / updated(구버전) / aborted(INSERT 롤백·중단) / in_progress / unknown
#   unknown: 판정 근거가 부족하다 — pg_xact 에 없음, 서브트랜잭션, MultiXact xmax (pg_multixact 미해석)
def classify(h: TupleHeader, clog, block: int, lp: int) -> tuple[str, str, str]:
    smin = xmin_status(h, clog)
    smax = xmax_status(h, clog)
    if smin in ("aborted", "uncommitted"):
        return "aborted", smin, smax
    if smin == "in_progress":
        return "in_progress", smin, smax
    if smin not in ("committed", "frozen"):
        return "unknown", smin, smax
    if smax in ("multi", "unknown", "sub_committed"):
        return "unknown", smin, smax
    if smax == "committed":
        if h.ctid_block == block and h.ctid_pos == lp:
            return "deleted", smin, smax
        return "updated", smin, smax
    return "live", smin, smax
