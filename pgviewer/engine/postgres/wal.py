"""WAL(pg_wal) 레코드 해석 — 커밋 시각, 힙 변경 이벤트, WAL 에만 남은 튜플.

출처 (PostgreSQL 17 · 18):
  src/include/access/xlog_internal.h   XLogPageHeaderData (짧은 24 / 긴 40바이트), XLOG_PAGE_MAGIC (PG17 0xD116 · PG18 0xD118)
  src/include/access/xlogrecord.h      XLogRecord(24) · XLogRecordBlockHeader · BlockImageHeader · 블록 ID 252~255
  src/backend/access/transam/xlogreader.c  DecodeXLogRecord, RestoreBlockImage
  src/include/access/heapam_xlog.h     xl_heap_insert/delete/update/multi_insert, xl_heap_header
  src/include/access/xact.h            xl_xact_commit/abort, xinfo
  src/include/access/rmgrlist.h        리소스 매니저 ID

파일 이름이 아니라 페이지 헤더의 xlp_pageaddr 로 정렬한다. 재활용된 세그먼트 파일에는 이름과 다른
(더 오래된) 주소의 WAL 이 남아 있다 — 그 내용도 그대로 읽는다.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .crc32c import crc32c
from .heap import (TUPLE_HEADER_SIZE, LP_NORMAL, line_pointers, page_problem, parse_page_header,
                   parse_tuple_header)
from .varlena import VarlenaError, lz4_decompress, pglz_decompress

PAGE_MAGIC = {0xD118: 18, 0xD116: 17, 0xD113: 16, 0xD110: 15, 0xD10D: 14}
# PG 17 과 PG 18 은 XLogRecord · BKPIMAGE 플래그 · heap/heap2 opcode 가 동일하다 (xlog_internal.h 매직만 다름).
# 그 이전 메이저는 플래그·opcode 가 달라 해석하지 않는다.
SUPPORTED_MAGICS = frozenset({0xD116, 0xD118})
XLP_FIRST_IS_CONTRECORD = 0x0001
XLP_LONG_HEADER = 0x0002
SHORT_PHD, LONG_PHD = 24, 40
REC_HDR = struct.Struct("<IIQBBxxI")  # xl_tot_len, xl_xid, xl_prev, xl_info, xl_rmid, pad, xl_crc
REC_HDR_SIZE = 24

RMGR = ["XLOG", "Transaction", "Storage", "CLOG", "Database", "Tablespace", "MultiXact", "RelMap",
        "Standby", "Heap2", "Heap", "Btree", "Hash", "Gin", "Gist", "Sequence", "SPGist", "BRIN",
        "CommitTs", "ReplicationOrigin", "Generic", "LogicalMessage"]
RM_XACT, RM_HEAP2, RM_HEAP = 1, 9, 10

BKPBLOCK_FORK_MASK, BKPBLOCK_HAS_IMAGE, BKPBLOCK_HAS_DATA, BKPBLOCK_WILL_INIT, BKPBLOCK_SAME_REL = (
    0x0F, 0x10, 0x20, 0x40, 0x80)
BKPIMAGE_HAS_HOLE, BKPIMAGE_APPLY, BKPIMAGE_COMPRESS_PGLZ, BKPIMAGE_COMPRESS_LZ4, BKPIMAGE_COMPRESS_ZSTD = (
    0x01, 0x02, 0x04, 0x08, 0x10)

HEAP_OPS = {0x00: "INSERT", 0x10: "DELETE", 0x20: "UPDATE", 0x30: "TRUNCATE", 0x40: "HOT_UPDATE",
            0x50: "CONFIRM", 0x60: "LOCK", 0x70: "INPLACE"}
HEAP2_OPS = {0x00: "REWRITE", 0x10: "PRUNE_ON_ACCESS", 0x20: "PRUNE_VACUUM_SCAN", 0x30: "PRUNE_VACUUM_CLEANUP",
             0x40: "VISIBLE", 0x50: "MULTI_INSERT", 0x60: "LOCK_UPDATED", 0x70: "NEW_CID"}
XLOG_HEAP_INIT_PAGE = 0x80
# heapam_xlog.h: OLD_ALL_VISIBLE_CLEARED 1<<0 · NEW_ALL_VISIBLE_CLEARED 1<<1 · CONTAINS_OLD_TUPLE 1<<2 ·
#                CONTAINS_OLD_KEY 1<<3 · CONTAINS_NEW_TUPLE 1<<4 · PREFIX_FROM_OLD 1<<5 · SUFFIX_FROM_OLD 1<<6
XLH_UPDATE_PREFIX_FROM_OLD = 1 << 5
XLH_UPDATE_SUFFIX_FROM_OLD = 1 << 6
XACT_OPS = {0x00: "COMMIT", 0x10: "PREPARE", 0x20: "ABORT", 0x30: "COMMIT_PREPARED", 0x40: "ABORT_PREPARED"}
XLOG_XACT_HAS_INFO = 0x80
XACT_XINFO_HAS_DBINFO = 1 << 0
XACT_XINFO_HAS_SUBXACTS = 1 << 1
# heapam_xlog.h: ALL_VISIBLE_CLEARED 1<<0 · CONTAINS_OLD_TUPLE 1<<1 · CONTAINS_OLD_KEY 1<<2 · IS_SUPER 1<<3
XLH_DELETE_IS_SUPER = 1 << 3

PG_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def lsn_str(v: int) -> str:
    return f"{v >> 32:X}/{v & 0xFFFFFFFF:X}"


def pg_time(us: int) -> str | None:
    try:
        return (PG_EPOCH + timedelta(microseconds=us)).isoformat()
    except OverflowError:
        return None


def maxalign(v: int) -> int:
    return (v + 7) & ~7


@dataclass
class Block:
    id: int
    fork: int
    flags: int
    data_len: int
    spc: int = 0
    db: int = 0
    rel: int = 0
    blkno: int = 0
    image: bytes | None = None  # 복원된 전체 페이지 (FPI)
    image_error: str | None = None
    data: bytes = b""


@dataclass
class Record:
    lsn: int
    tot_len: int
    xid: int
    info: int
    rmid: int
    blocks: list[Block] = field(default_factory=list)
    main: bytes = b""

    @property
    def rmgr(self) -> str:
        return RMGR[self.rmid] if self.rmid < len(RMGR) else f"rmgr{self.rmid}"


class _Run:
    """주소가 이어지는 페이지 묶음을 '헤더를 뺀 본문' 하나의 바이트열로 다룬다."""

    def __init__(self, pages: list[tuple[int, bytes, int, int]], page_size: int):
        # pages: (pageaddr, page bytes, info, rem_len)
        self.pages = pages
        self.ps = page_size
        self.start = pages[0][0]
        self.hdr = [LONG_PHD if p[2] & XLP_LONG_HEADER else SHORT_PHD for p in pages]
        self.body = b"".join(p[1][h:] for p, h in zip(pages, self.hdr))
        self.body_start = []
        acc = 0
        for h in self.hdr:
            self.body_start.append(acc)
            acc += page_size - h

    def pos(self, lsn: int) -> int | None:
        i = (lsn - self.start) // self.ps
        if not 0 <= i < len(self.pages):
            return None
        off = lsn - self.pages[i][0]
        if off < self.hdr[i]:
            off = self.hdr[i]
        return self.body_start[i] + off - self.hdr[i]

    def lsn(self, pos: int) -> int:
        # body_start 에서 pos 를 포함하는 페이지 찾기
        lo, hi = 0, len(self.pages) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.body_start[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return self.pages[lo][0] + self.hdr[lo] + (pos - self.body_start[lo])

    def next_start(self, i: int) -> int | None:
        """i 번째 페이지에서 새 레코드가 시작될 수 있는 첫 LSN (이어지는 레코드 조각을 건너뛴다)."""
        addr, _, info, rem = self.pages[i]
        body = self.ps - self.hdr[i]
        if info & XLP_FIRST_IS_CONTRECORD:
            if maxalign(rem) >= body:
                return None
            return addr + self.hdr[i] + maxalign(rem)
        return addr + self.hdr[i]


class WalReader:
    def __init__(self, source, blcksz: int = 8192, verify_crc: bool = True):
        self.source = source
        self.blcksz = blcksz
        self.verify_crc = verify_crc
        self.stats = {"files": [], "pages": 0, "records": 0, "crc_errors": 0, "decode_errors": 0,
                      "handle_errors": 0, "image_errors": 0, "runs": [], "by_rmgr": {}}
        self.page_size = 8192

    def _pages(self):
        """pg_wal 의 모든 페이지를 (타임라인, 주소) 로 모은다. 같은 주소가 여러 번 나오면 더 높은 타임라인을 쓴다
        (승격 뒤의 WAL 이 앞 타임라인의 같은 주소를 대신한다)."""
        pages: dict[int, tuple] = {}
        ps = 8192
        names = [n for n in self.source.listdir("pg_wal") if len(n) == 24 and all(c in "0123456789ABCDEF" for c in n)]
        conflicts = 0
        for name in sorted(names):
            data = self.source.read("pg_wal/" + name)
            if len(data) >= LONG_PHD:
                magic, info = struct.unpack_from("<HH", data, 0)
                if magic in PAGE_MAGIC and info & XLP_LONG_HEADER:
                    ps = struct.unpack_from("<I", data, 36)[0] or 8192
                    if magic not in SUPPORTED_MAGICS:
                        raise ValueError(f"WAL 페이지 magic {magic:#06x} (PG {PAGE_MAGIC[magic]}) — "
                                         f"PG 17(0xD116) · PG 18(0xD118) 형식만 해석합니다")
            used = 0
            first_addr = None
            for off in range(0, len(data) - ps + 1, ps):
                magic, info, tli, addr, rem = struct.unpack_from("<HHIQI", data, off)
                if magic not in SUPPORTED_MAGICS or addr % ps:
                    continue
                used += 1
                first_addr = addr if first_addr is None else first_addr
                cur = pages.get(addr)
                if cur is not None:
                    conflicts += 1
                    if cur[5] >= tli:
                        continue
                pages[addr] = (addr, data[off:off + ps], info, rem, name, tli)
            # 파일 이름 = 타임라인(8) + log(8) + seg(8) → 절대 세그먼트 번호 = log × (4GiB / 세그먼트 크기) + seg
            seg_size = len(data) or 1
            name_seg = int(name[8:16], 16) * (0x100000000 // seg_size) + int(name[16:], 16)
            self.stats["files"].append({"name": name, "pages": used,
                                        "first_lsn": lsn_str(first_addr) if first_addr is not None else None,
                                        "recycled": first_addr is not None and first_addr // seg_size != name_seg})
        self.stats["page_conflicts"] = conflicts
        self.page_size = ps
        return [pages[a] for a in sorted(pages)]

    def records(self):
        """레코드를 LSN 순으로. 연속이 끊기는 곳(구간 경계·다시 맞추기)에서는 None 을 내보낸다."""
        pages = self._pages()
        self.stats["pages"] = len(pages)
        runs, cur = [], []
        for p in pages:
            if cur and p[0] != cur[-1][0] + self.page_size:
                runs.append(cur)
                cur = []
            cur.append(p)
        if cur:
            runs.append(cur)
        for run_pages in runs:
            run = _Run([(p[0], p[1], p[2], p[3]) for p in run_pages], self.page_size)
            self.stats["runs"].append({"start": lsn_str(run.start),
                                       "end": lsn_str(run_pages[-1][0] + self.page_size),
                                       "pages": len(run_pages)})
            yield None
            yield from self._read_run(run)

    def _resync(self, run: _Run, page_i: int):
        while page_i < len(run.pages):
            lsn = run.next_start(page_i)
            if lsn is not None:
                return lsn
            page_i += 1
        return None

    def _read_run(self, run: _Run):
        lsn = self._resync(run, 0)  # 첫 페이지가 통째로 앞 레코드의 이어짐이면 다음 페이지에서 시작
        while lsn is not None:
            pos = run.pos(lsn)
            ok = False
            if pos is not None and pos + REC_HDR_SIZE <= len(run.body):
                tot, xid, prev, info, rmid, crc = REC_HDR.unpack_from(run.body, pos)
                if tot >= REC_HDR_SIZE and pos + tot <= len(run.body):
                    raw = run.body[pos:pos + tot]
                    if not self.verify_crc or crc32c(raw[REC_HDR_SIZE:] + raw[:20]) == crc:
                        ok = True
                        self.stats["records"] += 1
                        name = RMGR[rmid] if rmid < len(RMGR) else str(rmid)
                        self.stats["by_rmgr"][name] = self.stats["by_rmgr"].get(name, 0) + 1
                        try:
                            rec = self._decode(lsn, raw, tot, xid, info, rmid)
                        except Exception as e:
                            self._error("decode", lsn, e)
                            rec = None
                        if rec is not None:
                            yield rec
                        lsn = maxalign(run.lsn(pos + tot))
                        i = (lsn - run.start) // run.ps
                        if 0 <= i < len(run.pages) and lsn - run.pages[i][0] < run.hdr[i]:
                            lsn = run.pages[i][0] + run.hdr[i]  # 페이지 헤더 위치면 본문 시작으로
                    elif tot:
                        self.stats["crc_errors"] += 1
            if ok:
                continue
            # 다시 맞추기 (0 으로 채워진 WAL 끝, CRC 불일치 등): 다음 페이지의 첫 레코드 시작점
            yield None
            lsn = self._resync(run, (lsn - run.start) // run.ps + 1)

    def _error(self, kind: str, lsn: int, e: Exception):
        self.stats[kind + "_errors"] += 1
        samples = self.stats.setdefault("error_samples", [])
        if len(samples) < 20:
            samples.append(f"{kind} @ {lsn_str(lsn)}: {type(e).__name__}: {e}")

    def _decode(self, lsn, raw, tot, xid, info, rmid) -> Record:
        rec = Record(lsn, tot, xid, info, rmid)
        pos = REC_HDR_SIZE
        datatotal = 0
        main_len = 0
        last = (0, 0, 0)
        blocks: list[Block] = []
        images: list[tuple[Block, int, int, int, int]] = []
        while tot - pos > datatotal:
            bid = raw[pos]
            pos += 1
            if bid == 255:
                main_len = raw[pos]
                pos += 1
                break
            if bid == 254:
                main_len = struct.unpack_from("<I", raw, pos)[0]
                pos += 4
                break
            if bid == 253:
                pos += 2
                continue
            if bid == 252:
                pos += 4
                continue
            if bid > 32:
                raise ValueError(f"블록 ID {bid}")
            flags, dlen = struct.unpack_from("<BH", raw, pos)
            pos += 3
            b = Block(bid, flags & BKPBLOCK_FORK_MASK, flags, dlen)
            datatotal += dlen
            if flags & BKPBLOCK_HAS_IMAGE:
                ilen, hole_off, binfo = struct.unpack_from("<HHB", raw, pos)
                pos += 5
                compressed = binfo & (BKPIMAGE_COMPRESS_PGLZ | BKPIMAGE_COMPRESS_LZ4 | BKPIMAGE_COMPRESS_ZSTD)
                if binfo & BKPIMAGE_HAS_HOLE and compressed:
                    hole_len = struct.unpack_from("<H", raw, pos)[0]
                    pos += 2
                elif binfo & BKPIMAGE_HAS_HOLE:
                    hole_len = self.blcksz - ilen
                else:
                    hole_len = 0
                datatotal += ilen
                images.append((b, ilen, hole_off, hole_len, binfo))
            if not flags & BKPBLOCK_SAME_REL:
                last = struct.unpack_from("<III", raw, pos)
                pos += 12
            b.spc, b.db, b.rel = last
            b.blkno = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
            blocks.append(b)
        # 데이터 영역: 블록 순서대로 (이미지 → 데이터), 마지막에 main data
        img_by_block = {id(x[0]): x for x in images}
        for b in blocks:
            if b.flags & BKPBLOCK_HAS_IMAGE:
                _, ilen, hole_off, hole_len, binfo = img_by_block[id(b)]
                img = raw[pos:pos + ilen]
                pos += ilen
                try:
                    b.image = self._restore_image(img, hole_off, hole_len, binfo)
                except Exception as e:
                    b.image_error = str(e)
                    self.stats["image_errors"] += 1
            if b.flags & BKPBLOCK_HAS_DATA:
                b.data = raw[pos:pos + b.data_len]
                pos += b.data_len
        rec.main = raw[pos:pos + main_len]
        rec.blocks = blocks
        return rec

    def _restore_image(self, img: bytes, hole_off: int, hole_len: int, binfo: int) -> bytes:
        want = self.blcksz - hole_len
        if binfo & BKPIMAGE_COMPRESS_PGLZ:
            img = pglz_decompress(img, want)
        elif binfo & BKPIMAGE_COMPRESS_LZ4:
            img = lz4_decompress(img, want)
        elif binfo & BKPIMAGE_COMPRESS_ZSTD:
            raise VarlenaError("zstd 로 압축된 전체 페이지 이미지는 지원하지 않습니다")
        if len(img) != want:
            raise VarlenaError(f"페이지 이미지 크기 {len(img)} != {want}")
        return img[:hole_off] + bytes(hole_len) + img[hole_off:]


# ------------------------------------------------------------ 해석 결과

@dataclass
class WalTuple:
    db: int
    rel: int          # relfilenumber
    block: int
    offnum: int
    lsn: int
    source: str       # fpi / insert / update / multi_insert
    record_xid: int
    data: bytes       # 23바이트 헤더를 포함한 튜플 전체


class WalIndex:
    """WAL 전체를 한 번 읽어 만든 색인."""

    def __init__(self, source, blcksz: int = 8192, verify_crc: bool = True):
        self.reader = WalReader(source, blcksz, verify_crc)
        self.xacts: dict[int, dict] = {}            # xid → {status, time, lsn}
        self.events: dict[tuple[int, int], list] = {}  # (db, relfilenumber) → [event]
        self.tuples: dict[tuple[int, int], list[WalTuple]] = {}
        # (db, rel, block) → {offnum: 가장 최근 튜플} — UPDATE 의 앞뒤 공유 바이트 복원용. 페이지 단위로 비운다
        self._latest: dict[tuple, dict[int, bytes]] = {}
        self.first_lsn = self.last_lsn = None
        self.max_xid = 0
        for rec in self.reader.records():
            if rec is None:
                # 연속이 끊겼다 — 그 사이의 변경을 모르므로 UPDATE 복원용 '최근 튜플'을 버린다
                self._latest.clear()
                continue
            if self.first_lsn is None:
                self.first_lsn = rec.lsn
            self.last_lsn = rec.lsn
            self.max_xid = max(self.max_xid, rec.xid)
            try:
                self._handle(rec)
            except Exception as e:  # 레코드 하나 때문에 WAL 전체를 버리지 않는다
                self.reader._error("handle", rec.lsn, e)
        self.stats = self.reader.stats
        self.stats["xacts"] = len(self.xacts)
        self.stats["first_lsn"] = lsn_str(self.first_lsn) if self.first_lsn is not None else None
        self.stats["last_lsn"] = lsn_str(self.last_lsn) if self.last_lsn is not None else None
        self.stats["max_xid"] = self.max_xid
        times = [x["time"] for x in self.xacts.values() if x.get("time")]
        self.stats["time_range"] = [min(times), max(times)] if times else None

    # ---- 레코드 처리
    def _handle(self, rec: Record):
        for b in rec.blocks:
            if b.image is not None and b.fork == 0:
                self._page_image(rec, b)
        if rec.rmid == RM_XACT:
            self._xact(rec)
        elif rec.rmid == RM_HEAP:
            self._heap(rec)
        elif rec.rmid == RM_HEAP2:
            self._heap2(rec)

    def _event(self, rec: Record, b: Block, op: str, **kw):
        ev = {"lsn": lsn_str(rec.lsn), "lsn_int": rec.lsn, "xid": rec.xid, "op": op, "block": b.blkno, **kw}
        self.events.setdefault((b.db, b.rel), []).append(ev)

    def _add_tuple(self, rec: Record, b: Block, offnum: int, data: bytes, source: str):
        t = WalTuple(b.db, b.rel, b.blkno, offnum, rec.lsn, source, rec.xid, data)
        self.tuples.setdefault((b.db, b.rel), []).append(t)
        self._latest.setdefault((b.db, b.rel, b.blkno), {})[offnum] = data

    def _forget_block(self, b: Block):
        self._latest.pop((b.db, b.rel, b.blkno), None)

    def _page_image(self, rec: Record, b: Block):
        page = b.image
        h = parse_page_header(page)
        if page_problem(h, len(page)):
            return
        if h.special != len(page):  # 힙 페이지가 아니다 (인덱스 등 — special 영역이 있다)
            return
        self._forget_block(b)  # 이 페이지의 '최근 튜플'은 이미지 내용으로 바꾼다
        n = 0
        for lp, off, flags, length in line_pointers(page, h):
            if flags == LP_NORMAL and length >= TUPLE_HEADER_SIZE and off + length <= len(page):
                self._add_tuple(rec, b, lp, page[off:off + length], "fpi")
                n += 1
        self._event(rec, b, "FPI", tuples=n, record=f"{rec.rmgr}")

    @staticmethod
    def _synth(xid: int, block: int, offnum: int, infomask2: int, infomask: int, hoff: int, rest: bytes) -> bytes:
        # WAL 은 튜플 헤더의 고정부(23바이트)를 싣지 않는다 — xmin 은 레코드의 xid, ctid 는 자기 위치로 채운다
        hdr = struct.pack("<IIIHHHHHB", xid, 0, 0, block >> 16, block & 0xFFFF, offnum, infomask2, infomask, hoff)
        return hdr + rest

    def _heap(self, rec: Record):
        op = HEAP_OPS.get(rec.info & 0x70, hex(rec.info))
        b0 = rec.blocks[0] if rec.blocks else None
        if b0 is None:
            return
        m = rec.main
        if rec.info & XLOG_HEAP_INIT_PAGE and b0.image is None:
            self._forget_block(b0)  # 페이지를 새로 초기화하는 레코드
        if op == "INSERT" and len(m) >= 3:
            offnum = struct.unpack_from("<H", m, 0)[0]
            if b0.data and len(b0.data) >= 5:
                im2, im, hoff = struct.unpack_from("<HHB", b0.data, 0)
                self._add_tuple(rec, b0, offnum, self._synth(rec.xid, b0.blkno, offnum, im2, im, hoff, b0.data[5:]),
                                "insert")
            self._event(rec, b0, "INSERT", offnum=offnum, init=bool(rec.info & XLOG_HEAP_INIT_PAGE))
        elif op == "DELETE" and len(m) >= 8:
            xmax, offnum, _infobits, flags = struct.unpack_from("<IHBB", m, 0)
            # XLH_DELETE_IS_SUPER: 충돌로 취소된 투기적 INSERT(ON CONFLICT) — 사용자가 지운 행이 아니다
            self._event(rec, b0, "DELETE", offnum=offnum, xmax=xmax, super=bool(flags & XLH_DELETE_IS_SUPER))
        elif op in ("UPDATE", "HOT_UPDATE") and len(m) >= 14:
            old_xmax, old_off, _oinfo, flags, _new_xmax, new_off = struct.unpack_from("<IHBBIH", m, 0)
            old_b = rec.blocks[1] if len(rec.blocks) > 1 else b0
            if b0.data:
                d = b0.data
                p = 0
                prefix = suffix = 0
                if flags & XLH_UPDATE_PREFIX_FROM_OLD:
                    prefix = struct.unpack_from("<H", d, p)[0]
                    p += 2
                if flags & XLH_UPDATE_SUFFIX_FROM_OLD:
                    suffix = struct.unpack_from("<H", d, p)[0]
                    p += 2
                im2, im, hoff = struct.unpack_from("<HHB", d, p)
                p += 5
                body = d[p:]
                old = self._latest.get((old_b.db, old_b.rel, old_b.blkno), {}).get(old_off)
                new = None
                source = "update"
                if prefix or suffix:
                    # heap_xlog_update 와 같은 방식으로 복원 — 옛 튜플을 WAL 안에서 이어서 알고 있을 때만
                    if old is not None and len(old) >= old[22] + prefix + suffix and hoff >= TUPLE_HEADER_SIZE:
                        source = "update_reconstructed"
                        ohoff = old[22]
                        bits = hoff - TUPLE_HEADER_SIZE
                        new = body[:bits] + old[ohoff:ohoff + prefix] + body[bits:] + \
                            (old[len(old) - suffix:] if suffix else b"")
                else:
                    new = body
                if new is not None:
                    self._add_tuple(rec, b0, new_off, self._synth(rec.xid, b0.blkno, new_off, im2, im, hoff, new),
                                    source)
            self._event(rec, old_b, op, offnum=old_off, xmax=old_xmax, new_block=b0.blkno, new_offnum=new_off)
        elif op in ("LOCK", "INPLACE", "TRUNCATE", "CONFIRM"):
            offnum = struct.unpack_from("<H", m, 4 if op == "LOCK" else 0)[0] if len(m) >= 6 else None
            self._event(rec, b0, op, offnum=offnum)

    def _heap2(self, rec: Record):
        op = HEAP2_OPS.get(rec.info & 0x70, hex(rec.info))
        b0 = rec.blocks[0] if rec.blocks else None
        if b0 is None:
            return
        if op == "MULTI_INSERT" and len(rec.main) >= 4:
            ntup = struct.unpack_from("<H", rec.main, 2)[0]
            init = bool(rec.info & XLOG_HEAP_INIT_PAGE)
            if not init and len(rec.main) < 4 + 2 * ntup:
                return
            offs = list(range(1, ntup + 1)) if init else list(struct.unpack_from(f"<{ntup}H", rec.main, 4))
            d, p = b0.data, 0
            for i in range(ntup if d else 0):
                p = (p + 1) & ~1  # SHORTALIGN
                if p + 7 > len(d):
                    break
                dlen, im2, im, hoff = struct.unpack_from("<HHHB", d, p)
                p += 7
                self._add_tuple(rec, b0, offs[i],
                                self._synth(rec.xid, b0.blkno, offs[i], im2, im, hoff, d[p:p + dlen]), "multi_insert")
                p += dlen
            self._event(rec, b0, "MULTI_INSERT", count=ntup)
        elif op.startswith("PRUNE"):
            self._event(rec, b0, op)

    def _xact(self, rec: Record):
        op = XACT_OPS.get(rec.info & 0x70)
        # PREPARE 의 main 은 TwoPhaseFileHeader 라 시각이 아니고, COMMIT/ABORT_PREPARED 의 대상 xid 는
        # xinfo 뒤쪽(HAS_TWOPHASE)에 있다 — 해석하지 않는다 (잘못된 시각을 붙이지 않기 위해)
        if op not in ("COMMIT", "ABORT") or len(rec.main) < 8:
            return
        t = pg_time(struct.unpack_from("<q", rec.main, 0)[0])
        status = "committed" if op.startswith("COMMIT") else "aborted" if op.startswith("ABORT") else "prepared"
        xids = [rec.xid]
        if rec.info & XLOG_XACT_HAS_INFO and len(rec.main) >= 12:
            xinfo = struct.unpack_from("<I", rec.main, 8)[0]
            p = 12
            if xinfo & XACT_XINFO_HAS_DBINFO:
                p += 8
            if xinfo & XACT_XINFO_HAS_SUBXACTS and len(rec.main) >= p + 4:
                n = struct.unpack_from("<i", rec.main, p)[0]
                if 0 < n < 100000 and len(rec.main) >= p + 4 + 4 * n:
                    xids += list(struct.unpack_from(f"<{n}I", rec.main, p + 4))
        for x in xids:
            if x:
                self.xacts[x] = {"status": status, "time": t, "lsn": lsn_str(rec.lsn), "top": rec.xid}

    # ---- 조회
    def xact(self, xid: int) -> dict | None:
        return self.xacts.get(xid)
