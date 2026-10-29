"""pg_xact (커밋 로그) — 트랜잭션 ID 의 커밋 상태.

출처: src/backend/access/transam/clog.c
  트랜잭션 하나에 2비트, 바이트당 4개 (CLOG_XACTS_PER_BYTE 4), 페이지 BLCKSZ,
  세그먼트당 SLRU_PAGES_PER_SEGMENT(32) 페이지. 세그먼트 이름은 16진수 4자리.
"""

from __future__ import annotations

import threading

IN_PROGRESS, COMMITTED, ABORTED, SUB_COMMITTED = 0, 1, 2, 3
STATUS_NAMES = {IN_PROGRESS: "in_progress", COMMITTED: "committed",
                ABORTED: "aborted", SUB_COMMITTED: "sub_committed"}

FROZEN_XID = 2
FIRST_NORMAL_XID = 3


class Clog:
    def __init__(self, source, blcksz: int = 8192, next_xid: int | None = None, clean_shutdown: bool = False):
        """next_xid · clean_shutdown: 정상 종료된 클러스터면 nextXid 보다 작은데 커밋 기록이 없는 xid 는
        끝난(중단된) 트랜잭션이다. 정상 종료가 아니면 커밋이 WAL 에만 있고 pg_xact 에 아직 안 쓰였을 수
        있으므로 판단하지 않는다 ('in_progress')."""
        self.source = source
        self.xacts_per_seg = blcksz * 4 * 32
        self.next_xid = next_xid
        self.clean_shutdown = clean_shutdown
        self._segs: dict[int, bytes | None] = {}
        self._lock = threading.Lock()
        self.wal = None  # WalIndex — pg_xact 에 결과가 없을 때 WAL 의 COMMIT/ABORT 기록으로 보충

    def _seg(self, segno: int) -> bytes | None:
        with self._lock:
            if segno not in self._segs:
                name = f"pg_xact/{segno:04X}"
                self._segs[segno] = self.source.read(name) if self.source.exists(name) else None
            return self._segs[segno]

    def status(self, xid: int) -> str:
        """'committed' / 'aborted' / 'uncommitted'(끝났지만 커밋 안 됨) / 'in_progress' / 'sub_committed' / 'unknown'."""
        s = self._raw_status(xid)
        if s in ("in_progress", "unknown") and self.wal is not None:
            x = self.wal.xact(xid)
            if x and x["status"] in ("committed", "aborted"):
                return x["status"]
        if s == "in_progress" and self.clean_shutdown and self.next_xid is not None and xid < self.next_xid:
            return "uncommitted"
        return s

    def _raw_status(self, xid: int) -> str:
        if xid == 0:
            return "invalid"
        if xid < FIRST_NORMAL_XID:  # BootstrapTransactionId(1), FrozenTransactionId(2)
            return "committed"
        segno, rem = divmod(xid, self.xacts_per_seg)
        data = self._seg(segno)
        byte = rem // 4
        if data is None or byte >= len(data):
            return "unknown"
        return STATUS_NAMES[(data[byte] >> ((xid % 4) * 2)) & 0x03]
