"""global/pg_control (ControlFileData) 해석.

출처: src/include/catalog/pg_control.h (PostgreSQL 17, PG_CONTROL_VERSION 1700).
오프셋은 x86_64 정렬 기준이며, 샘플(PG 17.x)에서 값의 타당성과 CRC 일치로 확인했다.

다른 버전은 앞부분(시스템 식별자·버전·상태)과, floatFormat 상수(1234567.0)를 기준점으로 찾은
뒷부분(blcksz 등)만 해석한다. 이 기준점 방식은 버전 간 CheckPoint 구조 차이에 영향을 받지 않는다.
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone

from .crc32c import crc32c

FLOAT_FORMAT = struct.pack("<d", 1234567.0)

DB_STATES = {
    0: "starting up",
    1: "shut down",
    2: "shut down in recovery",
    3: "shutting down",
    4: "in crash recovery",
    5: "in archive recovery",
    6: "in production",
}

WAL_LEVELS = {0: "minimal", 1: "replica", 2: "logical"}

# PG 17 (1700) 의 앞부분: (이름, 오프셋, struct 형식)
_FIELDS_1700 = [
    ("system_identifier", 0, "Q"),
    ("pg_control_version", 8, "I"),
    ("catalog_version_no", 12, "I"),
    ("state", 16, "i"),
    ("time", 24, "q"),
    ("checkpoint", 32, "Q"),
    # CheckPoint checkPointCopy (오프셋 40)
    ("ckpt_redo", 40, "Q"),
    ("ckpt_timeline", 48, "I"),
    ("ckpt_prev_timeline", 52, "I"),
    ("ckpt_full_page_writes", 56, "?"),
    ("ckpt_wal_level", 60, "i"),
    ("ckpt_next_xid", 64, "Q"),  # FullTransactionId (epoch << 32 | xid)
    ("ckpt_next_oid", 72, "I"),
    ("ckpt_next_multi", 76, "I"),
    ("ckpt_next_multi_offset", 80, "I"),
    ("ckpt_oldest_xid", 84, "I"),
    ("ckpt_oldest_xid_db", 88, "I"),
    ("ckpt_oldest_multi", 92, "I"),
    ("ckpt_oldest_multi_db", 96, "I"),
    ("ckpt_time", 104, "q"),
    ("ckpt_oldest_commit_ts_xid", 112, "I"),
    ("ckpt_newest_commit_ts_xid", 116, "I"),
    ("ckpt_oldest_active_xid", 120, "I"),
    ("unlogged_lsn", 128, "Q"),
    ("min_recovery_point", 136, "Q"),
    ("min_recovery_point_tli", 144, "I"),
    ("backup_start_point", 152, "Q"),
    ("backup_end_point", 160, "Q"),
    ("backup_end_required", 168, "?"),
    ("wal_level", 172, "i"),
    ("wal_log_hints", 176, "?"),
    ("max_connections", 180, "i"),
    ("max_worker_processes", 184, "i"),
    ("max_wal_senders", 188, "i"),
    ("max_prepared_xacts", 192, "i"),
    ("max_locks_per_xact", 196, "i"),
    ("track_commit_timestamp", 200, "?"),
]

# floatFormat 기준점 뒤쪽 (모든 지원 버전 공통): (이름, floatFormat 기준 상대 오프셋, 형식)
_TAIL = [
    ("max_align", -4, "I"),
    ("blcksz", 8, "I"),
    ("relseg_size", 12, "I"),
    ("xlog_blcksz", 16, "I"),
    ("xlog_seg_size", 20, "I"),
    ("name_data_len", 24, "I"),
    ("index_max_keys", 28, "I"),
    ("toast_max_chunk_size", 32, "I"),
    ("loblksize", 36, "I"),
    ("float8_by_val", 40, "?"),
    ("data_checksum_version", 44, "I"),
]
_NONCE_REL = 48  # mock_authentication_nonce[32]
_CRC_REL = 80


def lsn_str(v: int) -> str:
    return f"{v >> 32:X}/{v & 0xFFFFFFFF:X}"


def _time_str(t: int) -> str | None:
    try:
        return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def parse_control(data: bytes) -> dict:
    out: dict = {"warnings": []}
    version = struct.unpack_from("<I", data, 8)[0]
    out["system_identifier"] = struct.unpack_from("<Q", data, 0)[0]
    out["pg_control_version"] = version
    out["catalog_version_no"] = struct.unpack_from("<I", data, 12)[0]
    out["state"] = struct.unpack_from("<i", data, 16)[0]

    if version == 1700:
        for name, off, fmt in _FIELDS_1700:
            out[name] = struct.unpack_from("<" + fmt, data, off)[0]
        out["layout"] = "pg_control 1700 (PostgreSQL 17)"
    else:
        out["warnings"].append(
            f"pg_control_version {version} 은 앞부분·뒷부분만 해석합니다 (CheckPoint 상세는 PG 17 만)"
        )
        out["layout"] = f"pg_control {version} (부분 해석)"

    anchor = data.find(FLOAT_FORMAT)
    if anchor < 0:
        out["warnings"].append("floatFormat 기준점(1234567.0)을 찾지 못했습니다 — blcksz 8192 로 가정")
    else:
        out["float_format_offset"] = anchor
        for name, rel, fmt in _TAIL:
            out[name] = struct.unpack_from("<" + fmt, data, anchor + rel)[0]
        crc_off = anchor + _CRC_REL
        stored = struct.unpack_from("<I", data, crc_off)[0]
        calc = crc32c(data[:crc_off])
        out["crc_offset"] = crc_off
        out["crc_stored"] = f"{stored:08x}"
        out["crc_calculated"] = f"{calc:08x}"
        out["crc_ok"] = stored == calc
        if stored != calc:
            out["warnings"].append("pg_control CRC 불일치 — 손상되었거나 구조 해석이 맞지 않습니다")

    # 표시용 파생 값
    out["state_name"] = DB_STATES.get(out["state"], f"unknown({out['state']})")
    for k in ("time", "ckpt_time"):
        if k in out:
            out[k + "_utc"] = _time_str(out[k])
    for k in ("checkpoint", "ckpt_redo", "unlogged_lsn", "min_recovery_point",
              "backup_start_point", "backup_end_point"):
        if k in out:
            out[k + "_lsn"] = lsn_str(out[k])
    if "ckpt_next_xid" in out:
        nx = out["ckpt_next_xid"]
        out["ckpt_next_xid_epoch"] = nx >> 32
        out["ckpt_next_xid_xid"] = nx & 0xFFFFFFFF
    for k in ("wal_level", "ckpt_wal_level"):
        if k in out:
            out[k + "_name"] = WAL_LEVELS.get(out[k], str(out[k]))
    out["system_identifier"] = str(out["system_identifier"])  # JS 정밀도 보호
    return out
