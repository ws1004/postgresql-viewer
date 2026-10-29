"""타입별 값 디코딩 (디스크 바이트 → JSON 으로 내보낼 수 있는 Python 값).

출처 (PostgreSQL 17):
  src/include/catalog/pg_type.dat     내장 타입 OID
  src/backend/utils/adt/numeric.c     NumericShort / NumericLong
  src/include/utils/jsonb.h           JsonbContainer · JEntry
  src/include/utils/array.h           ArrayType
  src/include/utils/inet.h            inet_struct
  src/include/datatype/timestamp.h    2000-01-01 기준 마이크로초
"""

from __future__ import annotations

import ipaddress
import math
import struct
import uuid as _uuid
from datetime import date, datetime, timedelta, timezone

from .varlena import varlena_size

PG_EPOCH = datetime(2000, 1, 1)
PG_EPOCH_DATE = date(2000, 1, 1)
INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)

ALIGN = {"c": 1, "s": 2, "i": 4, "d": 8}


def align(off: int, a: str) -> int:
    n = ALIGN[a]
    return (off + n - 1) & ~(n - 1)


class DecodeError(Exception):
    pass


# ------------------------------------------------------------ 고정 길이

def _ts(us: int, tz: bool):
    if us == INT64_MAX:
        return "infinity"
    if us == INT64_MIN:
        return "-infinity"
    try:
        dt = PG_EPOCH + timedelta(microseconds=us)
    except OverflowError:
        return f"<timestamp {us}>"
    return dt.replace(tzinfo=timezone.utc).isoformat() if tz else dt.isoformat()


def _date(days: int):
    if days == INT32_MAX:
        return "infinity"
    if days == INT32_MIN:
        return "-infinity"
    try:
        return (PG_EPOCH_DATE + timedelta(days=days)).isoformat()
    except OverflowError:
        return f"<date {days}>"


def _time(us: int) -> str:
    s, us = divmod(us, 1_000_000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" + (f".{us:06d}" if us else "")


def _interval(b: bytes) -> str:
    us, days, months = struct.unpack("<qii", b)
    parts = []
    if months:
        y, m = int(months / 12), months - int(months / 12) * 12  # 0 쪽으로 자른다 (-13 → -1 years -1 mons)
        parts.append(" ".join(x for x in (f"{y} years" if y else "", f"{m} mons" if m else "") if x))
    if days:
        parts.append(f"{days} days")
    if us or not parts:
        sign = "-" if us < 0 else ""
        parts.append(sign + _time(abs(us)))
    return " ".join(parts)


def _char(b: bytes) -> str:
    c = b[0]
    if c == 0:
        return ""
    return chr(c) if 32 <= c < 127 else f"\\{c:03o}"


def _name(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("utf-8", "backslashreplace")


def _tid(b: bytes) -> str:
    hi, lo, pos = struct.unpack("<HHH", b)
    return f"({(hi << 16) | lo},{pos})"


def _money(v: int) -> str:
    q, r = divmod(abs(v), 100)
    return f"{'-' if v < 0 else ''}{q}.{r:02d}"


def _lsn(b: bytes) -> str:
    v = struct.unpack("<Q", b)[0]
    return f"{v >> 32:X}/{v & 0xFFFFFFFF:X}"


FIXED = {
    16: lambda b: b[0] != 0,                              # bool
    18: _char,                                            # "char"
    19: _name,                                            # name
    20: lambda b: struct.unpack("<q", b)[0],              # int8
    21: lambda b: struct.unpack("<h", b)[0],              # int2
    23: lambda b: struct.unpack("<i", b)[0],              # int4
    24: lambda b: struct.unpack("<I", b)[0],              # regproc
    26: lambda b: struct.unpack("<I", b)[0],              # oid
    27: _tid,                                             # tid
    28: lambda b: struct.unpack("<I", b)[0],              # xid
    29: lambda b: struct.unpack("<I", b)[0],              # cid
    700: lambda b: struct.unpack("<f", b)[0],             # float4
    701: lambda b: struct.unpack("<d", b)[0],             # float8
    790: lambda b: _money(struct.unpack("<q", b)[0]),     # money (lc_monetary 소수 2자리 가정 — 문자열)
    1082: lambda b: _date(struct.unpack("<i", b)[0]),     # date
    1083: lambda b: _time(struct.unpack("<q", b)[0]),     # time
    1114: lambda b: _ts(struct.unpack("<q", b)[0], False),  # timestamp
    1184: lambda b: _ts(struct.unpack("<q", b)[0], True),   # timestamptz (UTC 로 표시)
    1186: _interval,                                      # interval
    2950: lambda b: str(_uuid.UUID(bytes=bytes(b))),      # uuid
    3220: _lsn,                                           # pg_lsn
    5069: lambda b: struct.unpack("<Q", b)[0],            # xid8
}
# reg* 계열 (regclass, regtype, regnamespace, regrole, regprocedure, regoper, regoperator, regconfig, regdictionary, regcollation)
for _oid in (2205, 2206, 4089, 4096, 2202, 2203, 2204, 3734, 3769, 4191):
    FIXED[_oid] = FIXED[26]


# ------------------------------------------------------------ 가변 길이

TEXT_TYPES = {25, 1043, 1042, 114, 142, 194}  # text varchar bpchar json xml pg_node_tree

# pg_database.encoding → Python 코덱 (src/include/mb/pg_wchar.h 의 서버 인코딩 번호 중 확인한 것만)
ENCODINGS = {0: "utf-8", 1: "euc_jp", 2: "euc_cn", 3: "euc_kr", 6: "utf-8", 8: "latin-1"}


def decode_text(p: bytes, codec: str = "utf-8") -> str:
    """텍스트 디코딩. 깨진 바이트는 \\xNN 으로 남겨 원래 바이트를 잃지 않는다."""
    return p.decode(codec, "backslashreplace")


def numeric_from_bytes(p: bytes) -> str:
    if len(p) < 2:
        raise DecodeError("numeric 이 너무 짧습니다")
    h = struct.unpack_from("<H", p, 0)[0]
    if h & 0xC000 == 0xC000:  # NUMERIC_SPECIAL
        return {0xC000: "NaN", 0xD000: "Infinity", 0xF000: "-Infinity"}.get(h & 0xF000, "NaN")
    if h & 0x8000:  # NUMERIC_SHORT
        neg = bool(h & 0x2000)
        dscale = (h & 0x1F80) >> 7
        weight = h & 0x003F
        if h & 0x0040:
            weight -= 64
        dpos = 2
    else:
        neg = (h & 0xC000) == 0x4000
        dscale = h & 0x3FFF
        weight = struct.unpack_from("<h", p, 2)[0]
        dpos = 4
    nd = (len(p) - dpos) // 2
    digits = struct.unpack_from(f"<{nd}h", p, dpos) if nd else ()
    if any(not 0 <= d < 10000 for d in digits):
        raise DecodeError("numeric 자릿값이 0..9999 범위 밖입니다")
    # 정수 연산만 쓴다 (Decimal 컨텍스트 정밀도로 반올림되지 않게). 값 = M × 10^e
    m = 0
    for d in digits:
        m = m * 10000 + d
    e = 4 * (weight - nd + 1)
    s = str(m) if nd else "0"
    if e >= 0:
        ip, fp = s + "0" * e, ""
    else:
        s = s.rjust(-e + 1, "0")
        ip, fp = s[:e], s[e:]
    ip = ip.lstrip("0") or "0"
    fp = fp[:dscale].ljust(dscale, "0")  # dscale 밖 자리는 저장 규칙상 0 이다
    out = ip + ("." + fp if dscale else "")
    return ("-" + out) if neg and out.strip("0.") else out


def _num_value(s: str):
    """jsonb 숫자를 JSON 값으로. 정수는 int, 유한한 소수는 float, 그 밖(NaN·범위 초과)은 문자열."""
    try:
        if s.lstrip("-").isdigit():
            return int(s)
        f = float(s)
        return f if math.isfinite(f) else s
    except ValueError:
        return s


JB_CMASK = 0x0FFFFFFF
JB_FSCALAR = 0x10000000
JB_FOBJECT = 0x20000000
JB_FARRAY = 0x40000000
JENTRY_OFFLENMASK = 0x0FFFFFFF
JENTRY_TYPEMASK = 0x70000000
JENTRY_HAS_OFF = 0x80000000
JENTRY_ISSTRING, JENTRY_ISNUMERIC, JENTRY_ISBOOL_FALSE, JENTRY_ISBOOL_TRUE, JENTRY_ISNULL, JENTRY_ISCONTAINER = (
    0x00000000, 0x10000000, 0x20000000, 0x30000000, 0x40000000, 0x50000000)


def _intalign(x: int) -> int:
    return (x + 3) & ~3


def _jsonb_container(p: bytes, base: int, depth: int = 0, budget: list | None = None):
    if depth > 200:
        raise DecodeError("jsonb 중첩이 너무 깊습니다")
    if budget is None:
        budget = [len(p)]  # 노드 하나당 최소 4바이트(JEntry)이므로 노드 수는 바이트 수를 넘을 수 없다
    header = struct.unpack_from("<I", p, base)[0]
    count = header & JB_CMASK
    is_obj = bool(header & JB_FOBJECT)
    n = count * 2 if is_obj else count
    if base + 4 + 4 * n > len(p):
        raise DecodeError(f"jsonb: JEntry {n}개가 데이터를 넘습니다")
    budget[0] -= n
    if budget[0] < 0:
        raise DecodeError("jsonb: 노드 수가 데이터 크기보다 많습니다 (손상 또는 자식 공유)")
    entries = struct.unpack_from(f"<{n}I", p, base + 4) if n else ()
    data_base = base + 4 + 4 * n
    values = []
    end = 0
    for je in entries:
        start = end
        end = (je & JENTRY_OFFLENMASK) if je & JENTRY_HAS_OFF else start + (je & JENTRY_OFFLENMASK)
        if end < start or data_base + end > len(p):
            raise DecodeError("jsonb: JEntry 오프셋이 거꾸로이거나 데이터를 넘습니다")
        t = je & JENTRY_TYPEMASK
        if t == JENTRY_ISSTRING:
            values.append(p[data_base + start:data_base + end].decode("utf-8", "backslashreplace"))
        elif t == JENTRY_ISNUMERIC:
            pos = data_base + _intalign(start)
            size = varlena_size(p, pos)
            hdr = 1 if p[pos] & 0x01 else 4
            values.append(_num_value(numeric_from_bytes(p[pos + hdr:pos + size])))
        elif t == JENTRY_ISBOOL_FALSE:
            values.append(False)
        elif t == JENTRY_ISBOOL_TRUE:
            values.append(True)
        elif t == JENTRY_ISNULL:
            values.append(None)
        elif t == JENTRY_ISCONTAINER:
            if end - start < 4:
                raise DecodeError("jsonb: 하위 컨테이너가 너무 작습니다")
            values.append(_jsonb_container(p, data_base + _intalign(start), depth + 1, budget))
        else:
            raise DecodeError(f"jsonb: 알 수 없는 JEntry 타입 {t:#x}")
    if is_obj:
        return dict(zip(values[:count], values[count:]))
    if header & JB_FSCALAR:
        return values[0] if values else None
    return values


def jsonb_from_bytes(p: bytes):
    return _jsonb_container(p, 0)


def _inet(p: bytes) -> str:
    # inet_struct: family(1) · bits(1) · ipaddr[주소 길이] (src/include/utils/inet.h). inet/cidr 구분은 타입 OID 로 한다
    family, bits, addr = p[0], p[1], p[2:]
    if family == 2 and len(addr) == 4:  # PGSQL_AF_INET
        s, full = ".".join(str(x) for x in addr), 32
    elif family == 3 and len(addr) == 16:  # PGSQL_AF_INET6
        s, full = str(ipaddress.IPv6Address(bytes(addr))), 128
    else:
        return "\\x" + p.hex()
    return s if bits == full else f"{s}/{bits}"


def bytes_display(p: bytes) -> str:
    return "\\x" + p.hex()


VARLENA = {
    17: bytes_display,          # bytea
    1700: numeric_from_bytes,   # numeric (정밀도 보존을 위해 문자열)
    3802: jsonb_from_bytes,     # jsonb
    869: _inet,                 # inet
    650: _inet,                 # cidr
}
