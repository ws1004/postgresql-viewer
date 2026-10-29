"""클러스터 · 데이터베이스 · 릴레이션 — 엔진의 공개 API.

    cluster = open_cluster("PGDATA 폴더 또는 zip")
    for db in cluster.databases(): ...
    d = cluster.database(16390)
    rel = d.relation_by_name("public", "posts")
    for row in d.rows(rel.oid): ...

카탈로그 부트스트랩 (서버 없이 스키마를 알아내는 방법):
  1. pg_attribute 의 앞 5개 컬럼(attrelid · attname · atttypid · attlen · attnum)만 고정 정의로 읽는다.
     이 앞부분은 PG 12 이후 바뀌지 않았다 (src/include/catalog/pg_attribute.h).
  2. attrelid = 1249 (pg_attribute 자신) 행에서 pg_attribute 의 전체 고정 길이 컬럼 정의를 얻는다.
  3. 그 정의로 pg_attribute 를 다시 읽고, attalign 값이 2단계의 가정과 같은지 검증한다.
  4. 이후 모든 릴레이션(카탈로그 포함)은 pg_attribute 에 적힌 attlen · attalign 으로 디코딩한다.
"""

from __future__ import annotations

import re
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from ...source import Source, open_source
from .clog import Clog
from .control import parse_control
from .heap import (HEAP_HASNULL, HEAP_UPDATED, LP_NORMAL, LP_FLAG_NAMES, TUPLE_HEADER_SIZE,
                   classify, line_pointers, page_problem, parse_page_header, parse_tuple_header,
                   TupleHeader)
from .relmap import parse_relmap
from .types import (ALIGN, ENCODINGS, FIXED, TEXT_TYPES, VARLENA, DecodeError, align, bytes_display,
                    decode_text)
from .varlena import (VarlenaError, detoast_external_data, detoast_inline, parse_external,
                      varlena_kind, varlena_size)

# 카탈로그 OID (src/include/catalog/*.h 의 CATALOG(..., OID, ...))
PG_TYPE, PG_ATTRIBUTE, PG_CLASS, PG_DATABASE, PG_NAMESPACE, PG_ENUM = 1247, 1249, 1259, 1262, 2615, 3501
PG_INDEX = 2610
BTREE_AM = 403
DEFAULTTABLESPACE_OID, GLOBALTABLESPACE_OID = 1663, 1664
ANYARRAY = 2277

# 부트스트랩용: 타입 OID → attalign (pg_type.dat 의 typalign)
BOOT_TYPE_ALIGN = {16: "c", 18: "c", 19: "c", 20: "d", 21: "s", 23: "i", 24: "i", 26: "i",
                   28: "i", 29: "i", 700: "i", 701: "d"}

RELKIND_NAMES = {"r": "table", "i": "index", "S": "sequence", "t": "toast", "v": "view",
                 "m": "matview", "c": "composite", "f": "foreign", "p": "partitioned",
                 "I": "partitioned index"}
HEAP_KINDS = {"r", "t", "m", "S"}


class Missing:
    """튜플의 natts 보다 뒤에 추가된 컬럼 (ALTER TABLE ADD COLUMN 이후 다시 쓰이지 않은 행)."""
    __slots__ = ()


MISSING = Missing()


@dataclass(slots=True)
class Varlena:
    raw: bytes


@dataclass(slots=True)
class Cstring:
    raw: bytes


@dataclass(slots=True)
class Attr:
    num: int
    name: str
    typid: int
    len: int
    align: str
    byval: bool = False
    dropped: bool = False
    typmod: int = -1
    notnull: bool = False
    ndims: int = 0

    def to_dict(self, types=None) -> dict:
        t = types.get(self.typid) if types else None
        return {"num": self.num, "name": self.name, "typid": self.typid,
                "type": t.display_name if t else (None if self.dropped else f"oid:{self.typid}"),
                "len": self.len, "align": self.align, "notnull": self.notnull,
                "dropped": self.dropped, "typmod": self.typmod}


@dataclass(slots=True)
class TypeInfo:
    oid: int
    name: str
    nsp: int
    len: int
    byval: bool
    typtype: str
    category: str
    elem: int
    align: str
    basetype: int
    relid: int
    display_name: str = ""


@dataclass(slots=True)
class RawTuple:
    block: int
    lp: int
    lp_off: int
    lp_len: int
    file_offset: int
    header: TupleHeader
    data: bytes
    file: str = ""


@dataclass
class Relation:
    oid: int
    name: str
    nsp_oid: int
    nsp: str
    kind: str
    filenode: int
    tablespace: int
    toastrelid: int
    persistence: str
    natts: int
    relpages: int
    reltuples: float
    shared: bool
    path: str | None = None
    attrs: list[Attr] = field(default_factory=list)
    am: int = 0             # pg_class.relam (403 = btree)
    filenumber: int = 0     # 실제 파일 번호 (relfilenode, 매핑 카탈로그는 pg_filenode.map)

    @property
    def has_heap(self) -> bool:
        return self.kind in HEAP_KINDS

    def summary(self, source=None) -> dict:
        return {"oid": self.oid, "schema": self.nsp, "name": self.name, "kind": self.kind,
                "kind_name": RELKIND_NAMES.get(self.kind, self.kind), "filenode": self.filenode,
                "path": self.path, "toastrelid": self.toastrelid, "persistence": self.persistence,
                "natts": self.natts, "relpages": self.relpages, "reltuples": self.reltuples,
                "shared": self.shared}


# ------------------------------------------------------------ 저수준: 파일 → 튜플

def iter_pages(cluster: "Cluster", path: str):
    """(블록 번호, 페이지 바이트, 세그먼트 파일, 파일 내 오프셋) — 정상·비정상 페이지 모두."""
    bs = cluster.blcksz
    seg = 0
    while True:
        seg_path = path if seg == 0 else f"{path}.{seg}"
        if not cluster.source.exists(seg_path):
            return
        data = cluster.source.read(seg_path)
        for bi in range(len(data) // bs):
            yield seg * cluster.relseg_size + bi, data[bi * bs:(bi + 1) * bs], seg_path, bi * bs
        seg += 1


def scan_relation_file(cluster: "Cluster", path: str, problems: list | None = None):
    """릴레이션 파일(세그먼트 포함)의 모든 정상 라인 포인터 튜플."""
    bs = cluster.blcksz
    blocks_per_seg = cluster.relseg_size
    seg = 0
    while True:
        seg_path = path if seg == 0 else f"{path}.{seg}"
        if not cluster.source.exists(seg_path):
            break
        data = cluster.source.read(seg_path)
        if len(data) % bs and problems is not None:
            problems.append({"block": seg * blocks_per_seg + len(data) // bs,
                             "problem": f"잘린 마지막 블록 ({len(data) % bs}바이트) — 읽지 않음"})
        for bi in range(len(data) // bs):
            block = seg * blocks_per_seg + bi
            page = data[bi * bs:(bi + 1) * bs]
            h = parse_page_header(page)
            prob = page_problem(h, bs)
            if prob:
                if prob != "new" and problems is not None:
                    problems.append({"block": block, "problem": prob})
                continue
            for lp, off, flags, length in line_pointers(page, h):
                if flags != LP_NORMAL or length < TUPLE_HEADER_SIZE or off + length > bs:
                    continue
                tup = page[off:off + length]
                # file_offset 은 세그먼트 파일(seg_path) 안의 오프셋 — 헥스 에디터로 바로 찾아갈 수 있는 값
                yield RawTuple(block, lp, off, length, bi * bs + off, parse_tuple_header(tup), tup, seg_path)
        seg += 1


def deform(tup: bytes, h: TupleHeader, attrs: list[Attr], spans: list | None = None,
           out: list | None = None) -> list:
    """튜플 데이터 영역을 컬럼별 Datum(bytes / Varlena / Cstring / None / MISSING)으로 나눈다.

    heap_deform_tuple() 과 같은 규칙: 고정 길이는 attalign 으로 정렬, varlena 는 현재 바이트가 0(패딩)일
    때만 정렬한다 (att_align_pointer). 정렬은 튜플 시작 기준이다 (튜플은 MAXALIGN 위치에 있다).
    spans 를 주면 컬럼마다 (시작 오프셋, 길이) 를 채운다 (NULL·MISSING 은 (None, 0)).
    out 을 주면 거기에 채운다 — 중간에 실패해도 앞쪽 컬럼은 out 에 남는다.
    """
    natts = h.natts
    n = len(tup)
    nbitmap = (natts + 7) // 8 if h.infomask & HEAP_HASNULL else 0
    if not TUPLE_HEADER_SIZE + nbitmap <= h.hoff <= n:
        raise DecodeError(f"t_hoff {h.hoff} 가 범위 밖 (헤더+비트맵 {TUPLE_HEADER_SIZE + nbitmap}, 튜플 {n})")
    bitmap = tup[TUPLE_HEADER_SIZE:TUPLE_HEADER_SIZE + nbitmap] if nbitmap else None
    return deform_core(tup, h.hoff, bitmap, natts, attrs, spans, out)


def deform_core(tup: bytes, off: int, bitmap: bytes | None, natts: int, attrs: list[Attr],
                spans: list | None = None, out: list | None = None) -> list:
    """힙 튜플과 인덱스 튜플이 같이 쓰는 분해 루프. off = 데이터 시작, bitmap = NULL 비트맵(비트 1 = 값 있음)."""
    n = len(tup)
    if out is None:
        out = []
    for i, a in enumerate(attrs):
        if i >= natts:
            out.append(MISSING)
            if spans is not None:
                spans.append((None, 0))
            continue
        if bitmap is not None and not (bitmap[i >> 3] >> (i & 7)) & 1:
            out.append(None)
            if spans is not None:
                spans.append((None, 0))
            continue
        if a.len == -1:
            if off < n and tup[off] == 0:
                off = align(off, a.align)
            if off >= n:
                raise DecodeError(f"{a.name}: 튜플 끝을 넘었습니다")
            size = varlena_size(tup, off)
            if size == 0 or off + size > n:
                raise DecodeError(f"{a.name}: varlena 길이 {size} 가 튜플을 넘습니다")
            out.append(Varlena(tup[off:off + size]))
            off += size
        elif a.len == -2:
            off = align(off, a.align)
            end = tup.find(b"\0", off)
            if end < 0:
                raise DecodeError(f"{a.name}: cstring 끝이 없습니다")
            out.append(Cstring(tup[off:end]))
            off = end + 1
        else:
            off = align(off, a.align)
            if off + a.len > n:
                raise DecodeError(f"{a.name}: 고정 길이 {a.len} 가 튜플을 넘습니다")
            out.append(tup[off:off + a.len])
            off += a.len
        if spans is not None:
            raw = out[-1].raw if isinstance(out[-1], (Varlena, Cstring)) else out[-1]
            size = len(raw) + (1 if isinstance(out[-1], Cstring) else 0)
            spans.append((off - size, size))
    return out


# ------------------------------------------------------------ Cluster

class Cluster:
    def __init__(self, source: Source):
        self.source = source
        self.pg_version = source.read("PG_VERSION").decode("ascii", "replace").strip()
        if not re.fullmatch(r"\d+(\.\d+)?", self.pg_version):
            raise ValueError(f"PG_VERSION 내용이 이상합니다: {self.pg_version[:20]!r}")
        self.control = parse_control(source.read("global/pg_control"))
        self.blcksz = self.control.get("blcksz", 8192)
        self.relseg_size = self.control.get("relseg_size", 131072)
        if self.blcksz not in (1024, 2048, 4096, 8192, 16384, 32768) or self.relseg_size <= 0:
            raise ValueError(f"pg_control 의 blcksz {self.blcksz} / relseg_size {self.relseg_size} 가 비정상입니다")
        # 정상 종료가 아니어도, 체크포인트의 nextXid 보다 작은데 커밋 기록이 없는 xid 는 끝난(중단된) 트랜잭션이다
        self.clog = Clog(source, self.blcksz, self.control.get("ckpt_next_xid_xid"),
                         clean_shutdown=self.control.get("state") in (1, 2))
        gm = parse_relmap(source.read("global/pg_filenode.map"))
        self.global_map: dict[int, int] = gm["mappings"]
        self.global_map_crc_ok = gm["crc_ok"]
        self.db_dirs = sorted(int(x) for x in source.listdir("base") if x.isdigit())
        self._dbs: dict[int, Database] = {}
        self._db_rows: list[dict] | None = None
        self._lock = threading.RLock()
        self._wal = None
        self._wal_error: str | None = None
        self._wal_lock = threading.Lock()  # WAL 읽기 전용 — 카탈로그 락(_lock)을 오래 막지 않는다

    def wal(self):
        """WAL 색인 (처음 부를 때 pg_wal 전체를 읽는다). 읽을 수 없으면 None.
        읽은 뒤에는 pg_xact 에 결과가 없는 트랜잭션의 커밋/롤백을 WAL 기록으로 보충한다.
        락 순서: _wal_lock 은 다른 락을 잡은 채 기다리지 않는다 (WalIndex 는 Source 만 읽는다)."""
        from .wal import WalIndex
        with self._wal_lock:
            if self._wal is None and self._wal_error is None:
                try:
                    self._wal = WalIndex(self.source, self.blcksz)
                    self.clog.wal = self._wal
                except Exception as e:
                    self._wal_error = f"{type(e).__name__}: {e}"
            return self._wal

    def wal_summary(self) -> dict:
        w = self.wal()
        if w is None:
            return {"available": False, "error": self._wal_error}
        return {"available": True, **w.stats}

    # 표시용
    def info(self) -> dict:
        return {"pg_version": self.pg_version, "pgdata": self.source.pgdata,
                "input": str(self.source.input_path), "input_kind": self.source.kind,
                "control": self.control, "global_map_crc_ok": self.global_map_crc_ok,
                "database_dirs": self.db_dirs}

    def _bootstrap_db(self) -> "Database":
        # 공유 카탈로그(pg_database)의 컬럼 정의는 어느 데이터베이스의 pg_attribute 에도 있다
        for oid in [1] + self.db_dirs:
            if oid in self.db_dirs:
                return self.database(oid)
        raise RuntimeError("base/ 아래 데이터베이스 폴더가 없습니다")

    def databases(self) -> list[dict]:
        with self._lock:
            if self._db_rows is None:
                boot = self._bootstrap_db()
                attrs = boot.attrs_for(PG_DATABASE)
                path = f"global/{self.global_map[PG_DATABASE]}"
                rows = []
                for t in scan_relation_file(self, path):
                    state, _, _ = classify(t.header, self.clog, t.block, t.lp)
                    if state != "live":
                        continue
                    try:
                        r = boot.decode_dict(t, attrs)
                    except Exception as e:
                        boot.diagnostics.append({"check": "pg_database 행 해석", "ok": False,
                                                 "detail": f"({t.block},{t.lp}) {type(e).__name__}: {e}"})
                        continue
                    rows.append({"oid": r["oid"], "name": r["datname"], "encoding": r.get("encoding"),
                                 "tablespace": r.get("dattablespace"),
                                 "is_template": r.get("datistemplate"),
                                 "allow_conn": r.get("datallowconn"),
                                 "collate": r.get("datcollate"), "ctype": r.get("datctype"),
                                 "has_dir": r["oid"] in self.db_dirs})
                rows.sort(key=lambda r: r["oid"])
                # base/ 에 폴더는 있는데 pg_database 에 없는 DB (DROP 됐지만 폴더가 남은 경우 등)
                known = {r["oid"] for r in rows}
                for oid in self.db_dirs:
                    if oid not in known:
                        rows.append({"oid": oid, "name": f"(pg_database 에 없음: {oid})", "encoding": None,
                                     "tablespace": None, "is_template": False, "allow_conn": None,
                                     "collate": None, "ctype": None, "has_dir": True, "orphan": True})
                self._db_rows = rows
                for db in self._dbs.values():  # 부트스트랩 때 meta 없이 만든 DB 에 이름·인코딩을 채운다
                    db.apply_meta(next((r for r in rows if r["oid"] == db.oid), None))
            return self._db_rows

    def database(self, oid: int) -> "Database":
        with self._lock:
            if oid not in self._dbs:
                if oid not in self.db_dirs:
                    raise KeyError(f"base/{oid} 폴더가 없습니다")
                meta = None
                if self._db_rows is not None:
                    meta = next((r for r in self._db_rows if r["oid"] == oid), None)
                self._dbs[oid] = Database(self, oid, meta)
            return self._dbs[oid]

    def close(self):
        self.source.close()


def open_cluster(path) -> Cluster:
    source = open_source(path)
    try:
        return Cluster(source)
    except Exception:
        source.close()  # zip 핸들이 남으면 Windows 에서 파일이 잠긴다
        raise


# ------------------------------------------------------------ Database

class Database:
    ROW_CACHE = 32

    def __init__(self, cluster: Cluster, oid: int, meta: dict | None):
        self.cluster = cluster
        self.oid = oid
        self.name = None
        self.codec = "utf-8"
        self.default_tablespace = DEFAULTTABLESPACE_OID
        lm = parse_relmap(cluster.source.read(f"base/{oid}/pg_filenode.map"))
        self.local_map: dict[int, int] = lm["mappings"]
        self.diagnostics: list[dict] = [{"check": "pg_filenode.map CRC", "ok": lm["crc_ok"]}]
        self._lock = threading.RLock()
        self._rows_cache: OrderedDict[int, list[dict]] = OrderedDict()
        self._toast: dict[int, dict] = {}
        self._toast_errors: dict[int, int] = {}
        self._rel_locks: dict[int, threading.Lock] = {}
        self._key_index = None
        self._key_lock = threading.Lock()
        self.apply_meta(meta)
        self._bootstrap()

    def apply_meta(self, meta: dict | None):
        if not meta:
            return
        self.name = meta["name"]
        self.default_tablespace = meta.get("tablespace") or DEFAULTTABLESPACE_OID
        enc = meta.get("encoding")
        if enc is not None:
            self.codec = ENCODINGS.get(enc, "utf-8")
            if enc not in ENCODINGS:
                self.diagnostics.append({"check": "텍스트 인코딩", "ok": False,
                                         "detail": f"pg_database.encoding {enc} 미지원 — UTF-8 로 읽고 깨진 바이트는 \\xNN 으로 표시"})

    # ---------------- 부트스트랩
    def _bootstrap(self):
        att_path = f"base/{self.oid}/{self.local_map[PG_ATTRIBUTE]}"
        boot = [Attr(1, "attrelid", 26, 4, "i"), Attr(2, "attname", 19, 64, "c"),
                Attr(3, "atttypid", 26, 4, "i"), Attr(4, "attlen", 21, 2, "s"),
                Attr(5, "attnum", 21, 2, "s")]
        live = []
        for t in scan_relation_file(self.cluster, att_path):
            state, _, _ = classify(t.header, self.cluster.clog, t.block, t.lp)
            if state == "live":
                live.append(t)
        self_rows = []
        skipped = 0
        for t in live:
            try:
                d = deform(t.data, t.header, boot)
            except Exception:
                skipped += 1
                continue
            relid = struct.unpack("<I", d[0])[0]
            if relid == PG_ATTRIBUTE:
                self_rows.append((struct.unpack("<h", d[4])[0], FIXED[19](d[1]),
                                  struct.unpack("<I", d[2])[0], struct.unpack("<h", d[3])[0]))
        # attnum < 0 은 시스템 컬럼(ctid, xmin …) 행이다
        self_rows = sorted(r for r in self_rows if r[0] > 0)
        if not self_rows or self_rows[0][0] != 1:
            raise RuntimeError("pg_attribute 자기 기술 행을 찾지 못했습니다")
        prefix: list[Attr] = []
        for num, name, typid, length in self_rows:
            if num <= 0:
                continue
            if length <= 0 or typid not in BOOT_TYPE_ALIGN:
                break  # 첫 가변 길이 컬럼부터는 3단계 이후의 정의로 읽는다
            prefix.append(Attr(num, name, typid, length, BOOT_TYPE_ALIGN[typid]))
        need = {"attrelid", "attname", "atttypid", "attlen", "attnum", "attalign", "attbyval",
                "attisdropped", "atttypmod", "attnotnull", "attndims"}
        missing = need - {a.name for a in prefix}
        if missing:
            raise RuntimeError(f"pg_attribute 고정부에 필요한 컬럼이 없습니다: {sorted(missing)}")
        self.diagnostics.append({"check": "pg_attribute 고정부 컬럼", "ok": True,
                                 "detail": ", ".join(f"{a.name}({a.len})" for a in prefix)})

        attrs_by_rel: dict[int, list[Attr]] = {}
        for t in live:
            try:
                d = deform(t.data, t.header, prefix)
                r = {a.name: self._fixed(v, a.typid) for a, v in zip(prefix, d)}
                if r["attalign"] not in ALIGN:
                    raise DecodeError(f"attalign {r['attalign']!r}")
            except Exception:
                skipped += 1
                continue
            if r["attnum"] <= 0:
                continue
            attrs_by_rel.setdefault(r["attrelid"], []).append(Attr(
                r["attnum"], r["attname"], r["atttypid"], r["attlen"], r["attalign"],
                r["attbyval"], r["attisdropped"], r["atttypmod"], r["attnotnull"], r["attndims"]))
        for lst in attrs_by_rel.values():
            lst.sort(key=lambda a: a.num)
        self._attrs = attrs_by_rel

        # 3단계: 자기 기술 행의 attalign 이 가정과 같은지
        mism = [a.name for a, b in zip(prefix, attrs_by_rel[PG_ATTRIBUTE]) if a.align != b.align]
        self.diagnostics.append({"check": "pg_attribute attalign 자기 검증", "ok": not mism,
                                 "detail": f"{len(prefix)}개 컬럼 비교" + (f", 불일치 {mism}" if mism else "")})
        if skipped:
            self.diagnostics.append({"check": "pg_attribute 행 해석", "ok": False,
                                     "detail": f"손상으로 건너뛴 live 행 {skipped}개 — 일부 테이블의 컬럼 정의가 빠졌을 수 있음"})
        self._catalog_errors: dict[str, int] = {}

        self.types: dict[int, TypeInfo] = {}
        for r in self._catalog_rows(PG_TYPE, mapped=True):
            self.types[r["oid"]] = TypeInfo(r["oid"], r["typname"], r["typnamespace"], r["typlen"],
                                            r["typbyval"], r["typtype"], r["typcategory"], r["typelem"],
                                            r["typalign"], r["typbasetype"], r["typrelid"])
        class_rows = list(self._catalog_rows(PG_CLASS, mapped=True))
        class_by_oid = {r["oid"]: r for r in class_rows}
        self.namespaces = {r["oid"]: r["nspname"]
                           for r in self._catalog_rows(PG_NAMESPACE, class_row=class_by_oid.get(PG_NAMESPACE))}
        self.enum_labels = {r["oid"]: r["enumlabel"]
                            for r in self._catalog_rows(PG_ENUM, class_row=class_by_oid.get(PG_ENUM))}
        for t in self.types.values():
            nsp = self.namespaces.get(t.nsp, "")
            t.display_name = t.name if nsp in ("pg_catalog", "") else f"{nsp}.{t.name}"
            if t.category == "A" and t.elem in self.types:
                pass  # 배열 표시 이름은 아래에서
        for t in self.types.values():
            if t.category == "A" and t.elem in self.types and t.name.startswith("_"):
                t.display_name = self.types[t.elem].display_name + "[]"

        self.relations: dict[int, Relation] = {}
        for r in class_rows:
            rel = Relation(
                oid=r["oid"], name=r["relname"], nsp_oid=r["relnamespace"],
                nsp=self.namespaces.get(r["relnamespace"], str(r["relnamespace"])),
                kind=r["relkind"], filenode=r["relfilenode"], tablespace=r["reltablespace"],
                toastrelid=r["reltoastrelid"], persistence=r["relpersistence"], natts=r["relnatts"],
                relpages=r["relpages"], reltuples=r["reltuples"], shared=r["relisshared"],
                attrs=self._attrs.get(r["oid"], []), am=r.get("relam") or 0)
            rel.filenumber = rel.filenode or (self.cluster.global_map if rel.shared else self.local_map).get(rel.oid, 0)
            rel.path = self._relation_path(rel)
            self.relations[rel.oid] = rel
        # 인덱스 정의 (pg_index — 매핑 카탈로그가 아니라 pg_class 로 파일을 찾는다)
        self.indexes: list[dict] = []
        for r in self._catalog_rows(PG_INDEX, class_row=class_by_oid.get(PG_INDEX)):
            keys = r.get("indkey") or []
            self.indexes.append({"index": r["indexrelid"], "table": r["indrelid"], "indkey": keys,
                                 "nkey": r.get("indnkeyatts") or len(keys), "primary": r.get("indisprimary"),
                                 "unique": r.get("indisunique")})
        self.diagnostics.append({"check": "카탈로그 로드", "ok": True,
                                 "detail": f"pg_class {len(class_rows)}행, pg_type {len(self.types)}행, "
                                           f"pg_namespace {len(self.namespaces)}행, pg_enum {len(self.enum_labels)}행"})
        if self._catalog_errors:
            self.diagnostics.append({"check": "카탈로그 행 해석", "ok": False,
                                     "detail": "건너뛴 손상 행 (카탈로그 OID: 개수) " + str(self._catalog_errors)})

    def _fixed(self, v, typid):
        return FIXED[typid](v) if isinstance(v, bytes) else v

    def attrs_for(self, reloid: int) -> list[Attr]:
        return self._attrs.get(reloid, [])

    def _catalog_rows(self, reloid: int, mapped: bool = False, class_row: dict | None = None):
        """카탈로그의 live 행. 손상된 행은 건너뛰고 self._catalog_errors 에 센다 (한 행 때문에 DB 전체가 안 열리지 않게)."""
        key = str(reloid)
        if mapped:
            path = f"base/{self.oid}/{self.local_map[reloid]}"
        else:
            if class_row is None:
                self._catalog_errors[key + " (pg_class 에 없음)"] = 1
                return
            path = f"base/{self.oid}/{class_row['relfilenode']}"
        attrs = self.attrs_for(reloid)
        for t in scan_relation_file(self.cluster, path):
            state, _, _ = classify(t.header, self.cluster.clog, t.block, t.lp)
            if state != "live":
                continue
            try:
                yield self.decode_dict(t, attrs)
            except Exception:
                self._catalog_errors[key] = self._catalog_errors.get(key, 0) + 1

    def decode_dict(self, t: RawTuple, attrs: list[Attr]) -> dict:
        """카탈로그용: 디코딩 실패는 예외로 올린다."""
        d = deform(t.data, t.header, attrs)
        return {a.name: self.convert(v, a.typid) for a, v in zip(attrs, d) if not a.dropped}

    def _relation_path(self, rel: Relation) -> str | None:
        if not rel.has_heap and rel.kind != "i":
            return None
        if rel.filenode:
            fn = rel.filenode
        else:
            fn = (self.cluster.global_map if rel.shared else self.local_map).get(rel.oid)
            if fn is None:
                return None
        if rel.shared or rel.tablespace == GLOBALTABLESPACE_OID:
            return f"global/{fn}"
        ts = rel.tablespace or self.default_tablespace
        if ts == DEFAULTTABLESPACE_OID:
            return f"base/{self.oid}/{fn}"
        catver = self.cluster.control.get("catalog_version_no")
        return f"pg_tblspc/{ts}/PG_{self.cluster.pg_version}_{catver}/{self.oid}/{fn}"

    # ---------------- 값 변환
    def convert(self, datum, typid: int, depth: int = 0):
        if datum is None:
            return None
        if datum is MISSING:
            return None
        t = self.types.get(typid) if hasattr(self, "types") else None
        if t is not None and depth < 10:
            if t.typtype == "d" and t.basetype:
                return self.convert(datum, t.basetype, depth + 1)
            if t.typtype == "e" and isinstance(datum, bytes):
                oid = struct.unpack("<I", datum)[0]
                return self.enum_labels.get(oid, f"<enum oid {oid}>")
        if isinstance(datum, Varlena):
            p = self.detoast(datum.raw)
            if typid in TEXT_TYPES or (t is not None and t.category == "S"):
                return decode_text(p, self.codec)
            fn = VARLENA.get(typid)
            if fn:
                return fn(p)
            if (t is not None and t.category == "A" and t.elem) or typid == ANYARRAY:
                return self._array(p, depth)  # anyarray 는 배열 헤더의 elemtype 으로 원소를 읽는다
            return bytes_display(p)
        if isinstance(datum, Cstring):
            return decode_text(datum.raw, self.codec)
        fn = FIXED.get(typid)
        if fn:
            return fn(datum)
        return bytes_display(datum)

    def _array(self, p: bytes, depth: int):
        """ArrayType: ndim, dataoffset, elemtype, dims[], lbounds[], [null bitmap], data.
        오프셋은 4바이트 헤더를 포함한 varlena 시작 기준이라, p(헤더 뒤) 위치에 4를 더해 정렬한다."""
        ndim, dataoffset, elemtype = struct.unpack_from("<iiI", p, 0)
        if ndim == 0:
            return []
        if not 0 < ndim <= 6:
            raise DecodeError(f"array ndim {ndim}")
        dims = struct.unpack_from(f"<{ndim}i", p, 12)
        nitems = 1
        for d in dims:
            if d < 0:
                raise DecodeError(f"array 차원 {d}")
            nitems *= d
        if nitems > len(p) * 8:  # NULL 하나도 비트맵 1비트는 차지한다 — 이보다 많을 수 없다
            raise DecodeError(f"array 원소 수 {nitems} 가 데이터 크기({len(p)}B)에 비해 너무 많습니다")
        bitmap_pos = 12 + 8 * ndim
        if dataoffset:
            bitmap = p[bitmap_pos:bitmap_pos + (nitems + 7) // 8]
            pos = dataoffset - 4
        else:
            bitmap = None
            pos = ((16 + 8 * ndim + 7) & ~7) - 4  # ARR_OVERHEAD_NONULLS = MAXALIGN(16 + 8*ndim)
        et = self.types.get(elemtype)
        if et is None:
            raise DecodeError(f"array 원소 타입 {elemtype} 미상")
        if bitmap is not None and len(bitmap) < (nitems + 7) // 8:
            raise DecodeError("array NULL 비트맵이 잘렸습니다")
        items = []
        for i in range(nitems):
            if bitmap is not None and not (bitmap[i >> 3] >> (i & 7)) & 1:
                items.append(None)
                continue
            if pos >= len(p) or (et.len > 0 and pos + et.len > len(p)):
                raise DecodeError(f"array 원소 {i} 가 데이터 끝을 넘습니다")
            if et.len == -1:
                size = varlena_size(p, pos)
                items.append(self.convert(Varlena(p[pos:pos + size]), elemtype, depth + 1))
                pos += size
            elif et.len == -2:
                end = p.index(b"\0", pos)
                items.append(decode_text(p[pos:end], self.codec))
                pos = end + 1
            else:
                items.append(self.convert(p[pos:pos + et.len], elemtype, depth + 1))
                pos += et.len
            pos = align(pos + 4, et.align) - 4
        for d in reversed(dims[1:]):  # 다차원 → 중첩 리스트
            items = [items[k:k + d] for k in range(0, len(items), d)]
        return items

    # ---------------- TOAST
    def detoast(self, raw: bytes) -> bytes:
        if varlena_kind(raw) != "external":
            return detoast_inline(raw)
        ext = parse_external(raw)
        chunks = self._toast_index(ext["toastrelid"]).get(ext["valueid"], {})
        chunk_max = self.cluster.control.get("toast_max_chunk_size") or 1996
        expected = -(-ext["extsize"] // chunk_max)
        present = sum(1 for s in range(expected) if s in chunks)
        if present < expected:
            raise VarlenaError(f"TOAST 청크가 남아 있지 않음 — VACUUM 등으로 정리된 것으로 보임 "
                               f"({expected}개 중 {present}개 남음, toastrelid {ext['toastrelid']}, "
                               f"valueid {ext['valueid']}, 원래 크기 {ext['rawsize'] - 4}바이트)")
        return detoast_external_data(ext, b"".join(chunks[s][0] for s in range(expected)))

    def _toast_index(self, toastrelid: int) -> dict:
        wal = self.cluster.wal()  # 락 순서를 지키려고 DB 락보다 먼저 연다
        with self._lock:
            if toastrelid in self._toast:
                return self._toast[toastrelid]
            rel = self.relations.get(toastrelid)
            idx: dict[int, dict[int, tuple[bytes, bool]]] = {}
            if rel and rel.path:
                attrs = rel.attrs
                for t in scan_relation_file(self.cluster, rel.path):
                    state, _, _ = classify(t.header, self.cluster.clog, t.block, t.lp)
                    try:
                        d = deform(t.data, t.header, attrs)
                        cid = struct.unpack("<I", d[0])[0]
                        seq = struct.unpack("<i", d[1])[0]
                        data = detoast_inline(d[2].raw)
                    except Exception:
                        self._toast_errors[toastrelid] = self._toast_errors.get(toastrelid, 0) + 1
                        continue
                    live = state == "live"
                    cur = idx.setdefault(cid, {}).get(seq)
                    # 같은 청크가 여럿이면 live 를 우선 (삭제된 행의 청크도 복구용으로 남겨 둔다)
                    if cur is None or (live and not cur[1]):
                        idx[cid][seq] = (data, live)
            # 파일에서 정리된 청크는 WAL(INSERT 레코드 · 전체 페이지 이미지)에서 보충한다
            if wal is not None and rel is not None and rel.filenumber:
                for wt in wal.tuples.get((0 if rel.shared else self.oid, rel.filenumber), []):
                    try:
                        th = parse_tuple_header(wt.data)
                        d = deform(wt.data, th, rel.attrs)
                        cid = struct.unpack("<I", d[0])[0]
                        seq = struct.unpack("<i", d[1])[0]
                        data = detoast_inline(d[2].raw)
                    except Exception:
                        continue
                    idx.setdefault(cid, {}).setdefault(seq, (data, False))
            self._toast[toastrelid] = idx
            if self._toast_errors.get(toastrelid):
                self.diagnostics.append({"check": f"TOAST {toastrelid} 청크 해석", "ok": False,
                                         "detail": f"해석 못 한 청크 튜플 {self._toast_errors[toastrelid]}개"})
            return idx

    # ---------------- 공개 API
    def relation(self, oid: int) -> Relation:
        return self.relations[oid]

    def relation_by_name(self, schema: str, name: str) -> Relation | None:
        for r in self.relations.values():
            if r.nsp == schema and r.name == name:
                return r
        return None

    def list_relations(self) -> list[dict]:
        out = []
        for r in sorted(self.relations.values(), key=lambda r: (r.nsp, r.name)):
            s = r.summary()
            s["file_size"] = self._file_size(r.path) if r.path else None
            out.append(s)
        return out

    def _file_size(self, path: str) -> int | None:
        src = self.cluster.source
        if not src.exists(path):
            return None
        total, seg = 0, 0
        while True:
            p = path if seg == 0 else f"{path}.{seg}"
            if not src.exists(p):
                return total
            total += src.size(p)
            seg += 1

    def columns(self, oid: int) -> list[dict]:
        return [a.to_dict(self.types) for a in self.relations[oid].attrs]

    def make_row(self, rel: Relation, data: bytes, th: TupleHeader, base: dict) -> dict:
        """튜플 바이트 → 행 dict. base(rid · source · tid · state · 위치 …)에 값과 오류를 더한다."""
        row = {**base, "xmin": th.xmin, "xmax": th.xmax, "ctid": [th.ctid_block, th.ctid_pos],
               "new_version": bool(th.infomask & HEAP_UPDATED), "values": [], "errors": {}}
        datums: list = []
        try:
            deform(data, th, rel.attrs, out=datums)
        except Exception as e:
            # 앞쪽에서 분해된 컬럼은 살리고, 나머지는 '해석 불가'로 표시한다 (NULL 과 구분)
            row["errors"]["*"] = f"튜플 분해 실패 ({len(datums)}번째 컬럼 이후): {e}"
            for a in rel.attrs[len(datums):]:
                row["errors"][a.name] = "해석 불가 (튜플 분해 실패)"
            datums += [None] * (len(rel.attrs) - len(datums))
        for a, d in zip(rel.attrs, datums):
            if d is MISSING:
                row["values"].append(None)
                row["errors"][a.name] = "missing (튜플 natts 밖 — 기본값/NULL)"
                continue
            try:
                row["values"].append(self.convert(d, a.typid) if not a.dropped else self._dropped(d))
            except Exception as e:
                row["values"].append(None)
                row["errors"][a.name] = f"{type(e).__name__}: {e}"
        return row

    def _built(self, oid: int) -> tuple[list[dict], dict, dict]:
        """(행, rid → 원시 튜플, 부가 정보) — 릴레이션별 락으로 같은 테이블을 동시에 두 번 만들지 않는다."""
        with self._lock:
            if oid in self._rows_cache:
                self._rows_cache.move_to_end(oid)
                return self._rows_cache[oid]
            rel_lock = self._rel_locks.setdefault(oid, threading.Lock())
        rel = self.relations[oid]
        if not rel.has_heap or not rel.path:
            raise ValueError(f"{rel.nsp}.{rel.name}: 힙 데이터가 없는 릴레이션입니다 ({RELKIND_NAMES.get(rel.kind)})")
        self.cluster.wal()  # 락 순서: WAL 을 먼저 (DB 락을 잡은 채 WAL 을 기다리지 않게)
        from .recovery import build_rows
        with rel_lock:
            with self._lock:
                if oid in self._rows_cache:
                    return self._rows_cache[oid]
            built = build_rows(self, rel)
        with self._lock:
            self._rows_cache[oid] = built
            while len(self._rows_cache) > self.ROW_CACHE:
                self._rows_cache.popitem(last=False)
        return built

    def rows(self, oid: int, sources: set[str] | None = None) -> list[dict]:
        """릴레이션의 모든 행. 캐시한다.
        source: heap(현재 파일의 튜플) · carved(빈 공간에서 찾은 옛 튜플) · wal(WAL 에만 남은 튜플).
        state : live · deleted · updated(구버전) · aborted · in_progress · unknown."""
        rows = self._built(oid)[0]
        return rows if sources is None else [r for r in rows if r["source"] in sources]

    def raw_tuple(self, oid: int, rid: str) -> tuple[bytes, dict]:
        raw = self._built(oid)[1]
        if rid not in raw:
            raise KeyError(f"행 {rid} 없음")
        return raw[rid]

    def _dropped(self, d):
        # 삭제된 컬럼은 타입 정보가 없다 (atttypid = 0). 원시 바이트만 보여 준다.
        if d is None:
            return None
        raw = d.raw if isinstance(d, (Varlena, Cstring)) else d
        return bytes_display(raw)

    def _page_info(self, rel: Relation, block: int) -> dict | None:
        bs = self.cluster.blcksz
        seg, bi = divmod(block, self.cluster.relseg_size)
        p = rel.path if seg == 0 else f"{rel.path}.{seg}"
        if not self.cluster.source.exists(p):
            return None
        page = self.cluster.source.read(p, bi * bs, bs)
        if len(page) < bs:
            return None
        h = parse_page_header(page)
        return {"lsn": f"{h.lsn >> 32:X}/{h.lsn & 0xFFFFFFFF:X}", "checksum": h.checksum, "flags": h.flags,
                "lower": h.lower, "upper": h.upper, "special": h.special, "prune_xid": h.prune_xid,
                "items": h.n_items}

    def row_detail(self, oid: int, rid: str) -> dict:
        """행 하나의 출처 · 튜플 헤더 · 컬럼별 위치 · 원시 바이트 · 커밋 시각 · 버전 이력."""
        rel = self.relations[oid]
        rows = self._built(oid)[0]
        row = next((r for r in rows if r["rid"] == rid), None)
        if row is None:
            raise KeyError(f"행 {rid} 없음")
        tup, meta = self.raw_tuple(oid, rid)
        th = parse_tuple_header(tup)
        out = {"rid": rid, "source": row["source"], "state": row["state"], "tid": row["tid"], **meta,
               "header": th.to_dict(), "xmin_status": row["xmin_status"], "xmax_status": row["xmax_status"],
               "hex": tup.hex(), "wal": row.get("wal"), "carve": row.get("carve")}
        if row["source"] in ("heap", "carved"):
            out["page"] = self._page_info(rel, row["tid"][0])
        w = self.cluster.wal()
        if w is not None:
            out["xmin_xact"] = w.xact(th.xmin)
            fate = (row.get("wal") or {}).get("fate") or {}
            xmax = row["xmax"] if row["xmax_status"] == "committed" else (th.xmax or fate.get("xid") or 0)
            out["xmax_xact"] = w.xact(xmax) if xmax and row["xmax_status"] == "committed" else None
        spans, pos_list, datums = [], [], []
        try:
            deform(tup, th, rel.attrs, pos_list, out=datums)
        except Exception as e:
            out["deform_error"] = str(e)
        for a, d, (start, size) in zip(rel.attrs, datums, pos_list):
            info = {"name": a.name, "start": start, "len": size,
                    "null": d is None, "missing": d is MISSING, "dropped": a.dropped}
            if isinstance(d, Varlena):
                info["varlena"] = varlena_kind(d.raw)
                if info["varlena"] == "external":
                    info["toast"] = parse_external(d.raw)
            spans.append(info)
        out["spans"] = spans
        # 버전 이력: 같은 체인의 행들 (오래된 것부터)
        ch = row.get("chain")
        if ch:
            members = sorted((r for r in rows if (r.get("chain") or {}).get("id") == ch["id"]),
                             key=lambda r: r["chain"]["pos"])
            out["versions"] = [{k: r.get(k) for k in ("rid", "source", "state", "tid", "xmin", "xmax",
                                                      "xmin_time", "xmax_time", "values", "chain")}
                               for r in members]
        return out

    def tuple_detail(self, oid: int, block: int, lp: int) -> dict:
        """현재 힙 파일의 튜플 (block, lp) 상세 — row_detail 의 축약."""
        return self.row_detail(oid, f"h:{block}:{lp}")

    def stats(self, oid: int) -> dict:
        counts: dict[str, int] = {}
        sources: dict[str, int] = {}
        errors = chains = 0
        for r in self.rows(oid):
            counts[r["state"]] = counts.get(r["state"], 0) + 1
            sources[r["source"]] = sources.get(r["source"], 0) + 1
            if r["errors"]:
                errors += 1
            if (r.get("chain") or {}).get("pos") == 1:
                chains += 1
        extra = self._built(oid)[2]
        return {"total": sum(counts.values()), "by_state": counts, "by_source": sources,
                "rows_with_errors": errors, "chains": chains, "damaged_pages": extra["problems"],
                "carve_duplicates": extra["carve_rejected"]["duplicate"]}

    def index_traces(self, oid: int) -> dict:
        from .recovery import index_traces
        return index_traces(self, self.relations[oid])

    def key_index(self) -> dict:
        """기본키(단일 컬럼) 값 → 그 값을 가진 행들. 다른 테이블의 같은 값을 링크로 이을 때 쓴다."""
        from .recovery import key_index
        with self._key_lock:  # DB 락이 아니라 전용 락 — 만드는 동안 다른 테이블 조회를 막지 않는다
            if self._key_index is None:
                self._key_index = key_index(self)
            return self._key_index

    def wal_events(self, oid: int) -> list[dict]:
        rel = self.relations[oid]
        w = self.cluster.wal()
        if w is None or not rel.filenumber:
            return []
        evs = w.events.get((0 if rel.shared else self.oid, rel.filenumber), [])
        out = []
        for e in evs:
            x = w.xact(e["xid"]) if e["xid"] else None
            out.append({**{k: v for k, v in e.items() if k != "lsn_int"},
                        "xact_status": x["status"] if x else None, "time": x["time"] if x else None})
        return out
