"""웹 API (FastAPI). 엔진을 호출만 하고 해석 로직은 두지 않는다.

    uvicorn pgviewer.api.main:app --host 127.0.0.1 --port 8000

로컬 전용이다. 경로 입력 API 가 이 PC 의 파일을 읽으므로 다음을 막는다.
  - 다른 Host 헤더 (DNS rebinding) → 400
  - 다른 출처(Origin)에서 온 POST/DELETE (CSRF) → 403
  - UNC 경로(\\\\server\\share) — 여는 순간 NTLM 인증이 외부로 나갈 수 있다
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from ..engine.postgres import Cluster, open_cluster
from ..source import SourceError

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
UPLOADS = ROOT / "work" / "uploads"
MAX_UPLOAD = 8 << 30  # 8 GiB
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
JS_SAFE_INT = 2 ** 53

app = FastAPI(title="PostgreSQL Viewer", version="0.1.0")


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    if host not in ALLOWED_HOSTS:
        return JSONResponse({"detail": "로컬 전용 도구입니다 (Host 불일치)"}, status_code=400)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).hostname not in {"127.0.0.1", "localhost", "::1"}:
            return JSONResponse({"detail": "다른 출처의 요청은 받지 않습니다"}, status_code=403)
    resp = await call_next(request)
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
    return resp


def json_safe(v):
    """JSON·JavaScript 로 보낼 때 값이 바뀌지 않게: NaN/Infinity 와 2^53 을 넘는 정수는 문자열로."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v) if abs(v) >= JS_SAFE_INT else v
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    return str(v)


class Session:
    def __init__(self, cluster: Cluster, upload_dir: Path | None = None, upload_hash: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.cluster = cluster
        self.upload_dir = upload_dir
        self.opened_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.hash: dict | None = None
        self.hash_error: str | None = None
        self.upload_hash = upload_hash
        threading.Thread(target=self._hash, daemon=True).start()

    def _hash(self):
        try:
            self.hash = self.cluster.source.input_hash()
        except Exception as e:  # 해시 실패는 열람을 막지 않는다
            self.hash_error = str(e)

    def summary(self) -> dict:
        info = self.cluster.info()
        return {"id": self.id, "input": info["input"], "input_kind": info["input_kind"],
                "pgdata": info["pgdata"], "pg_version": info["pg_version"], "uploaded": self.upload_dir is not None,
                "opened_at": self.opened_at, "hash": self.hash, "hash_error": self.hash_error,
                "upload_hash": self.upload_hash}

    def close(self):
        self.cluster.close()
        if self.upload_dir is not None:
            shutil.rmtree(self.upload_dir, ignore_errors=True)


SESSIONS: dict[str, Session] = {}
_lock = threading.Lock()


def _session(sid: str) -> Session:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "세션이 없습니다")
    return s


def _db(sid: str, dboid: int):
    try:
        return _session(sid).cluster.database(dboid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(422, f"데이터베이스 {dboid} 해석 실패: {type(e).__name__}: {e}")


def _rel(db, oid: int):
    if oid not in db.relations:
        raise HTTPException(404, f"릴레이션 {oid} 없음")
    return db.relations[oid]


def _check_path(path: str) -> str:
    if path.startswith(("\\\\", "//")):
        raise HTTPException(400, "네트워크(UNC) 경로는 열지 않습니다 — 로컬로 복사한 뒤 여세요")
    p = Path(path)
    if p.is_dir() and p == Path(p.anchor):
        raise HTTPException(400, "드라이브 루트는 열 수 없습니다 — PGDATA 폴더나 그 상위 폴더를 지정하세요")
    return path


def _open(path: str, upload_dir: Path | None = None, upload_hash=None) -> Session:
    cluster = None
    try:
        cluster = open_cluster(path)
        cluster.databases()  # 카탈로그를 미리 읽어 오류를 여기서 드러낸다
    except Exception as e:
        if cluster is not None:
            cluster.close()
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        if isinstance(e, SourceError):
            raise HTTPException(400, str(e))
        raise HTTPException(422, f"PGDATA 해석 실패: {type(e).__name__}: {e}")
    s = Session(cluster, upload_dir, upload_hash)
    with _lock:
        SESSIONS[s.id] = s
    return s


# ------------------------------------------------------------------ 세션

class OpenReq(BaseModel):
    path: str


@app.post("/api/open")
def open_path(req: OpenReq):
    return _open(_check_path(req.path.strip().strip('"'))).summary()


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    name = re.sub(r"[^\w.\-]", "_", Path(file.filename or "").name)
    if not name.lower().endswith(".zip"):
        raise HTTPException(400, "zip 파일만 올릴 수 있습니다")
    dest_dir = UPLOADS / uuid.uuid4().hex[:12]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ("upload_" + name)  # 접두어로 Windows 예약 이름(con.zip 등)을 피한다
    h = hashlib.sha256()
    total = 0
    try:
        with open(dest, "wb") as f:
            while chunk := file.file.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD:
                    raise HTTPException(413, f"업로드 한도 {MAX_UPLOAD >> 30} GiB 초과 — 경로 입력으로 여세요")
                h.update(chunk)
                f.write(chunk)
    except BaseException:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    return _open(str(dest), dest_dir, h.hexdigest()).summary()


@app.get("/api/sessions")
def sessions():
    with _lock:
        items = list(SESSIONS.values())
    return [s.summary() for s in items]


@app.delete("/api/s/{sid}")
def close(sid: str):
    with _lock:
        s = SESSIONS.pop(sid, None)
    if not s:
        raise HTTPException(404, "세션이 없습니다")
    s.close()
    return {"ok": True}


@app.get("/api/s/{sid}")
def info(sid: str):
    s = _session(sid)
    return json_safe({"session": s.summary(), "cluster": s.cluster.info(), "databases": s.cluster.databases()})


@app.post("/api/s/{sid}/verify")
def verify(sid: str):
    """입력 해시를 다시 계산해 열었을 때 값과 비교한다 (원본이 바뀌지 않았는지)."""
    s = _session(sid)
    if not s.hash:
        raise HTTPException(409, "처음 해시가 아직 계산되지 않았습니다")
    now = s.cluster.source.input_hash()
    return {"before": s.hash, "now": now, "same": now["value"] == s.hash["value"]}


# ------------------------------------------------------------------ 데이터베이스 · 릴레이션

@app.get("/api/s/{sid}/wal")
def wal_summary(sid: str):
    return json_safe(_session(sid).cluster.wal_summary())


@app.get("/api/s/{sid}/db/{dboid}")
def database(sid: str, dboid: int):
    db = _db(sid, dboid)
    return json_safe({"oid": db.oid, "diagnostics": db.diagnostics, "relations": db.list_relations()})


def _time_columns(db, rel, rows) -> dict:
    """int8 컬럼 중 값 대부분이 epoch 밀리초(1e12~1e13)·마이크로초(1e15~1e16)인 것 → {컬럼 번호: 단위}.
    시각을 정수로 저장하는 앱을 위한 표시 힌트다 (값 자체는 바꾸지 않는다)."""
    out = {}
    for i, a in enumerate(rel.attrs):
        if a.dropped or a.typid != 20:
            continue
        vals = [r["values"][i] for r in rows[:2000] if isinstance(r["values"][i], int) and r["values"][i] > 0]
        if not vals:
            continue
        for unit, lo, hi in (("ms", 10 ** 12, 10 ** 13), ("us", 10 ** 15, 10 ** 16)):
            if sum(lo <= v < hi for v in vals) >= 0.8 * len(vals):
                out[i] = unit
                break
    return out


@app.get("/api/s/{sid}/db/{dboid}/rel/{oid}")
def relation(sid: str, dboid: int, oid: int):
    db = _db(sid, dboid)
    rel = _rel(db, oid)
    out = {"relation": rel.summary(), "columns": db.columns(oid), "stats": None, "error": None,
           "time_columns": {}, "indexes": [db.relations[i["index"]].name for i in db.indexes
                                           if i["table"] == oid and i["index"] in db.relations]}
    if rel.has_heap and rel.path:
        try:
            out["stats"] = db.stats(oid)
            out["time_columns"] = _time_columns(db, rel, db.rows(oid))
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
    return json_safe(out)


def _search_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)


def _filtered(db, oid: int, states: str | None, q: str | None, sort: int | None, desc: bool,
              sources: str | None = None, rids: str | None = None, chained: bool = False):
    try:
        rows = db.rows(oid)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if rids:
        want = set(rids.split(","))
        return [r for r in rows if r["rid"] in want]
    if sources:
        want = set(sources.split(","))
        rows = [r for r in rows if r["source"] in want]
    if states:
        want = set(states.split(","))
        rows = [r for r in rows if r["state"] in want]
    if chained:
        rows = [r for r in rows if r.get("chain")]
    if q:
        ql = q.lower()
        rows = [r for r in rows if any(ql in _search_text(v).lower() for v in r["values"])]
    if sort is not None:
        def key(r):
            v = r["values"][sort] if 0 <= sort < len(r["values"]) else None
            if isinstance(v, bool):
                return (0, int(v))
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                return (0, v)
            return (1, _search_text(v))

        def isnull(r):
            return not 0 <= sort < len(r["values"]) or r["values"][sort] is None
        rows = sorted((r for r in rows if not isnull(r)), key=key, reverse=desc) + [r for r in rows if isnull(r)]
    elif chained:
        rows = sorted(rows, key=lambda r: (r["chain"]["id"], r["chain"]["pos"]))
    return rows


def _links(db, oid: int, rows: list[dict]) -> None:
    """문자열 값이 다른 테이블(또는 같은 테이블의 다른 컬럼)의 기본키와 같으면 그 행으로 가는 링크를 단다.
    같은 테이블의 같은 키 컬럼(= 같은 행의 다른 버전)은 수정 이력으로 보이므로 링크하지 않는다."""
    try:
        ki = db.key_index()
    except Exception:
        return
    names = [a.name for a in db.relations[oid].attrs]
    for r in rows:
        links = {}
        for i, v in enumerate(r["values"]):
            if isinstance(v, str) and len(v) >= 8:
                targets = [t for t in ki.get(v, [])
                           if t["rid"] != r["rid"] and not (t["oid"] == oid and t["column"] == names[i])]
                if targets:
                    links[i] = targets[:20]
        r["links"] = links


@app.get("/api/s/{sid}/db/{dboid}/rel/{oid}/rows")
def rows(sid: str, dboid: int, oid: int, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
         states: str | None = None, q: str | None = None, sort: int | None = None, desc: bool = False,
         sources: str | None = None, rids: str | None = None, chained: bool = False):
    db = _db(sid, dboid)
    _rel(db, oid)
    rs = _filtered(db, oid, states, q, sort, desc, sources, rids, chained)
    page = [dict(r) for r in rs[offset:offset + limit]]
    _links(db, oid, page)
    return json_safe({"total": len(rs), "offset": offset, "rows": page})


@app.get("/api/s/{sid}/db/{dboid}/rel/{oid}/row")
def row_detail(sid: str, dboid: int, oid: int, rid: str):
    db = _db(sid, dboid)
    _rel(db, oid)
    try:
        return json_safe(db.row_detail(oid, rid))
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/s/{sid}/db/{dboid}/rel/{oid}/tuple/{block}/{lp}")
def tuple_detail(sid: str, dboid: int, oid: int, block: int, lp: int):
    return row_detail(sid, dboid, oid, f"h:{block}:{lp}")


@app.get("/api/s/{sid}/db/{dboid}/rel/{oid}/wal")
def wal_events(sid: str, dboid: int, oid: int, op: str | None = None,
               offset: int = Query(0, ge=0), limit: int = Query(3000, ge=1, le=20000)):
    db = _db(sid, dboid)
    _rel(db, oid)
    evs = db.wal_events(oid)
    ops: dict[str, int] = {}
    for e in evs:
        ops[e["op"]] = ops.get(e["op"], 0) + 1
    if op:
        evs = [e for e in evs if e["op"] == op]
    return json_safe({"summary": db.cluster.wal_summary(), "ops": ops, "total": len(evs),
                      "offset": offset, "events": evs[offset:offset + limit]})


@app.get("/api/s/{sid}/db/{dboid}/rel/{oid}/index-traces")
def index_traces(sid: str, dboid: int, oid: int):
    db = _db(sid, dboid)
    rel = _rel(db, oid)
    if not rel.has_heap or not rel.path:
        raise HTTPException(400, "힙 테이블이 아닙니다")
    return json_safe(db.index_traces(oid))


def _disposition(name: str) -> str:
    ascii_name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "export"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name, safe='')}"


def _csv_cell(v):
    """CSV 셀. 객체는 JSON 으로. 수식으로 해석될 문자열(= + - @ 탭 CR)은 앞에 ' 를 붙인다
    (스프레드시트에서 열 때 수식 실행 방지). 원본 그대로가 필요하면 JSON 내보내기를 쓴다."""
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False)
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


META_COLS = ["_rid", "_source", "_state", "_tid", "_xmin", "_xmin_time", "_xmax", "_xmax_time", "_chain",
             "_file", "_file_offset", "_wal_lsn"]


def _meta(r: dict) -> list:
    ch = r.get("chain")
    return [r["rid"], r["source"], r["state"], r["tid"], r["xmin"], r.get("xmin_time"), r["xmax"],
            r.get("xmax_time"), f"{ch['id']}#{ch['pos']}/{ch['len']}" if ch else None, r["file"], r["offset"],
            (r.get("wal") or {}).get("lsn")]


@app.get("/api/s/{sid}/db/{dboid}/rel/{oid}/export")
def export(sid: str, dboid: int, oid: int, format: str = "csv", states: str | None = None,
           q: str | None = None, sources: str | None = None, chained: bool = False):
    db = _db(sid, dboid)
    rel = _rel(db, oid)
    rs = _filtered(db, oid, states, q, None, False, sources, None, chained)
    cols = [a.name for a in rel.attrs]
    fname = f"{rel.nsp}.{rel.name}"
    if format == "json":
        recs = []
        for r in rs:
            rec = {**dict(zip(META_COLS, _meta(r))), **dict(zip(cols, r["values"]))}
            if r.get("wal"):
                rec["_wal"] = r["wal"]
            if r["errors"]:
                rec["_errors"] = r["errors"]
            recs.append(rec)
        # JSON 파일은 큰 정수를 그대로 둔다 (JSON 자체는 정밀도 제한이 없다). NaN/Infinity 만 문자열로.
        body = json.dumps(_nan_safe(recs), ensure_ascii=False, indent=1, default=str, allow_nan=False)
        return Response(body, media_type="application/json",
                        headers={"Content-Disposition": _disposition(fname + ".json")})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(META_COLS + cols)
    for r in rs:
        m = _meta(r)
        m[3] = f"({r['tid'][0]},{'' if r['tid'][1] is None else r['tid'][1]})"
        w.writerow(m + [_csv_cell(v) for v in r["values"]])
    return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": _disposition(fname + ".csv")})


def _nan_safe(v):
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, list):
        return [_nan_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _nan_safe(x) for k, x in v.items()}
    return v


# ------------------------------------------------------------------ 화면

@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
