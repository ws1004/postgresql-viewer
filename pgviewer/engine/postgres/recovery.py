"""복구 계층 — 현재 힙의 행에 '사라진 행'을 더하고, 행 사이의 관계(수정 이력)를 잇는다.

  heap    현재 파일에서 라인 포인터가 가리키는 튜플 (live · deleted · updated …)
  wal     WAL 의 전체 페이지 이미지 · INSERT/UPDATE 레코드에만 남은 튜플 (wal.py)
  carved  라인 포인터가 가리키지 않는 페이지 빈 공간에서 찾은 옛 튜플 (carve.py)

처리 순서는 heap → wal → carved 다. 같은 튜플(같은 블록 · xmin · 비트맵+데이터)이 여럿이면 앞의 것만 남긴다
(WAL 쪽이 위치 · 시각 · 결말 정보가 더 많다).

수정 이력(버전 체인): 튜플 A 의 xmax 가 커밋됐고 A.ctid(또는 WAL UPDATE 의 새 위치)에 xmin == A.xmax 인 튜플 B 가
있으면 A → B 다. PostgreSQL 이 업데이트 체인을 따라갈 때 쓰는 규칙과 같다 (heap_get_latest_tid 의 priorXmax 검사).
빈 공간 행은 원래 슬롯을 모르므로 후보가 정확히 하나일 때만 '추정' 연결한다.
"""

from __future__ import annotations

import math

from .btree import leaf_items
from .carve import carve_page
from .heap import (LP_DEAD, LP_NORMAL, LP_REDIRECT, TUPLE_HEADER_SIZE, classify, line_pointers, page_problem,
                   parse_page_header, parse_tuple_header)
from .types import TEXT_TYPES
from .varlena import detoast_inline, parse_external, varlena_kind

SOURCE_RANK = {"heap": 0, "wal": 1, "carved": 2}


def _content_key(block: int, data: bytes, xmin: int) -> tuple:
    # 비트맵부터 끝까지 (NULL 위치가 다른 행을 같은 것으로 보지 않게)
    return (block, xmin, data[TUPLE_HEADER_SIZE:])


def _set_times(wal, r: dict):
    def t(xid):
        if wal is None or not xid:
            return None
        x = wal.xact(xid)
        return x["time"] if x else None
    r["xmin_time"] = t(r["xmin"])
    # 잠금 전용·MultiXact xmax 는 삭제가 아니고 MultiXactId 는 xid 와 번호 공간이 다르다 — 커밋된 xmax 만
    r["xmax_time"] = t(r["xmax"]) if r["xmax"] and r["xmax_status"] == "committed" else None


def build_rows(db, rel) -> tuple[list[dict], dict, dict]:
    """(행 목록, rid → (튜플 바이트, 위치 정보), 부가 정보{lp, problems, carve_rejected})."""
    from .cluster import MISSING, Varlena, deform, iter_pages

    cl = db.cluster
    bs = cl.blcksz
    clog = cl.clog
    wal = cl.wal()
    # 카빙 xmin 상한: 체크포인트의 nextXid 와 WAL 에서 본 최대 xid 중 큰 값 (체크포인트 뒤의 트랜잭션도 받는다)
    next_xid = max(cl.control.get("ckpt_next_xid_xid") or 0, (wal.max_xid + 1) if wal is not None else 0) or None
    rows: list[dict] = []
    raw: dict[str, tuple[bytes, dict]] = {}
    heap_by_tid: dict[tuple[int, int], dict] = {}
    lp_info: dict[tuple[int, int], tuple[int, int]] = {}  # (블록, lp) → (flags, lp_off) — REDIRECT 는 lp_off 가 대상 lp
    known: set[tuple] = set()
    problems = []

    # ---- 1. 현재 힙
    pages = []
    for block, page, seg_path, foff in iter_pages(cl, rel.path):
        h = parse_page_header(page)
        prob = page_problem(h, bs)
        if prob:
            if prob != "new":
                problems.append({"block": block, "file": seg_path, "file_offset": foff, "problem": prob})
            continue
        pages.append((block, page, h, seg_path, foff))
        for lp, off, flags, length in line_pointers(page, h):
            lp_info[(block, lp)] = (flags, off)
            if flags != LP_NORMAL or length < TUPLE_HEADER_SIZE or off + length > bs:
                continue
            data = page[off:off + length]
            th = parse_tuple_header(data)
            state, smin, smax = classify(th, clog, block, lp)
            rid = f"h:{block}:{lp}"
            row = db.make_row(rel, data, th, {
                "rid": rid, "source": "heap", "tid": [block, lp], "state": state,
                "xmin_status": smin, "xmax_status": smax, "file": seg_path, "offset": foff + off,
                "length": length})
            rows.append(row)
            heap_by_tid[(block, lp)] = row
            raw[rid] = (data, {"file": seg_path, "file_offset": foff + off, "block": block, "lp": lp,
                               "lp_off": off, "lp_len": length})
            known.add(_content_key(block, data, th.xmin))

    # ---- 2. WAL 에만 남은 튜플
    ev_by_xid: dict[tuple[int, int], dict] = {}  # (블록, xid) → DELETE/UPDATE 이벤트 (빈 공간 행 판정용)
    if wal is not None and rel.filenumber:
        key = (0 if rel.shared else db.oid, rel.filenumber)
        for e in wal.events.get(key, []):
            if e["op"] in ("DELETE", "UPDATE", "HOT_UPDATE"):
                ev_by_xid.setdefault((e["block"], e["xid"]), e)
        _wal_rows(db, rel, wal, key, rows, raw, heap_by_tid, known, clog)

    # ---- 3. 빈 공간 카빙
    codec = db.codec
    natts_max = len(rel.attrs)
    rejected = {"duplicate": 0}

    def try_deform(region: bytes, th) -> int | None:
        spans, datums = [], []
        try:
            deform(region, th, rel.attrs, spans, out=datums)
        except Exception:
            return None
        end = th.hoff
        for a, d, (s, n) in zip(rel.attrs, datums, spans):
            if s is not None:
                end = max(end, s + n)
            if d is None or d is MISSING or a.dropped:
                continue
            try:
                if isinstance(d, Varlena):
                    if varlena_kind(d.raw) == "external":
                        if parse_external(d.raw)["toastrelid"] != rel.toastrelid:
                            return None
                        continue
                    p = detoast_inline(d.raw)
                    t = db.types.get(a.typid)
                    if a.typid in TEXT_TYPES or (t is not None and t.category == "S"):
                        p.decode(codec)  # 엄격하게 — 깨진 텍스트면 오탐으로 본다
                    else:
                        db.convert(d, a.typid)
                else:
                    db.convert(d, a.typid)
            except Exception:
                return None
        return end

    has_varlena = any(a.len == -1 and not a.dropped for a in rel.attrs)
    if natts_max:
        for block, page, h, seg_path, foff in pages:
            for off, data, th in carve_page(page, h, bs, natts_max, next_xid, try_deform):
                ck = _content_key(block, data, th.xmin)
                if ck in known:
                    rejected["duplicate"] += 1  # 정리 중에 옮겨진 튜플의 사본, 또는 WAL 로 이미 복구한 튜플
                    continue
                known.add(ck)
                state, smin, smax, note = _carved_state(th, block, clog, heap_by_tid, ev_by_xid)
                rid = f"c:{block}:{off}"
                carve = {"page_offset": off, "note": note,
                         # 가변 길이 컬럼이 없으면 검증이 헤더 조건뿐이라 신뢰도가 낮다
                         "confidence": "normal" if has_varlena else "low"}
                row = db.make_row(rel, data, th, {
                    "rid": rid, "source": "carved", "tid": [block, None], "state": state,
                    "xmin_status": smin, "xmax_status": smax, "file": seg_path, "offset": foff + off,
                    "length": len(data), "carve": carve})
                rows.append(row)
                raw[rid] = (data, {"file": seg_path, "file_offset": foff + off, "block": block,
                                   "page_offset": off, "length": len(data)})

    # ---- 4. 커밋 시각 · 수정 이력
    for r in rows:
        _set_times(wal, r)
    _link_chains(rows)
    return rows, raw, {"lp": lp_info, "problems": problems, "carve_rejected": rejected}


def _carved_state(th, block, clog, heap_by_tid, ev_by_xid):
    """빈 공간 행의 상태. 원래 라인 포인터를 모르므로 ctid 만으로는 DELETE(ctid=자기)와 같은 페이지 안의
    UPDATE 를 구분할 수 없다 — WAL 이벤트나 새 버전의 존재로 확인되면 확정, 아니면 '불확실'을 붙인다."""
    state, smin, smax = classify(th, clog, block, -1)  # 일단 '자기 자신이 아님'으로 판정 (xmax 커밋 → updated)
    note = None
    if state == "updated":
        ev = ev_by_xid.get((block, th.xmax))
        target = heap_by_tid.get((th.ctid_block, th.ctid_pos))
        if ev is not None:
            state = "deleted" if ev["op"] == "DELETE" else "updated"
            if ev.get("super"):
                state, note = "aborted", "충돌로 취소된 투기적 INSERT (WAL)"
            else:
                note = f"WAL {ev['op']} @ {ev['lsn']} 로 확인"
        elif th.ctid_block != block or (target is not None and target["xmin"] == th.xmax):
            state = "updated"
        else:
            state, note = "deleted", "삭제/같은 페이지 안 수정 구분 불확실 (WAL 기록 없음)"
    elif state == "live":
        state, note = "unknown", "파일에서 빠졌는데 삭제 기록이 없다"
    return state, smin, smax, note


def _wal_rows(db, rel, wal, key, rows, raw, heap_by_tid, known, clog):
    from .wal import lsn_str

    tuples = wal.tuples.get(key, [])
    events = wal.events.get(key, [])
    if not tuples:
        return
    # (block, offnum, xmin) 별로 하나: 가장 늦은 FPI(헤더가 가장 최신)를, 없으면 가장 이른 레코드를 쓴다
    groups: dict[tuple, dict] = {}
    for t in tuples:
        th = parse_tuple_header(t.data)
        g = groups.get((t.block, t.offnum, th.xmin))
        if g is None:
            groups[(t.block, t.offnum, th.xmin)] = {"first_lsn": t.lsn, "sources": {t.source}, "best": t, "best_th": th}
            continue
        g["sources"].add(t.source)
        g["first_lsn"] = min(g["first_lsn"], t.lsn)
        b = g["best"]
        better = (t.source == "fpi" and (b.source != "fpi" or t.lsn > b.lsn)) or \
                 (t.source != "fpi" and b.source != "fpi" and t.lsn < b.lsn)
        if better:
            g["best"], g["best_th"] = t, th
    slot_events: dict[tuple[int, int], list] = {}
    for e in events:
        if e["op"] in ("DELETE", "UPDATE", "HOT_UPDATE") and e.get("offnum"):
            slot_events.setdefault((e["block"], e["offnum"]), []).append(e)
    slot_tuples: dict[tuple[int, int], list] = {}
    for (b, o, xmin), g in groups.items():
        slot_tuples.setdefault((b, o), []).append((g["first_lsn"], xmin))

    def same_xact(a: int, b: int) -> bool:
        if a == b:
            return True
        xa, xb = wal.xact(a), wal.xact(b)
        return bool(xa and xb and xa["top"] == xb["top"])  # 서브트랜잭션

    def fate_of(b, o, xmin, first_lsn, th):
        """이 튜플을 지우거나 바꾼 이벤트: 같은 슬롯 · 튜플이 나타난 뒤 · 다음 점유자가 나타나기 전 · 커밋됨.
        헤더에 xmax 가 있으면 그 트랜잭션의 이벤트만 받는다."""
        later = sorted(l for l, x in slot_tuples.get((b, o), []) if l > first_lsn and x != xmin)
        limit = later[0] if later else None
        hdr_xmax = th.xmax if th.xmax and smax_real(th) else 0
        for e in slot_events.get((b, o), []):
            if e["lsn_int"] < first_lsn or (limit is not None and e["lsn_int"] > limit):
                continue  # 같은 LSN 은 받는다 (그 레코드에 실린 FPI 로 처음 나타난 튜플)
            if hdr_xmax and not same_xact(e["xid"], hdr_xmax):
                continue
            if clog.status(e["xid"]) != "committed":
                continue
            x = wal.xact(e["xid"])
            f = {"op": "SUPER_DELETE" if e.get("super") else e["op"], "lsn": e["lsn"], "xid": e["xid"],
                 "time": x["time"] if x else None}
            if e["op"] != "DELETE":
                f["new_tid"] = [e.get("new_block"), e.get("new_offnum")]
            return f
        return None

    def smax_real(th) -> bool:
        from .heap import HEAP_XMAX_INVALID, HEAP_XMAX_IS_MULTI, xmax_locked_only
        return not (th.infomask & (HEAP_XMAX_INVALID | HEAP_XMAX_IS_MULTI) or xmax_locked_only(th.infomask))

    for (b, o, xmin), g in sorted(groups.items(), key=lambda kv: kv[1]["first_lsn"]):
        h = heap_by_tid.get((b, o))
        if h is not None and h["xmin"] == xmin:
            continue  # 지금 파일에도 있는 튜플
        t, th = g["best"], g["best_th"]
        ck = _content_key(b, t.data, th.xmin)
        if ck in known:
            continue
        known.add(ck)
        state, smin, smax = classify(th, clog, b, o)
        fate = fate_of(b, o, xmin, g["first_lsn"], th)
        if smin in ("aborted", "uncommitted"):
            state = "aborted"
        elif state in ("deleted", "updated"):
            pass  # 헤더(FPI)의 판정이 우선 — 결말은 시각·위치 정보로만 쓴다
        elif fate:
            state = {"DELETE": "deleted", "SUPER_DELETE": "aborted"}.get(fate["op"], "updated")
        elif state == "live":
            state = "unknown"  # WAL 이후 사라졌지만 이유가 기록에 없다 (VACUUM FULL · TRUNCATE · 재작성 등)
        rid = f"w:{b}:{o}:{xmin}"
        info = {"first_lsn": lsn_str(g["first_lsn"]), "lsn": lsn_str(t.lsn), "sources": sorted(g["sources"]),
                "fate": fate, "synthetic_header": t.source != "fpi",
                "reconstructed": t.source == "update_reconstructed"}
        row = db.make_row(rel, t.data, th, {
            "rid": rid, "source": "wal", "tid": [b, o], "state": state, "xmin_status": smin,
            "xmax_status": smax, "file": None, "offset": None, "length": len(t.data), "wal": info})
        if not th.xmax and fate:
            # INSERT/UPDATE 레코드로 만든 튜플은 헤더에 xmax 가 없다 — 결말 이벤트의 트랜잭션으로 채운다
            row["xmax"], row["xmax_status"] = fate["xid"], "committed"
        rows.append(row)
        raw[rid] = (t.data, {"wal_lsn": lsn_str(t.lsn), "wal_source": t.source, "block": b, "lp": o})


def _link_chains(rows: list[dict]):
    exact: dict[tuple[int, int, int], dict] = {}   # (블록, lp, xmin) → heap/wal 행 (위치를 정확히 아는 행)
    carved: dict[tuple[int, int], list[dict]] = {}  # (블록, xmin) → 빈 공간 행 (lp 모름)
    for r in rows:
        if r["tid"][1] is None:
            carved.setdefault((r["tid"][0], r["xmin"]), []).append(r)
        else:
            exact.setdefault((r["tid"][0], r["tid"][1], r["xmin"]), r)
    nxt: dict[str, tuple[dict, str]] = {}
    prv: dict[str, dict] = {}
    for a in sorted(rows, key=lambda r: SOURCE_RANK[r["source"]]):
        if not a["xmax"] or a["state"] != "updated" or a["xmax_status"] != "committed":
            continue
        fate = (a.get("wal") or {}).get("fate") or {}
        loc = fate.get("new_tid") or a["ctid"]
        if not loc or loc[0] is None or loc[1] is None:
            continue
        b = exact.get((loc[0], loc[1], a["xmax"]))
        how = "exact"
        if b is None:
            cands = [c for c in carved.get((loc[0], a["xmax"]), []) if c["rid"] not in prv]
            if len(cands) != 1:
                continue  # 없거나 모호하다 — 잇지 않는다
            b, how = cands[0], "inferred"
        if b is a or b["rid"] in prv:
            continue
        nxt[a["rid"]] = (b, how)
        prv[b["rid"]] = a
    for r in rows:
        if r["rid"] in prv or r["rid"] not in nxt:
            continue
        chain, links, seen, cur = [r], [None], {r["rid"]}, r
        while cur["rid"] in nxt and nxt[cur["rid"]][0]["rid"] not in seen:
            cur, how = nxt[cur["rid"]]
            seen.add(cur["rid"])
            chain.append(cur)
            links.append(how)
        for i, m in enumerate(chain):
            m["chain"] = {"id": r["rid"], "pos": i + 1, "len": len(chain),
                          "prev": chain[i - 1]["rid"] if i else None,
                          "next": chain[i + 1]["rid"] if i + 1 < len(chain) else None,
                          "link": links[i]}  # 앞 버전과의 연결: exact(위치·xmin 일치) / inferred(빈 공간 후보 하나)


# ------------------------------------------------------------ 인덱스 흔적

def _comparable(v) -> bool:
    return not (isinstance(v, float) and math.isnan(v))


def index_traces(db, rel) -> dict:
    """인덱스 엔트리 중 힙과 맞지 않는 것.
    orphan       가리키는 라인 포인터가 비었거나(UNUSED) 페이지 밖 — 행이 지워지고 정리됐다
    dead         라인 포인터가 DEAD (행이 지워지고 정리 직전)
    stale        힙의 행과 키 값이 다르다 (슬롯이 다른 행에 재사용됨)
    undecodable  키나 힙 값을 해석하지 못해 비교할 수 없다
    REDIRECT(HOT 정리 후 남는 연결 포인터)는 대상 튜플을 따라가 비교한다 — 살아 있는 행이면 정상."""
    from .cluster import BTREE_AM, MISSING, deform_core, iter_pages

    cl = db.cluster
    rows, _raw, extra = db._built(rel.oid)
    lp_info = extra["lp"]
    heap = {tuple(r["tid"]): r for r in rows if r["source"] == "heap"}
    recovered: dict[tuple, list[dict]] = {}  # 한 슬롯은 시간에 따라 여러 행이 쓴다 — 키 값으로 골라야 한다
    for r in rows:
        if r["source"] != "heap" and r["tid"][1] is not None:
            recovered.setdefault(tuple(r["tid"]), []).append(r)
    names = [a.name for a in rel.attrs]

    def compare(row, ix, vals) -> str:
        for i in range(min(ix["nkey"], len(ix["indkey"]), len(vals))):
            att = ix["indkey"][i]
            if not att or att <= 0:
                continue  # 식 인덱스
            hv = row["values"][att - 1]
            if names[att - 1] in row["errors"] or not _comparable(hv) or not _comparable(vals[i]):
                return "undecodable"
            if hv != vals[i]:
                return "stale"
        return "ok"

    def resolve(tid):
        """tid → (힙 행 또는 None, 라인 포인터 상태)."""
        h = heap.get(tid)
        if h is not None:
            return h, "normal"
        flags, off = lp_info.get(tid, (None, 0))
        if flags == LP_REDIRECT:
            return heap.get((tid[0], off)), "redirect"
        if flags == LP_DEAD:
            return None, "dead"
        return None, "unused"

    out = {"indexes": [], "entries": []}
    for ix in db.indexes:
        if ix["table"] != rel.oid:
            continue
        irel = db.relations.get(ix["index"])
        if irel is None:
            continue
        info = {"oid": irel.oid, "name": irel.name, "columns": [a.name for a in irel.attrs],
                "primary": ix["primary"], "unique": ix["unique"], "total": 0, "orphan": 0, "dead": 0,
                "stale": 0, "undecodable": 0, "killed": 0, "supported": irel.am == BTREE_AM and bool(irel.path)}
        out["indexes"].append(info)
        if not info["supported"]:
            continue
        attrs = irel.attrs
        for block, page, _seg_path, _foff in iter_pages(cl, irel.path):
            if block == 0:
                continue  # 메타 페이지
            for lp, killed, key, data_off, hasnulls, tids in leaf_items(page, cl.blcksz):
                try:
                    bitmap = key[8:12] if hasnulls else None
                    datums = deform_core(key, data_off, bitmap, len(attrs), attrs)
                    vals = [None if d is None or d is MISSING else db.convert(d, a.typid)
                            for a, d in zip(attrs, datums)]
                    key_err = None
                except Exception as e:
                    vals, key_err = [], f"{type(e).__name__}: {e}"
                for tid in tids:
                    info["total"] += 1
                    info["killed"] += killed
                    h, lpstate = resolve(tid)
                    if key_err:
                        status = "undecodable"
                    elif h is None:
                        status = "dead" if lpstate == "dead" else "orphan"
                    else:
                        status = compare(h, ix, vals)
                    if status == "ok":
                        continue
                    info[status] += 1
                    rec = None
                    if not key_err:
                        rec = next((r for r in recovered.get(tid, []) if compare(r, ix, vals) == "ok"), None)
                    out["entries"].append({
                        "index": irel.name, "status": status, "killed": bool(killed), "heap_tid": list(tid),
                        "lp_state": lpstate, "keys": dict(zip(info["columns"], vals)) if not key_err else {"오류": key_err},
                        "index_block": block, "index_lp": lp,
                        "heap_now": {"rid": h["rid"], "state": h["state"]} if h else None,
                        "recovered": {"rid": rec["rid"], "source": rec["source"], "state": rec["state"]} if rec else None})
    return out


# ------------------------------------------------------------ 값 → 행 링크

def key_index(db) -> dict:
    """단일 컬럼 기본키 값(8자 이상 문자열) → [{oid, schema, name, column, rid, state}].
    rows() 캐시를 밀어내지 않도록 _built 를 거치지 않고 직접 만든 결과는 버린다."""
    idx: dict[str, list[dict]] = {}
    errors = []
    for ix in db.indexes:
        if not ix["primary"] or ix["nkey"] != 1 or not ix["indkey"] or ix["indkey"][0] <= 0:
            continue
        rel = db.relations.get(ix["table"])
        if rel is None or not rel.has_heap or not rel.path or rel.nsp in ("pg_catalog", "pg_toast"):
            continue
        if not db.cluster.source.exists(rel.path):
            continue
        col = ix["indkey"][0] - 1
        try:
            rows = db.rows(rel.oid)
        except Exception as e:
            errors.append(f"{rel.nsp}.{rel.name}: {type(e).__name__}: {e}")
            continue
        cname = rel.attrs[col].name if col < len(rel.attrs) else str(col)
        for r in rows:
            v = r["values"][col] if col < len(r["values"]) else None
            if isinstance(v, str) and len(v) >= 8:
                idx.setdefault(v, []).append({"oid": rel.oid, "schema": rel.nsp, "name": rel.name,
                                              "column": cname, "rid": r["rid"], "state": r["state"],
                                              "source": r["source"]})
    for lst in idx.values():  # 현재 행(live, 파일)을 앞에
        lst.sort(key=lambda t: (t["state"] != "live", SOURCE_RANK[t["source"]]))
    if errors:
        db.diagnostics.append({"check": "키 링크 색인", "ok": False, "detail": "; ".join(errors[:5])})
    return idx
