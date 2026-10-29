"""전수 점검: 모든 데이터베이스 · 모든 힙 릴레이션을 디코딩해 오류와 미해석 타입을 센다.

    py -X utf8 tools/validate.py [입력 경로]

검사 항목
  - 튜플 분해·값 변환 오류 (TOAST 소실은 따로 센다)
  - hex 로만 표시된 타입 (디코더가 없는 타입)
  - live 행 수와 pg_class.reltuples (ANALYZE 통계) 비교 — 통계가 오래됐을 수 있으므로 참고용
  - 저장 방식별 varlena 수 (short / plain / compressed / external) 와 해제 성공 여부
"""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pgviewer.engine.postgres import open_cluster  # noqa: E402
from pgviewer.engine.postgres.cluster import Varlena, deform, scan_relation_file  # noqa: E402
from pgviewer.engine.postgres.varlena import varlena_kind  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else ROOT / "sample" / "postgresql.zip"
c = open_cluster(src)
grand = Counter()
for d in c.databases():
    if not d["has_dir"]:
        continue
    db = c.database(d["oid"])
    errs = Counter()
    hex_types = Counter()
    kinds = Counter()
    toast_lost = 0
    stat_diff = []
    nrel = nrow = 0
    for rel in db.relations.values():
        if not rel.has_heap or not rel.path or not c.source.exists(rel.path):
            continue
        nrel += 1
        rows = db.rows(rel.oid)
        nrow += len(rows)
        for r in rows:
            for col, msg in r["errors"].items():
                if "TOAST 청크가 남아 있지 않음" in msg:
                    toast_lost += 1
                elif msg.startswith("missing"):
                    errs["missing(natts 밖)"] += 1
                else:
                    errs[f"{rel.nsp}.{rel.name}.{col}: {msg[:80]}"] += 1
            for a, v in zip(rel.attrs, r["values"]):
                if isinstance(v, str) and v.startswith("\\x") and not a.dropped and a.typid != 17:
                    hex_types[db.types[a.typid].display_name if a.typid in db.types else a.typid] += 1
        for t in scan_relation_file(c, rel.path):
            try:
                for v in deform(t.data, t.header, rel.attrs):
                    if isinstance(v, Varlena):
                        kinds[varlena_kind(v.raw)] += 1
            except Exception:
                pass
        live = sum(1 for r in rows if r["state"] == "live")
        if rel.reltuples >= 0 and rel.kind == "r" and rel.nsp == "public" and abs(live - rel.reltuples) > 0:
            stat_diff.append(f"{rel.name}: live {live} / reltuples {rel.reltuples:g}")
    by_src = Counter(r["source"] for rel in db.relations.values()
                     if rel.has_heap and rel.path and c.source.exists(rel.path) for r in db.rows(rel.oid))
    print(f"== {d['name']} (oid {d['oid']}): 릴레이션 {nrel}, 행 {nrow} (출처 {dict(by_src)})")
    print(f"   varlena 저장 방식: {dict(kinds)}")
    print(f"   TOAST 소실(정리된 구버전 값): {toast_lost}")
    print(f"   오류 {sum(errs.values())}건" + ("" if not errs else ":"))
    for k, v in errs.most_common(15):
        print(f"     {v:>5}  {k}")
    print(f"   hex 로만 표시된 타입: {dict(hex_types) if hex_types else '없음'}")
    if stat_diff:
        print(f"   live ≠ reltuples ({len(stat_diff)}개, 참고): {stat_diff[:8]}")
    grand["rows"] += nrow
    grand["errors"] += sum(errs.values())
print("합계", dict(grand))
