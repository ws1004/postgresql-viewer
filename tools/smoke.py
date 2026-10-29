"""샘플로 엔진을 한 번 돌려 보는 점검 스크립트.

    py tools/smoke.py [입력 경로]   (기본: sample/postgresql.zip)
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pgviewer.engine.postgres import open_cluster  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else ROOT / "sample" / "postgresql.zip"
t0 = time.time()
c = open_cluster(src)
info = c.info()
ctl = info["control"]
print("PG", info["pg_version"], "| pgdata", info["pgdata"], "| state", ctl["state_name"],
      "| crc_ok", ctl.get("crc_ok"), "| blcksz", ctl.get("blcksz"), "| nextXid", ctl.get("ckpt_next_xid_xid"),
      "| time", ctl.get("time_utc"))
for d in c.databases():
    print("DB", d)
for d in c.databases():
    if not d["has_dir"]:
        continue
    db = c.database(d["oid"])
    for diag in db.diagnostics:
        print("   ", d["name"], diag)
    rels = db.list_relations()
    user = [r for r in rels if r["schema"] not in ("pg_catalog", "information_schema", "pg_toast")]
    print("  ", d["name"], "relations", len(rels), "user", len(user))
    if d["name"] not in ("template0", "template1", "postgres"):
        tables = [r for r in user if r["kind"] == "r"]
        print("   tables:", len(tables))
        err_total = 0
        for r in tables:
            st = db.stats(r["oid"])
            err_total += st["rows_with_errors"]
            if st["total"] or st["rows_with_errors"]:
                print(f"     {r['name']:<40} {json.dumps(st['by_state'])} errors={st['rows_with_errors']}")
        print("   rows_with_errors total:", err_total)
print(f"elapsed {time.time() - t0:.1f}s")
