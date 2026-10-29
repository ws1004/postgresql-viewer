"""릴레이션 하나의 컬럼과 행을 출력한다 (값은 앞부분만).

    py -X utf8 tools/inspect_rel.py <db이름> <schema.table> [최대행수] [--errors]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pgviewer.engine.postgres import open_cluster  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
only_err = "--errors" in sys.argv
dbname, qual = args[0], args[1]
limit = int(args[2]) if len(args) > 2 else 5
c = open_cluster(ROOT / "sample" / "postgresql.zip")
dbrow = next(d for d in c.databases() if d["name"] == dbname)
db = c.database(dbrow["oid"])
schema, name = qual.split(".", 1)
rel = db.relation_by_name(schema, name)
cols = db.columns(rel.oid)
print("columns:", ", ".join(f"{c['name']}:{c['type']}" for c in cols))
n = 0
for r in db.rows(rel.oid):
    if only_err and not r["errors"]:
        continue
    vals = {c["name"]: (v if not isinstance(v, str) or len(v) < 120 else v[:120] + "…")
            for c, v in zip(cols, r["values"])}
    print(json.dumps({"tid": r["tid"], "state": r["state"], "errors": r["errors"], "values": vals},
                     ensure_ascii=False, default=str))
    n += 1
    if n >= limit:
        break
