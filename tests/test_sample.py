"""샘플(sample/postgresql.zip, PostgreSQL 17)로 엔진을 검증한다.

서버가 없으므로 정답 대조 대신 독립적인 불변식으로 확인한다.
    py -X utf8 -m unittest discover -s tests -v
"""

import hashlib
import json
import struct
import unittest
from pathlib import Path

from pgviewer.engine.postgres import open_cluster
from pgviewer.engine.postgres.clog import Clog
from pgviewer.engine.postgres.crc32c import crc32c
from pgviewer.engine.postgres.heap import (HEAP_XMAX_INVALID, HEAP_XMAX_IS_MULTI, HEAP_XMAX_LOCK_ONLY,
                                           TupleHeader, classify)
from pgviewer.engine.postgres.types import (FIXED, JB_FARRAY, JB_FOBJECT, JENTRY_ISCONTAINER, VARLENA,
                                            DecodeError, jsonb_from_bytes, numeric_from_bytes)
from pgviewer.engine.postgres.varlena import VarlenaError, lz4_decompress, pglz_decompress

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample" / "postgresql.zip"
SAMPLE_SHA256 = "179b8ead30fc5d79598238014949c2ec3030fc86fc951589ccc7fb97206b345b"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Unit(unittest.TestCase):
    def test_crc32c_known_vector(self):
        # RFC 3720 B.4: 32바이트 0 → 0x8A9136AA
        self.assertEqual(crc32c(b"\0" * 32), 0x8A9136AA)
        self.assertEqual(crc32c(b"123456789"), 0xE3069283)

    def test_numeric(self):
        # 긴 형식: sign/dscale=0x0002, weight=0, digits [12, 3400] → 12.34
        self.assertEqual(numeric_from_bytes(struct.pack("<Hhhh", 0x0002, 0, 12, 3400)), "12.34")
        # 짧은 형식: 0x8000 | dscale 0 | weight 0, digit 42 → 42
        self.assertEqual(numeric_from_bytes(struct.pack("<Hh", 0x8000, 42)), "42")
        # 짧은 형식 음수: sign 0x2000, dscale 1 (<<7), weight -1 (0x40 | 0x3F), digit 5000 → -0.5
        self.assertEqual(numeric_from_bytes(struct.pack("<Hh", 0x8000 | 0x2000 | (1 << 7) | 0x7F, 5000)), "-0.5")
        self.assertEqual(numeric_from_bytes(struct.pack("<H", 0xC000)), "NaN")

    def test_pglz_literal_and_backref(self):
        # 제어 바이트 0b00000010: 리터럴 'a', 역참조(len 3, off 1) → "aaaa"
        self.assertEqual(pglz_decompress(bytes([0x02, ord("a"), 0x00, 0x01]), 4), b"aaaa")

    def test_lz4_block(self):
        # 토큰 0x1F: 리터럴 1 ('a'), 매치 길이 15+4=19 … 확장 0 → 19, 오프셋 1, 마지막 리터럴 블록
        data = bytes([0x1F, ord("a"), 0x01, 0x00, 0x00, 0x10, ord("b")])
        self.assertEqual(lz4_decompress(data, 21), b"a" * 20 + b"b")

    def test_lz4_output_bound(self):
        # 매치 길이 확장 255 를 이어 붙인 폭탄 입력 → 기대 크기를 넘는 순간 멈춰야 한다
        bomb = bytes([0x1F, ord("a"), 0x01, 0x00]) + b"\xff" * 1000 + b"\x00"
        with self.assertRaises(VarlenaError):
            lz4_decompress(bomb, 100)

    def test_numeric_large_exact(self):
        # 40자리 정수 + 소수 — Decimal 컨텍스트 반올림 없이 그대로 나와야 한다
        digits = [1234] * 10 + [5678]
        p = struct.pack("<Hh", 4, 9) + struct.pack(f"<{len(digits)}h", *digits)  # dscale 4, weight 9
        self.assertEqual(numeric_from_bytes(p), "1234" * 10 + ".5678")
        self.assertEqual(numeric_from_bytes(struct.pack("<Hhh", 0x0003, -2, 5)), "0.000")  # 0.0005 를 dscale 3 으로

    def test_inet(self):
        # inet_struct: family(1) bits(1) addr — 192.168.1.5/32, 10.0.0.0/8, ::1/128
        self.assertEqual(VARLENA[869](bytes([2, 32, 192, 168, 1, 5])), "192.168.1.5")
        self.assertEqual(VARLENA[650](bytes([2, 8, 10, 0, 0, 0])), "10.0.0.0/8")
        self.assertEqual(VARLENA[869](bytes([3, 128]) + b"\0" * 15 + b"\1"), "::1")

    def test_jsonb_shared_child_rejected(self):
        # 배열 [x, x] 의 두 JEntry 가 길이 0 컨테이너로 같은 자식을 가리키는 손상 입력 → 오류 (지수 폭발 방지)
        p = struct.pack("<III", JB_FARRAY | 2, JENTRY_ISCONTAINER, JENTRY_ISCONTAINER) + struct.pack("<I", JB_FARRAY)
        with self.assertRaises(DecodeError):
            jsonb_from_bytes(p)

    def test_jsonb_simple(self):
        # {"a": true}
        p = struct.pack("<IIII", JB_FOBJECT | 1, 1, 0x30000000, 0) + b"a"
        self.assertEqual(jsonb_from_bytes(p[:12] + b"a"), {"a": True})

    def test_interval_negative(self):
        self.assertEqual(FIXED[1186](struct.pack("<qii", 0, 0, -13)), "-1 years -1 mons")

    def test_classify(self):
        clog = _FakeClog({100: "committed", 101: "committed", 102: "aborted", 103: "in_progress"})

        def h(xmin, xmax, infomask=0, ctid=(0, 1)):
            return TupleHeader(xmin, xmax, 0, ctid[0], ctid[1], 0, infomask, 24)

        self.assertEqual(classify(h(100, 0, HEAP_XMAX_INVALID), clog, 0, 1)[0], "live")
        self.assertEqual(classify(h(100, 101), clog, 0, 1)[0], "deleted")
        self.assertEqual(classify(h(100, 101, ctid=(0, 2)), clog, 0, 1)[0], "updated")
        self.assertEqual(classify(h(102, 0), clog, 0, 1)[0], "aborted")
        self.assertEqual(classify(h(100, 102), clog, 0, 1)[0], "live")       # DELETE 가 롤백됨
        self.assertEqual(classify(h(100, 103), clog, 0, 1)[0], "live")       # DELETE 진행 중
        self.assertEqual(classify(h(100, 101, HEAP_XMAX_LOCK_ONLY), clog, 0, 1)[0], "live")  # 잠금만
        self.assertEqual(classify(h(100, 5, HEAP_XMAX_IS_MULTI), clog, 0, 1)[0], "unknown")
        self.assertEqual(classify(h(103, 0), clog, 0, 1)[0], "in_progress")

    def test_clog_uncommitted_after_clean_shutdown(self):
        src = _FakeSource({"pg_xact/0000": bytes(8192)})  # 모두 0 (기록 없음)
        self.assertEqual(Clog(src, 8192, next_xid=500, clean_shutdown=True).status(400), "uncommitted")
        self.assertEqual(Clog(src, 8192, next_xid=500, clean_shutdown=True).status(600), "in_progress")
        self.assertEqual(Clog(src, 8192, next_xid=500, clean_shutdown=False).status(400), "in_progress")

    def test_chain_ambiguous_carved_not_linked(self):
        from pgviewer.engine.postgres.recovery import _link_chains

        def row(rid, tid, xmin, xmax=0, state="live", ctid=None):
            return {"rid": rid, "source": "carved" if tid[1] is None else "heap", "tid": tid, "xmin": xmin,
                    "xmax": xmax, "state": state, "xmax_status": "committed" if xmax else "none",
                    "ctid": ctid or tid}
        a = row("h:0:1", [0, 1], 100, 200, "updated", [0, 5])
        rows = [a, row("c:0:100", [0, None], 200), row("c:0:300", [0, None], 200)]
        _link_chains(rows)
        self.assertNotIn("chain", a)                      # 후보 둘 → 잇지 않는다
        rows = [a, row("c:0:100", [0, None], 200)]
        _link_chains(rows)
        self.assertEqual(rows[1]["chain"]["link"], "inferred")  # 후보 하나 → '추정' 연결
        exact = row("h:0:5", [0, 5], 200)
        rows = [dict(a), exact, row("c:0:100", [0, None], 200)]
        rows[0].pop("chain", None)
        _link_chains(rows)
        self.assertEqual(exact["chain"]["link"], "exact")     # 정확한 위치가 있으면 그쪽

    def test_json_safe(self):
        from pgviewer.api.main import json_safe
        self.assertEqual(json_safe([2 ** 60, float("nan"), 5, {"x": float("inf")}]),
                         [str(2 ** 60), "nan", 5, {"x": "inf"}])


class _FakeClog:
    def __init__(self, m):
        self.m = m

    def status(self, xid):
        return self.m.get(xid, "unknown")


class _FakeSource:
    def __init__(self, files):
        self.files = files

    def exists(self, p):
        return p in self.files

    def read(self, p, offset=0, length=-1):
        return self.files[p]


@unittest.skipUnless(SAMPLE.exists(), "샘플 없음")
class Sample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = _sha256(SAMPLE)
        cls.c = open_cluster(SAMPLE)
        cls.dbs = {d["name"]: d for d in cls.c.databases()}
        user_dbs = [d for n, d in cls.dbs.items() if n not in ("template0", "template1", "postgres")]
        cls.db = cls.c.database(user_dbs[0]["oid"])

    @classmethod
    def tearDownClass(cls):
        cls.c.close()

    def test_00_sample_unchanged(self):
        self.assertEqual(self.before, SAMPLE_SHA256)

    def test_control(self):
        ctl = self.c.info()["control"]
        self.assertTrue(ctl["crc_ok"])
        self.assertEqual(ctl["pg_control_version"], 1700)
        self.assertEqual(ctl["blcksz"], 8192)
        self.assertEqual(ctl["name_data_len"], 64)
        self.assertEqual(ctl["state_name"], "shut down")

    def test_databases(self):
        self.assertTrue({"template0", "template1", "postgres"}.issubset(set(self.dbs)))
        self.assertGreater(len(self.dbs), 3)  # 최소 사용자 DB 하나는 있어야 한다
        self.assertTrue(all(d["encoding"] == 6 for d in self.dbs.values()))  # 6 = UTF8

    def test_builtin_type_names(self):
        # pg_type.dat 의 고정 OID
        expect = {16: "bool", 19: "name", 20: "int8", 23: "int4", 25: "text", 1043: "varchar",
                  1184: "timestamptz", 3802: "jsonb", 2277: "anyarray", 1009: "_text"}
        for oid, name in expect.items():
            self.assertEqual(self.db.types[oid].name, name)

    def test_builtin_catalog_names(self):
        for oid, name in {1259: "pg_class", 1249: "pg_attribute", 1247: "pg_type", 1262: "pg_database",
                          2615: "pg_namespace"}.items():
            self.assertEqual(self.db.relations[oid].name, name)
            self.assertEqual(self.db.relations[oid].nsp, "pg_catalog")

    def test_relnatts_matches_attributes(self):
        for rel in self.db.relations.values():
            self.assertEqual(rel.natts, len(rel.attrs), f"{rel.nsp}.{rel.name}")

    def test_bootstrap_diagnostics(self):
        for d in self.db.diagnostics:
            self.assertTrue(d["ok"], d)

    def test_no_decode_errors(self):
        for rel in self.db.relations.values():
            if not rel.has_heap or not rel.path or not self.c.source.exists(rel.path):
                continue
            for r in self.db.rows(rel.oid):
                for col, msg in r["errors"].items():
                    self.assertIn("TOAST 청크가 남아 있지 않음", msg, f"{rel.name}.{col} {r['tid']}")
                    self.assertNotEqual(r["state"], "live", "live 행의 TOAST 가 사라질 수는 없다")

    def test_text_is_valid_utf8(self):
        # 'replace' 디코딩이 끼어들었다면 U+FFFD 가 보인다 → 경계 계산이 틀렸다는 신호
        rel = self.db.relation_by_name("public", "posts")
        cols = [a.name for a in rel.attrs]
        for r in self.db.rows(rel.oid):
            for name, v in zip(cols, r["values"]):
                if isinstance(v, str):
                    self.assertNotIn("�", v, f"posts.{name} {r['tid']}")

    def test_jsonb_roundtrip(self):
        rel = self.db.relation_by_name("public", "posts")
        i = [a.name for a in rel.attrs].index("props")
        for r in self.db.rows(rel.oid):
            v = r["values"][i]
            self.assertIsInstance(v, dict)
            json.dumps(v)

    def test_toasted_catalog_text(self):
        # pg_rewrite.ev_action 은 pg_node_tree 텍스트 — 압축·외부 TOAST 를 풀면 "({QUERY" 로 시작한다
        rel = self.db.relation_by_name("pg_catalog", "pg_rewrite")
        i = [a.name for a in rel.attrs].index("ev_action")
        live = [r["values"][i] for r in self.db.rows(rel.oid) if r["state"] == "live"]
        self.assertGreater(len(live), 50)
        for v in live:
            self.assertTrue(v.startswith("({QUERY"), v[:40])

    # ---- 복구 계층 (WAL · 카빙 · 수정 이력 · 인덱스 흔적)

    def test_wal_integrity(self):
        s = self.c.wal_summary()
        self.assertTrue(s["available"])
        self.assertEqual(s["crc_errors"], 0)          # 모든 레코드 CRC-32C 일치 → 레코드 경계 해석이 맞다
        self.assertEqual(s["decode_errors"], 0)
        self.assertEqual(s["image_errors"], 0)
        self.assertEqual(len(s["runs"]), 1)            # 0/1000000 ~ 0/2368000 끊김 없음
        recycled = {f["name"]: f["recycled"] for f in s["files"]}
        self.assertTrue(recycled["000000010000000000000003"])   # 이름은 3번인데 내용은 1번 세그먼트
        self.assertFalse(recycled["000000010000000000000002"])
        self.assertEqual(s["last_lsn"], self.c.info()["control"]["checkpoint_lsn"])  # 마지막 레코드 = 종료 체크포인트

    def test_wal_xid_matches_control(self):
        # 독립 대조: WAL 에서 본 가장 큰 xid + 1 = pg_control 의 nextXid (정상 종료 클러스터)
        self.assertEqual(self.c.wal().max_xid + 1, self.c.info()["control"]["ckpt_next_xid_xid"])

    def test_row_detail_for_every_recovered_row(self):
        # 결말(fate)이 없는 WAL 행 · 빈 공간 행도 상세가 열려야 한다 (리뷰 지적: fate=None 에서 실패)
        for name in ("posts", "channels", "roles", "sessions", "propertyfields"):
            rel = self.db.relation_by_name("public", name)
            for r in self.db.rows(rel.oid):
                if r["source"] != "heap":
                    d = self.db.row_detail(rel.oid, r["rid"])
                    self.assertEqual(d["rid"], r["rid"])

    def test_index_redirect_not_orphan(self):
        # HOT 정리 후 REDIRECT 를 가리키는 엔트리는 살아 있는 행 — '힙에 없음' 으로 보고하면 안 된다
        for name in ("roles", "channelmembers", "threadmemberships", "status"):
            rel = self.db.relation_by_name("public", name)
            for e in self.db.index_traces(rel.oid)["entries"]:
                self.assertNotEqual(e["lp_state"], "redirect", f"{name} {e}")

    def test_heap_rows_unchanged_by_recovery(self):
        # 복구 행이 더해져도 현재 파일의 행은 그대로
        rel = self.db.relation_by_name("public", "posts")
        heap = self.db.rows(rel.oid, {"heap"})
        self.assertEqual(len(heap), 86)
        self.assertEqual(sum(r["state"] == "live" for r in heap), 79)

    def test_post_edit_chain(self):
        rel = self.db.relation_by_name("public", "posts")
        i = [a.name for a in rel.attrs].index("message")
        rows = self.db.rows(rel.oid)
        # 같은 오타 문장을 다시 보낸 게시물(현재 파일, live)도 있다 — 편집 전 원문은 WAL 에서 복구한 쪽
        old = [r for r in rows if r["values"][i] == "Helo, this is a tset message." and r["source"] == "wal"]
        self.assertEqual(len(old), 1)
        o = old[0]
        self.assertEqual((o["source"], o["state"]), ("wal", "updated"))
        nxt = next(r for r in rows if r["rid"] == o["chain"]["next"])
        self.assertEqual(nxt["values"][i], "Hello, this is a test message.")
        self.assertEqual(nxt["xmin"], o["xmax"])       # 체인 규칙: 새 버전의 xmin = 옛 버전의 xmax
        self.assertLess(o["xmin_time"], nxt["xmin_time"])

    def test_chain_rule_everywhere(self):
        for rel in self.db.relations.values():
            if rel.nsp != "public" or not rel.has_heap or not rel.path:
                continue
            by = {r["rid"]: r for r in self.db.rows(rel.oid)}
            for r in by.values():
                ch = r.get("chain")
                if ch and ch["next"]:
                    self.assertEqual(by[ch["next"]]["xmin"], r["xmax"], f"{rel.name} {r['rid']}")

    def test_index_orphans_match_wal_rows(self):
        # 독립된 두 출처의 대조: 인덱스에 남은 '힙에 없는' 기본키 값 = 그 위치에서 WAL 로 복구한 행의 기본키 값
        checked = 0
        for name in ("posts", "channels", "sessions", "propertyfields"):
            rel = self.db.relation_by_name("public", name)
            rows = {r["rid"]: r for r in self.db.rows(rel.oid)}
            idcol = [a.name for a in rel.attrs].index("id")
            for e in self.db.index_traces(rel.oid)["entries"]:
                if e["index"] != f"{name}_pkey" or not e["recovered"]:
                    continue
                rec = rows[e["recovered"]["rid"]]
                self.assertEqual(rec["values"][idcol], e["keys"]["id"], f"{name} {e['heap_tid']}")
                checked += 1
        self.assertGreater(checked, 50)

    def test_carved_rows_plausible(self):
        nx = self.c.info()["control"]["ckpt_next_xid_xid"]
        n = 0
        for rel in self.db.relations.values():
            if rel.nsp != "public" or not rel.has_heap or not rel.path:
                continue
            for r in self.db.rows(rel.oid, {"carved"}):
                n += 1
                self.assertLess(r["xmin"], nx)
                self.assertNotIn("*", r["errors"])
                self.assertIn(r["state"], ("deleted", "updated", "aborted", "unknown"))
        self.assertGreater(n, 0)

    def test_referential_integrity(self):
        """사용자 DB 의 모든 테이블에서 라이브 행의 기본 디코딩이 성공하는지 확인한다."""
        n = 0
        for rel in self.db.relations.values():
            if rel.nsp in ("pg_catalog", "information_schema", "pg_toast"):
                continue
            if not rel.has_heap or not rel.path or not self.c.source.exists(rel.path):
                continue
            for r in self.db.rows(rel.oid, {"heap"}):
                if r["state"] == "live":
                    self.assertNotIn("*", r["errors"])
                    n += 1
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
