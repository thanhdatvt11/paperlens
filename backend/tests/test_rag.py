"""Behavior checks against real SQLite FTS5; API tests use temporary SQLite."""

import json
import os
import unittest

from app.services import rag_core


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            dict(chunk_id=11, page_number=2, chunk_index=0, text="Bộ nhớ lưu lịch sử hội thoại."),
            dict(chunk_id=12, page_number=3, chunk_index=1, text="CapCut thêm phụ đề tự động."),
            dict(chunk_id=13, page_number=4, chunk_index=2, text="Phụ đề trong video dùng để hiển thị lời nói."),
        ]

    def test_vietnamese_unaccented(self):
        rows = rag_core.search(self.chunks, "phu de", 2)
        self.assertEqual({row["chunk_id"] for row in rows}, {12, 13})
        self.assertEqual(rows[0]["text"], self.chunks[rows[0]["chunk_index"]]["text"])

    def test_d_stroke(self):
        self.assertEqual(rag_core.search(self.chunks, "tu dong", 1)[0]["chunk_id"], 12)

    def test_no_match(self):
        self.assertEqual(rag_core.search(self.chunks, "quantum"), [])

    def test_match_syntax_is_data(self):
        rag_core.search(self.chunks, '" OR NOT * : ( -- ; DROP TABLE passages')
        self.assertTrue(rag_core.search(self.chunks, "CapCut"))

    def test_topk_limit(self):
        with self.assertRaises(ValueError):
            rag_core.search(self.chunks, "phụ đề", 6)

    def test_context_budget(self):
        sources = rag_core.search(self.chunks, "phụ đề")
        huge = dict(sources[0], text="x" * 20000)
        selected = rag_core.fit_sources("phụ đề", [huge, sources[-1]])
        self.assertEqual(len(selected), 1)
        self.assertLess(len(rag_core.prompt_for("phụ đề", selected)), 2000)

    def test_original_quote_accepted(self):
        sources = rag_core.search(self.chunks, "CapCut")
        value = dict(status="answered", claims=[dict(text="Có thể thêm phụ đề.",
                     source_id="C12", quote="CapCut thêm phụ đề tự động.")])
        self.assertEqual(rag_core.parse_answer(json.dumps(value), sources), value)

    def test_rewritten_quote_rejected(self):
        value = dict(status="answered", claims=[dict(text="Ý chính", source_id="C12", quote="CapCut tự sinh sub.")])
        with self.assertRaises(ValueError):
            rag_core.parse_answer(json.dumps(value), rag_core.search(self.chunks, "CapCut"))

    def test_foreign_citation_rejected(self):
        value = dict(status="answered", claims=[dict(text="Ý chính", source_id="C99", quote="CapCut")])
        with self.assertRaises(ValueError):
            rag_core.parse_answer(json.dumps(value), rag_core.search(self.chunks, "CapCut"))

    def test_abstain_and_status_consistency(self):
        self.assertEqual(rag_core.parse_answer('{"status":"insufficient_evidence","claims":[]}', []),
                         dict(status="insufficient_evidence", claims=[]))
        with self.assertRaises(ValueError):
            rag_core.parse_answer('{"status":"answered","claims":[]}', [])


try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.controllers.rag_controller import router
    from app.core.database import Base, get_db
    from app.models.document import Document, DocumentVersion, DocumentChunk
    from app.services import rag_service
    API_AVAILABLE = True
except ModuleNotFoundError:
    if os.environ.get("PAPERLENS_REQUIRE_RAG_API_TESTS") == "1":
        raise
    API_AVAILABLE = False


@unittest.skipUnless(API_AVAILABLE, "API dependencies absent in this runtime")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.factory.begin() as db:
            db.add_all([Document(id=1, title="One"), Document(id=2, title="Two")])
            db.flush()
            db.add_all([DocumentVersion(id=index, document_id=1 if index != 2 else 2,
                        original_filename="test.pdf", storage_key="unused.pdf", sha256="a" * 64,
                        page_count=1) for index in (1, 2, 3)])
            db.flush()
            db.add_all([DocumentChunk(id=1, document_version_id=1, chunk_index=0,
                        page_number=1, text="CapCut thêm phụ đề tự động."),
                        DocumentChunk(id=2, document_version_id=2, chunk_index=0,
                        page_number=1, text="CapCut dữ liệu riêng phiên bản khác.")])
        def session():
            with self.factory() as db:
                yield db
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_db] = session
        self.client = TestClient(app)
        self.path = "/api/v1/documents/1/versions/1/rag/"

    def tearDown(self):
        self.client.close()
        self.engine.dispose()

    def test_scoped_search(self):
        response = self.client.post(self.path + "search", json={"question": "CapCut"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([s["chunk_id"] for s in response.json()["sources"]], [1])
        self.assertEqual(self.client.post("/api/v1/documents/1/versions/2/rag/search",
                         json={"question": "CapCut"}).status_code, 404)

    def test_vision_only_no_text(self):
        self.assertEqual(self.client.post("/api/v1/documents/1/versions/3/rag/search",
                         json={"question": "CapCut"}).status_code, 409)

    def test_validation(self):
        for payload in ({"question": " "}, {"question": "test", "top_k": True},
                        {"question": "test", "top_k": 6}):
            self.assertEqual(self.client.post(self.path + "search", json=payload).status_code, 422)

    def test_no_match_no_ollama(self):
        with self.factory() as db:
            class NoCalls:
                def chat(self, *args, **kwargs):
                    raise AssertionError("Ollama should not be called")
            result = rag_service.answer(db, 1, 1, "quantum", 3, NoCalls())
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_real_db_fake_provider_valid_and_invalid(self):
        class Provider:
            def __init__(self, quote, reason="stop"):
                self.quote, self.reason = quote, reason
            def chat(self, *args, **kwargs):
                return dict(model="fake", done=True, done_reason=self.reason,
                            content=json.dumps(dict(status="answered", claims=[
                                dict(text="CapCut hỗ trợ phụ đề.", source_id="C1", quote=self.quote)])))
        with self.factory() as db:
            result = rag_service.answer(db, 1, 1, "CapCut", 3, Provider("phụ đề tự động"))
            self.assertEqual(result["sources"][0]["page_number"], 1)
            with self.assertRaises(rag_service.RagOutputError):
                rag_service.answer(db, 1, 1, "CapCut", 3, Provider("nguồn bịa"))
            with self.assertRaises(rag_service.RagOutputError):
                rag_service.answer(db, 1, 1, "CapCut", 3, Provider("CapCut", "length"))


if __name__ == "__main__":
    unittest.main()
