"""Real temporary SQLite/API/worker tests with fake model responses, no Ollama."""

import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import pymupdf as fitz
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.adapters.ollama_adapter import OllamaAdapter
from app.controllers.restored_document_controller import router as old_router
from app.controllers.vision_review_controller import router
from app.core.config import Settings
from app.core.database import Base, get_db
from app.models.document import Document, DocumentChunk, DocumentVersion, ReviewRun
from app.models.vision_review import VisionReviewPage, VisionReviewRun
from app.repositories import review_run_repository as runs
from app.review_worker import ReviewWorker
from app.services import vision_contract as contract
from app.services import vision_pdf_service, vision_review_service


FINDING = {
    "severity": "minor", "category": "specific_wording_issue", "problem": "Test issue",
    "evidence": [{"observation": "Test observation", "quote": "Alpha", "location": "top_left"}],
    "why_evidence_supports": "Test reason", "suggestion": "Test suggestion",
}


class FakeVision:
    def __init__(self):
        self.calls = []
        self.fail_step = None
        self.verdict = "supported"
        self.foreign_id = False
        self.no_findings = False

    def chat(self, prompt, **kwargs):
        props = kwargs["response_format"]["properties"]
        stage = "review" if "page_type" in props else "check" if "decisions" in props else "summary"
        self.calls.append((stage, prompt, kwargs))
        if stage != "summary":
            assert kwargs["images"][0].startswith(b"\x89PNG")
        if self.fail_step == stage:
            self.fail_step = None
            return {"model": "test", "content": "", "done": False}
        if stage == "review":
            value = {"page_type": "explanation", "summary": "Nội dung thử nghiệm.",
                     "findings": [] if self.no_findings else [FINDING], "reading_limitations": []}
        elif stage == "check":
            code = {"supported": "direct_evidence", "unsupported": "evidence_insufficient",
                    "uncertain": "uncertain_reading"}[self.verdict]
            value = {"decisions": [{"candidate_id": 999 if self.foreign_id else 1,
                                   "reason_code": code}]}
        else:
            value = {"summary": "Đây là tóm tắt chung thử nghiệm."}
        return {"model": "test", "content": json.dumps(value), "done": True, "done_reason": "stop"}


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.storage_patch = patch.object(vision_pdf_service, "STORAGE_DIR", self.root)
        self.storage_patch.start()
        with fitz.open() as pdf:
            pdf.new_page().insert_text((50, 50), "Alpha")
            pdf.new_page().insert_text((50, 50), "Beta")
            pdf.save(self.root / "source.pdf")
        import hashlib
        digest = hashlib.sha256((self.root / "source.pdf").read_bytes()).hexdigest()
        self.engine = create_engine("sqlite:///" + str(self.root / "test.db"),
                                    connect_args={"check_same_thread": False})
        @event.listens_for(self.engine, "connect")
        def foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.factory() as db:
            with db.begin():
                db.add(Document(id=1, title="Test document"))
                db.flush()
                db.add(DocumentVersion(id=1, document_id=1, original_filename="source.pdf",
                                       storage_key="source.pdf", sha256=digest, page_count=2))
        app = FastAPI()
        app.include_router(old_router)
        app.include_router(router)
        def sessions():
            with self.factory() as db:
                yield db
        app.dependency_overrides[get_db] = sessions
        self.client = TestClient(app)
        self.adapter = FakeVision()
        self.worker = ReviewWorker(self.factory, vision_adapter_factory=lambda model: self.adapter)
        self.url = "/api/v1/documents/1/versions/1/vision-review"

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        self.storage_patch.stop()
        self.directory.cleanup()

    def enqueue(self, **extra):
        response = self.client.post(self.url, json={"instruction": "Review the page.", **extra})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["run_id"]

    def status(self, run_id):
        response = self.client.get(f"/api/v1/review-runs/{run_id}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_enqueue_uses_pages_without_text_chunks(self):
        run_id = self.enqueue(pages=[2, 1, 2])
        status = self.status(run_id)
        self.assertEqual((status["job_type"], status["total_pages"], status["total_chunks"]),
                         ("VISION_DOCUMENT_REVIEW", 2, 0))
        self.assertEqual(status["processed_pages"], 0)
        with self.factory() as db:
            self.assertEqual(list(db.scalars(select(DocumentChunk))), [])
            self.assertEqual(json.loads(db.get(VisionReviewRun, run_id).selected_pages_json), [1, 2])

    def test_selected_page_numbers_are_validated(self):
        for pages in ([0], [3], [True], [], [-1]):
            response = self.client.post(self.url, json={"instruction": "Review", "pages": pages})
            self.assertEqual(response.status_code, 422, response.text)

    def test_wrong_parent_and_unknown_run(self):
        response = self.client.post(self.url.replace("documents/1", "documents/2"), json={"instruction": "Review"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get("/api/v1/review-runs/999/pages").status_code, 404)

    def test_worker_persists_both_stages_and_summary(self):
        run_id = self.enqueue()
        self.assertTrue(self.worker.process_one())
        status = self.status(run_id)
        self.assertEqual((status["status"], status["processed_pages"], status["checked_pages"]), ("COMPLETED", 2, 2))
        self.assertEqual(status["result"]["source_pages"], [1, 2])
        self.assertEqual(len(status["result"]["findings"]), 2)
        self.assertFalse(status["result"]["checks"]["semantic_correctness_verified"])
        self.assertFalse(status["result"]["findings"][0]["evidence_visually_verified"])
        self.assertEqual([c[0] for c in self.adapter.calls], ["review", "check", "review", "check", "summary"])

    def test_zero_findings_skips_check_call(self):
        self.adapter.no_findings = True
        run_id = self.enqueue(pages=[1])
        self.worker.process_one()
        status = self.status(run_id)
        self.assertEqual(status["status"], "COMPLETED")
        self.assertEqual(status["result"]["findings"], [])
        self.assertEqual([c[0] for c in self.adapter.calls], ["review", "summary"])
        self.assertTrue(status["result"]["partial_document"])

    def test_rejected_and_uncertain_findings_are_not_rewritten(self):
        for verdict, bucket in (("unsupported", "rejected_findings"), ("uncertain", "uncertain_findings")):
            self.adapter.verdict = verdict
            run_id = self.enqueue(pages=[1])
            self.worker.process_one()
            result = self.status(run_id)["result"]
            self.assertEqual(result["findings"], [])
            self.assertEqual(result[bucket][0]["evidence"], FINDING["evidence"])

    def test_resume_keeps_saved_review_after_check_failure(self):
        self.adapter.fail_step = "check"
        run_id = self.enqueue(pages=[1])
        self.worker.process_one()
        status = self.status(run_id)
        self.assertEqual((status["status"], status["reviewed_pages"], status["processed_pages"]), ("FAILED", 1, 0))
        self.assertIn("incomplete", status["error"])
        saved = self.client.get(f"/api/v1/review-runs/{run_id}/pages").json()[0]
        self.assertIsNotNone(saved["review"])
        self.assertIsNone(saved["check"])
        response = self.client.post(f"/api/v1/review-runs/{run_id}/resume")
        self.assertEqual(response.status_code, 202, response.text)
        self.worker.process_one()
        self.assertEqual(self.status(run_id)["status"], "COMPLETED")
        self.assertEqual([c[0] for c in self.adapter.calls], ["review", "check", "check", "summary"])

    def test_resume_after_summary_failure_reuses_all_page_steps(self):
        self.adapter.fail_step = "summary"
        run_id = self.enqueue(pages=[1])
        self.worker.process_one()
        self.assertEqual(self.status(run_id)["processed_pages"], 1)
        self.client.post(f"/api/v1/review-runs/{run_id}/resume")
        self.worker.process_one()
        self.assertEqual([c[0] for c in self.adapter.calls], ["review", "check", "summary", "summary"])
        self.assertEqual(self.status(run_id)["status"], "COMPLETED")

    def test_resume_rejects_running_or_completed(self):
        run_id = self.enqueue(pages=[1])
        self.assertEqual(self.client.post(f"/api/v1/review-runs/{run_id}/resume").status_code, 409)
        self.worker.process_one()
        self.assertEqual(self.client.post(f"/api/v1/review-runs/{run_id}/resume").status_code, 409)

    def test_foreign_check_ids_fail_without_accepting_findings(self):
        self.adapter.foreign_id = True
        run_id = self.enqueue(pages=[1])
        self.worker.process_one()
        self.assertEqual(self.status(run_id)["status"], "FAILED")
        with self.factory() as db:
            self.assertIsNone(db.scalar(select(VisionReviewPage)).check_json)

    def test_worker_restart_preserves_checkpoints_for_explicit_resume(self):
        run_id = self.enqueue(pages=[1])
        with self.factory() as db:
            with db.begin():
                runs.claim_next_queued(db)
        self.assertEqual(self.worker.recover_interrupted_runs(), 1)
        self.assertEqual(self.status(run_id)["status"], "FAILED")
        self.assertEqual(self.client.post(f"/api/v1/review-runs/{run_id}/resume").status_code, 202)

    def test_source_change_rejected_before_model_calls(self):
        run_id = self.enqueue()
        (self.root / "source.pdf").write_bytes(b"changed PDF")
        self.worker.process_one()
        self.assertEqual(self.status(run_id)["status"], "FAILED")
        self.assertEqual(self.adapter.calls, [])

    def test_upload_vision_pdf_does_not_invoke_ocr(self):
        data = (self.root / "source.pdf").read_bytes()
        with patch("app.pdf_processing._ocr_page", side_effect=AssertionError("OCR must not run")):
            response = self.client.post("/api/v1/documents/1/vision-versions", files={"upload": ("scan.pdf", data, "application/pdf")})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["chunk_count"], 0)

    def test_vision_upload_rejects_non_pdf_and_empty_pdf(self):
        for name, data in (("bad.txt", b"bad"), ("empty.pdf", b""), ("bad.pdf", b"not pdf")):
            response = self.client.post("/api/v1/documents/1/vision-versions", files={"upload": (name, data, "application/pdf")})
            self.assertEqual(response.status_code, 422, response.text)

    def test_deadline_prevents_new_model_call(self):
        import time
        with self.assertRaises(TimeoutError):
            vision_review_service._call(self.adapter, "test", "test", {}, lambda x:x, time.monotonic()-1)
        self.assertEqual(self.adapter.calls, [])

    def test_summary_without_compression_progress_stops(self):
        run_id = self.enqueue()
        with self.factory() as db:
            with db.begin():
                runs.claim_next_queued(db)
        with patch.object(vision_review_service, "_fits", side_effect=lambda nodes: len(nodes)==1):
            import time
            with self.assertRaisesRegex(ValueError, "cannot reduce"):
                vision_review_service._summarize(self.factory, run_id, [(1, {"summary":"A"}), (2, {"summary":"B"})], self.adapter, time.monotonic()+60)
        self.assertEqual(self.adapter.calls, [])


class AdapterTests(unittest.TestCase):
    def test_image_bytes_context_timeout_and_metadata(self):
        captured = []
        def provider(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"model":"vision", "message":{"content":"{}"}, "done":False})
        adapter = OllamaAdapter(Settings(ollama_model="vision"), transport=httpx.MockTransport(provider))
        result = adapter.chat("Review", images=[b"png-data"], num_ctx=8192, request_timeout=30)
        self.assertEqual(captured[0]["messages"][0]["images"], [base64.b64encode(b"png-data").decode()])
        self.assertEqual(captured[0]["options"]["num_ctx"], 8192)
        self.assertFalse(result["done"])

    def test_text_default_context_stays_4096(self):
        captured = []
        def provider(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"model":"text", "message":{"content":"ok"}, "done":True})
        adapter = OllamaAdapter(Settings(), transport=httpx.MockTransport(provider))
        adapter.chat("Test")
        self.assertEqual(captured[0]["options"]["num_ctx"], 4096)
        self.assertNotIn("images", captured[0]["messages"][0])


class GenerationTests(unittest.TestCase):
    @staticmethod
    def adapter(*responses):
        class Queue:
            def __init__(self):
                self.calls = []
                self.responses = list(responses)

            def chat(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return self.responses.pop(0)
        return Queue()

    @staticmethod
    def response(value):
        return {"done": True, "done_reason": "stop", "content": json.dumps(value)}

    @staticmethod
    def review():
        return {"page_type": "explanation", "summary": "Nội dung thử nghiệm.",
                "findings": [FINDING], "reading_limitations": []}

    def generate(self, adapter, schema, validator, prompt="Review"):
        import time
        from app.services import vision_generation as generation
        return generation.generate(adapter, prompt, "system", schema, validator,
                                   time.monotonic()+30, image=b"png")

    def test_cutoff_retries_once_without_raising_token_limit(self):
        adapter = self.adapter({"done": True, "done_reason": "length", "content": ""}, self.response(self.review()))
        self.generate(adapter, contract.REVIEW_SCHEMA, contract.validate_review)
        self.assertEqual(len(adapter.calls), 2)
        self.assertEqual([c[1]["num_predict"] for c in adapter.calls], [2048, 2048])
        self.assertEqual(adapter.calls[1][1]["response_format"]["properties"]["findings"]["maxItems"], 1)

    def test_two_cutoffs_stop_without_a_third_call(self):
        bad = {"done": True, "done_reason": "length", "content": "unfinished", "eval_count": 2048}
        adapter = self.adapter(bad, bad)
        with self.assertRaisesRegex(ValueError, "after 2 attempts"):
            self.generate(adapter, contract.REVIEW_SCHEMA, contract.validate_review)
        self.assertEqual(len(adapter.calls), 2)

    def test_incomplete_response_is_not_auto_retried(self):
        adapter = self.adapter({"done": False, "content": ""})
        with self.assertRaisesRegex(ValueError, "not established"):
            self.generate(adapter, contract.REVIEW_SCHEMA, contract.validate_review)
        self.assertEqual(len(adapter.calls), 1)

    def test_repeated_prose_is_retried(self):
        repeated = self.review()
        repeated["summary"] = "This is a deliberately repetitive sentence with at least twelve separate words inside it. "*4
        adapter = self.adapter(self.response(repeated), self.response(self.review()))
        self.generate(adapter, contract.REVIEW_SCHEMA, contract.validate_review)
        self.assertEqual(len(adapter.calls), 2)

    def test_checker_uses_codes_and_normalizes_legacy_storage_shape(self):
        adapter = self.adapter(self.response({"decisions": [{"candidate_id": 1, "reason_code": "evidence_insufficient"}]}))
        prompt = json.dumps({"candidates": [dict(FINDING, candidate_id=1)]})
        result = self.generate(adapter, contract.CHECK_SCHEMA, lambda v:contract.validate_check(v,1), prompt)
        self.assertEqual(result["decisions"][0]["verdict"], "unsupported")
        self.assertEqual(set(result["decisions"][0]), {"candidate_id", "verdict", "reason"})
        props = adapter.calls[0][1]["response_format"]["properties"]["decisions"]["items"]["properties"]
        self.assertNotIn("reason", props)
        self.assertIn("reason_code", props)

    def test_checker_cannot_return_an_unknown_candidate_id(self):
        adapter = self.adapter(self.response({"decisions": [{"candidate_id": 999, "reason_code": "direct_evidence"}]}))
        with self.assertRaisesRegex(ValueError, "candidate IDs"):
            self.generate(adapter, contract.CHECK_SCHEMA, lambda v:contract.validate_check(v,1), json.dumps({"candidates": [dict(FINDING, candidate_id=1)]}))

    def test_checker_freeform_reason_is_rejected(self):
        adapter = self.adapter(self.response({"decisions": [{"candidate_id": 1, "verdict": "unsupported", "reason": "copied instructions"}]}))
        with self.assertRaisesRegex(ValueError, "Unexpected JSON fields"):
            self.generate(adapter, contract.CHECK_SCHEMA, lambda v:contract.validate_check(v,1), json.dumps({"candidates": [dict(FINDING, candidate_id=1)]}))

    def test_expired_deadline_prevents_retry(self):
        from app.services import vision_generation as generation
        adapter = self.adapter({"done": True, "done_reason": "length", "content": ""})
        with patch.object(generation.time, "monotonic", side_effect=[100, 102]):
            with self.assertRaises(TimeoutError):
                generation.generate(adapter, "Review", "system", contract.REVIEW_SCHEMA,
                                    contract.validate_review, 101)
        self.assertEqual(len(adapter.calls), 1)

    def test_quote_repetition_is_not_filtered_as_prose(self):
        value = self.review()
        value = json.loads(json.dumps(value))
        value["findings"][0]["evidence"][0]["quote"] = "These twelve separate words form an exact source quote repeated intentionally without alteration. "*4
        adapter = self.adapter(self.response(value))
        result = self.generate(adapter, contract.REVIEW_SCHEMA, contract.validate_review)
        self.assertEqual(result["findings"][0]["evidence"][0]["quote"], value["findings"][0]["evidence"][0]["quote"])


class ContractTests(unittest.TestCase):
    def test_internal_contradiction_needs_both_sides(self):
        finding = dict(FINDING, category="internal_contradiction")
        with self.assertRaisesRegex(ValueError, "two evidence"):
            contract.validate_review({"page_type":"example", "summary":"Test", "findings":[finding], "reading_limitations":[]})


if __name__ == "__main__":
    unittest.main()
