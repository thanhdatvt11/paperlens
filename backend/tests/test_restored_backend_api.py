"""Integration tests using a separate temporary SQLite database, never paperlens.db.

Run from backend after restoring files:
python -m unittest discover -s tests -p test_restored_backend_api.py -v
No Ollama calls: these tests cover persistence, queue, source binding and API routes.
"""
import json
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.controllers.restored_document_controller import router
from app.models.document import Document, DocumentVersion, DocumentChunk, ReviewRun
from app.repositories import review_run_repository as repository


class RestoredBackendTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        @event.listens_for(self.engine, 'connect')
        def foreign_keys(connection, _):
            connection.execute('PRAGMA foreign_keys=ON')
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.factory() as db:
            with db.begin():
                db.add(Document(id=6, title='Test document', description=None))
                db.flush()
                db.add(DocumentVersion(id=3, document_id=6, original_filename='test.pdf', storage_key='test.pdf', sha256='a'*64, page_count=2))
                db.flush()
                db.add_all([
                    DocumentChunk(id=87, document_version_id=3, chunk_index=0, page_number=1, text='# Assembly\nOriginal content.'),
                    DocumentChunk(id=104, document_version_id=3, chunk_index=1, page_number=2, text='# Quality\nMore content.'),
                ])
        app = FastAPI()
        app.include_router(router)
        def test_db():
            with self.factory() as db:
                yield db
        app.dependency_overrides[get_db] = test_db
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.engine.dispose()

    def enqueue(self):
        response = self.client.post('/api/v1/documents/6/versions/3/review', json={'instruction':'Summarize and review.'})
        self.assertEqual(response.status_code,202,response.text)
        return response.json()['run_id']

    def test_document_enqueue_persists_two_snapshots(self):
        run_id=self.enqueue()
        with self.factory() as db:
            run=repository.get_run(db,run_id)
            snapshots=repository.list_run_chunks(db,run_id)
            self.assertEqual(run.job_type,'DOCUMENT_REVIEW')
            self.assertEqual(run.status,'QUEUED')
            self.assertEqual(run.timeout_seconds,7200)
            self.assertEqual(run.total_chunks,2)
            self.assertEqual(run.processed_chunks,0)
            self.assertEqual([s.source_chunk_id for s in snapshots],[87,104])

    def test_chunk_preview_enqueue(self):
        response=self.client.post('/api/v1/documents/6/versions/3/chunks/104/review-preview',json={'instruction':'Review.'})
        self.assertEqual(response.status_code,202,response.text)
        with self.factory() as db:
            run=repository.get_run(db,response.json()['run_id'])
            self.assertEqual((run.job_type,run.total_chunks,run.chunk_id),('CHUNK_PREVIEW',1,104))

    def test_parent_mismatch_rejected(self):
        response=self.client.post('/api/v1/documents/7/versions/3/review',json={'instruction':'Review.'})
        self.assertEqual(response.status_code,404)
        with self.factory() as db:
            self.assertEqual(list(db.scalars(select(ReviewRun))),[])

    def test_wrong_chunk_rejected(self):
        response=self.client.post('/api/v1/documents/6/versions/3/chunks/999/review-preview',json={'instruction':'Review.'})
        self.assertEqual(response.status_code,404)

    def test_blank_instruction_rejected(self):
        self.assertEqual(self.client.post('/api/v1/documents/6/versions/3/review',json={'instruction':' '}).status_code,422)

    def test_get_status_and_missing_run(self):
        run_id=self.enqueue()
        response=self.client.get(f'/api/v1/review-runs/{run_id}')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['status'],'QUEUED')
        self.assertEqual(response.json()['total_chunks'],2)
        self.assertEqual(self.client.get('/api/v1/review-runs/99999').status_code,404)

    def test_snapshots_survive_source_text_change(self):
        run_id=self.enqueue()
        with self.factory() as db:
            with db.begin():
                db.get(DocumentChunk,87).text='Changed later.'
        with self.factory() as db:
            self.assertEqual(repository.list_run_chunks(db,run_id)[0].text,'# Assembly\nOriginal content.')

    def test_fifo_claim(self):
        first=self.enqueue(); second=self.enqueue()
        with self.factory() as db:
            with db.begin():
                self.assertEqual(repository.claim_next_queued(db).id,first)
        with self.factory() as db:
            with db.begin():
                self.assertEqual(repository.claim_next_queued(db).id,second)

    def test_final_structure_json_is_preserved(self):
        run_id=self.enqueue()
        result={'summary':'Test summary','key_points':[],'findings':[],
                'structure':{'sections':[{'title':'Assembly'}]}}
        with self.factory() as db:
            with db.begin():
                run=repository.get_run(db,run_id)
                run.status='COMPLETED';run.stage='FINISHED'
                run.result_json=json.dumps(result)
        response=self.client.get(f'/api/v1/review-runs/{run_id}')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['result'],result)

    def test_version_and_chunk_routes(self):
        response=self.client.get('/api/v1/documents/6/versions')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()[0]['id'],3)
        response=self.client.get('/api/v1/documents/6/versions/3/chunks')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual([item['id'] for item in response.json()],[87,104])
        self.assertEqual(self.client.get('/api/v1/documents/7/versions/3/chunks').status_code,404)

    def test_empty_version_cannot_be_reviewed(self):
        with self.factory() as db:
            with db.begin():
                db.add(DocumentVersion(id=4, document_id=6, original_filename='empty.pdf', storage_key='empty.pdf', sha256='b'*64, page_count=1))
        response=self.client.post('/api/v1/documents/6/versions/4/review',json={'instruction':'Review.'})
        self.assertEqual(response.status_code,422,response.text)


if __name__ == '__main__':
    unittest.main()
