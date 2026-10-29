"""Requeue run 25 only after its sole worker has been stopped."""
import argparse
from datetime import datetime
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-stopped', action='store_true', required=True)
    args = parser.parse_args()
    del args
    path = Path.cwd() / 'paperlens.db'
    if not path.is_file():
        raise SystemExit('Run this script from D:\\paperlens\\backend.')
    db = sqlite3.connect(path, timeout=10)
    try:
        db.execute('BEGIN IMMEDIATE')
        run = db.execute(
            'SELECT status, job_type, document_id, version_id, total_chunks, processed_chunks FROM review_runs WHERE id=25'
        ).fetchone()
        if run is None or run[0] not in ('FAILED', 'RUNNING') or run[1:4] != ('DOCUMENT_REVIEW', 5, 2):
            raise RuntimeError('Run 25 is not the expected failed/interrupted document review; no change made.')
        total, processed = run[4:]
        count, ready = db.execute(
            """SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='COMPLETED'
               AND result_json IS NOT NULL THEN 1 ELSE 0 END),0)
               FROM review_run_chunks WHERE review_run_id=25"""
        ).fetchone()
        if not total or not (total == processed == count == ready):
            raise RuntimeError('Chunk results are incomplete; no change made.')
        backup_path = path.with_name('paperlens_before_simple25_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.db')
        with sqlite3.connect(path) as source:
            with sqlite3.connect(backup_path) as target:
                source.backup(target)
        db.execute(
            """UPDATE review_runs SET status='QUEUED', stage='QUEUED',
               started_at=NULL, finished_at=NULL, error_message=NULL,
               result_json=NULL, timeout_seconds=3600 WHERE id=25"""
        )
        db.commit()
        print('Run 25 queued; all chunk results preserved; job timeout set to 3600 seconds.')
        print('Backup:', backup_path)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == '__main__':
    main()
