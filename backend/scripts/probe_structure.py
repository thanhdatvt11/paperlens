"""Read saved snapshots and match main headings to the numbered TOC."""
import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sqlite3
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',type=Path,required=True)
    parser.add_argument('--source-run-id',type=int,required=True)
    parser.add_argument('--pages',help='Comma-separated page numbers; omit for all pages')
    args=parser.parse_args()
    backend=args.backend.resolve()
    sys.path.insert(0,str(backend))
    from app.adapters.ollama_adapter import OllamaAdapter
    from app.core.config import get_settings
    from app.services.document_structure_service import StructureChunk,detect_sections,prepare_structure_chunks
    database=backend/'paperlens.db'
    if not database.is_file():
        raise RuntimeError('Existing database not found')
    if args.pages:
        raise ValueError('Main-section matching requires all snapshots; omit --pages')
    pages=None
    if pages is not None and (not pages or min(pages)<1):
        raise ValueError('Page numbers must be positive')
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        run=db.execute('SELECT job_type,status FROM review_runs WHERE id=?',(args.source_run_id,)).fetchone()
        if run is None or run[0]!='DOCUMENT_REVIEW':
            raise RuntimeError('Source must be an existing DOCUMENT_REVIEW run')
        rows=db.execute('SELECT source_chunk_id,chunk_index,page_number,text FROM review_run_chunks WHERE review_run_id=? ORDER BY chunk_index,id',(args.source_run_id,)).fetchall()
    prepared=prepare_structure_chunks([StructureChunk(*row) for row in rows])
    chunks=[c for c in prepared if pages is None or c.page_number in pages]
    if not chunks:
        raise RuntimeError('No snapshots match requested pages')
    if pages and pages-{c.page_number for c in chunks}:
        raise RuntimeError('Some requested pages are absent from snapshots')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    sections,warnings=detect_sections(chunks,OllamaAdapter(get_settings()),args.source_run_id)
    body=[s for s in sections if s.kind=='body' and s.title_origin=='source']
    from app.services.major_sections_service import read_toc_entries
    entries,_,_=read_toc_entries(chunks)
    result={'scope':'MAIN_SECTIONS_PROBE','strategy':'numbered_toc_matching','model_calls':0,'source_run_id':args.source_run_id,
            'pages':sorted({c.page_number for c in chunks}),
            'partial_document':len(chunks)!=len(rows),'chunk_count':len(chunks),
            'section_count':len(sections),'source_body_heading_count':len(body),
            'toc_entries':[asdict(e) for e in entries],
            'semantic_correctness_verified':False,
            'sections':[asdict(s) | {'heading_evidence':[e.model_dump() for e in s.heading_evidence],
                                    'spans':[span.model_dump() for span in s.spans]} for s in sections],
            'warnings':warnings}
    output=backend/f'main_sections_probe_run_{args.source_run_id}.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Probe file:',output)
    print('Sections:',len(sections),'Source body headings:',len(body))
    print('No job created; no database writes; no model calls; no summaries generated.')
    if not body:
        print('Structure needs correction: no source-backed body headings detected.')


if __name__=='__main__':
    main()
