"""Run from backend: python -m unittest discover -s tests -p test_document_structure.py -v"""
import json
import re
import unittest
from unittest.mock import patch

from app.services import document_structure_service as service


class FakeAdapter:
    """Deterministic protocol fixture, not a model quality evaluation."""
    def __init__(self, bad=None):
        self.calls = []
        self.bad = bad

    def chat(self, message, **kwargs):
        payload = json.loads(message)
        self.calls.append((payload, kwargs))
        if self.bad == 'truncated':
            return {'model': 'qwen3:1.7b', 'content': '{}', 'done': True, 'done_reason': 'length'}
        if payload['task'] == 'detect_structure':
            lines = payload['lines']
            is_toc = any(line['text'] == 'Contents' for line in lines)
            headings = []
            if not is_toc:
                for line in lines:
                    match = re.match(r'^(#{1,3}) ', line['text'])
                    if match:
                        headings.append({'heading_id': f"H{line['line']}_{line['line']}", 'level': len(match[1])})
            if self.bad == 'outside':
                headings = [{'heading_id': 'H999_999', 'level': 1}]
            value = {'kind': 'table_of_contents' if is_toc else 'body', 'headings': headings}
        else:
            value = {'summary': 'Tài liệu trình bày các khái niệm, quy trình thực hiện và những lưu ý liên quan.'}
            if self.bad == 'number':
                value = {'summary': 'Thị trường đạt 99999 tỷ USD.'}
        return {'model': 'qwen3:1.7b', 'content': json.dumps(value, ensure_ascii=False), 'done': True, 'done_reason': 'stop'}


def chunk(index, text, page=None):
    return service.StructureChunk(index + 1, index, page or index + 1, text)


class StructureTests(unittest.TestCase):
    def test_sections_and_original_quotes(self):
        chunks = [chunk(0, '# Assembly\nIntroduction.\n## Tools\nUse appropriate tools.\n# Quality\nCheck the product.\n')]
        adapter = FakeAdapter()
        summary, model, structure = service.analyze_and_summarize(chunks, adapter, 26)
        self.assertTrue(summary)
        self.assertEqual(model, 'qwen3:1.7b')
        self.assertEqual([s.title for s in structure.sections], ['Assembly', 'Tools', 'Quality'])
        self.assertEqual(structure.sections[1].parent_id, structure.sections[0].section_id)
        self.assertEqual(structure.sections[0].heading_evidence[0].quote, '# Assembly')
        self.assertFalse(structure.semantic_correctness_verified)
        assigned = [line for s in structure.sections for span in s.spans for line in range(span.start_line, span.end_line + 1)]
        self.assertEqual(sorted(assigned), list(range(1, 7)))
        root_calls = [p for p, _ in adapter.calls if p.get('current_section') == 'Assembly']
        self.assertIn('Tools', json.dumps(root_calls, ensure_ascii=False))

    def test_no_headings_uses_marked_generated_label(self):
        _, _, result = service.analyze_and_summarize([chunk(0, 'A plain paragraph.\nAnother line.')], FakeAdapter(), 1)
        self.assertEqual(result.sections[0].title_origin, 'generated')
        self.assertEqual(result.sections[0].heading_evidence, [])

    def test_toc_not_promoted_to_explained_chapters(self):
        chunks = [chunk(0, 'Contents\n# Assembly\n# Quality'), chunk(1, '# Actual section\nActual content.')]
        adapter = FakeAdapter()
        _, _, result = service.analyze_and_summarize(chunks, adapter, 1)
        self.assertEqual(result.sections[0].kind, 'table_of_contents')
        self.assertEqual(len(result.sections), 2)
        final = [p for p, _ in adapter.calls if p.get('current_section') == 'Tổng quan các phần chính của tài liệu']
        self.assertNotIn('Assembly', json.dumps(final))

    def test_only_toc_fails_instead_of_inventing_body(self):
        with self.assertRaisesRegex(service.StructureOutputError, 'Only a table'):
            service.analyze_and_summarize([chunk(0, 'Contents\n# Assembly')], FakeAdapter(), 1)

    def test_invalid_heading_retried_once_then_rejected(self):
        adapter = FakeAdapter('outside')
        with self.assertRaisesRegex(service.StructureOutputError, 'analysis failed'):
            service.analyze_and_summarize([chunk(0, '# Assembly\nText')], adapter, 1)
        self.assertEqual(len(adapter.calls), 2)

    def test_heading_must_not_be_blank(self):
        with self.assertRaisesRegex(service.StructureOutputError, 'blank'):
            service._validate_analysis({'kind':'body','headings':[{'start_line':2,'end_line':2,'level':1}]}, chunk(0,'Text\n\nTitle'),1,3)

    def test_model_cannot_supply_invented_title(self):
        with self.assertRaises(ValueError):
            service._validate_analysis({'kind':'body','headings':[{'start_line':1,'end_line':1,'level':1,'title':'Invented'}]},chunk(0,'Title'),1,1)

    def test_overlapping_heading_ranges_rejected(self):
        with self.assertRaisesRegex(service.StructureOutputError, 'overlap'):
            service._validate_analysis({'kind':'body','headings':[{'start_line':1,'end_line':2,'level':1},{'start_line':2,'end_line':3,'level':2}]},chunk(0,'A\nB\nC'),1,3)

    def test_truncated_response_fails(self):
        with self.assertRaises(service.StructureOutputError):
            service.analyze_and_summarize([chunk(0,'# Assembly\nText')],FakeAdapter('truncated'),1)

    def test_new_number_fails(self):
        with self.assertRaisesRegex(service.StructureOutputError,'introduced a number'):
            service.analyze_and_summarize([chunk(0,'# Assembly\nText')],FakeAdapter('number'),1)

    def test_all_windows_cover_all_lines(self):
        c=chunk(0,'\n'.join('Text line' for _ in range(53)))
        ranges=service._window_ranges(c)
        self.assertEqual([i for start,end in ranges for i in range(start,end+1)],list(range(1,54)))

    def test_duplicate_chunk_id_rejected(self):
        c=chunk(0,'Text')
        with self.assertRaisesRegex(service.StructureOutputError,'Duplicate'):
            service.analyze_and_summarize([c,c],FakeAdapter(),1)

    def test_page_order_rejected(self):
        with self.assertRaisesRegex(service.StructureOutputError,'page order'):
            service.analyze_and_summarize([chunk(0,'Text',2),chunk(1,'Text',1)],FakeAdapter(),1)

    def test_empty_source_rejected(self):
        with self.assertRaises(service.StructureOutputError):
            service.analyze_and_summarize([chunk(0,' ')],FakeAdapter(),1)

    def test_overlap_heading_not_duplicated(self):
        chunks=[chunk(0,'# Assembly\nText\n## Tools\nUse tools.',1),chunk(1,'## Tools\nUse tools.\nMore content.',1)]
        _,_,result=service.analyze_and_summarize(chunks,FakeAdapter(),1)
        self.assertEqual([s.title for s in result.sections],['Assembly','Tools'])

    def test_large_outline_and_context_bounded(self):
        chunks=[chunk(i,f'# Subject {i}\n' + 'Source content. '*18) for i in range(63)]
        adapter=FakeAdapter()
        _,_,result=service.analyze_and_summarize(chunks,adapter,1)
        self.assertEqual(len(result.sections),63)
        for payload, kwargs in adapter.calls:
            self.assertTrue(service._fits(kwargs['system_prompt'],payload,kwargs['response_format']))
        self.assertLess(len(adapter.calls),220)

    def test_no_progress_stops(self):
        pieces=[service._Piece('tiny',frozenset({'A'})),service._Piece('tiny',frozenset({'B'}))]
        class Echo:
            def chat(self, text, **kwargs):
                return {'model':'test','content':json.dumps({'summary':'longer than the input'}),'done':True}
        real_fits=service._fits
        def fit(prompt,payload,schema):
            return len(payload.get('items',[]))<=1 and real_fits(prompt,payload,schema)
        with patch.object(service,'_fits',fit):
            with self.assertRaisesRegex(service.StructureOutputError,'no compression progress'):
                service._summarize(pieces,'Heading',600,Echo(),1,{'A':'A','B':'B'})


    def test_candidate_spans_are_prevalidated(self):
        c=chunk(0,'# Title\n\nShort text.\n## Tools\nMore text.\n')
        candidates=service._heading_candidates(c,1,5)
        lines=c.text.splitlines(keepends=True)
        self.assertTrue(candidates)
        for start,end in candidates.values():
            self.assertTrue(lines[start-1].strip())
            self.assertTrue(lines[end-1].strip())
            self.assertLessEqual(end-start,2)
            self.assertLessEqual(len(''.join(lines[start-1:end]).strip()),240)

    def test_model_selects_ids_not_raw_ranges(self):
        c=chunk(0,'# Title\n\nShort text.')
        schema=service._analysis_schema(c,1,3)
        fields=schema['$defs']['_CandidatePick']['properties']
        self.assertEqual(set(fields),{'heading_id','level'})
        self.assertEqual(set(fields['heading_id']['enum']),set(service._heading_candidates(c,1,3)))

    def test_overlap_and_order_normalized_with_visible_warning(self):
        c=chunk(0,'# Parent\n## Child\nText.')
        result=service._resolve_candidate_analysis({
            'kind':'body', 'headings':[
                {'heading_id':'H2_2','level':2},
                {'heading_id':'H1_2','level':1},
                {'heading_id':'H1_2','level':1},
            ]},c,1,3)
        self.assertEqual([(h.start_line,h.end_line) for h in result.headings],[(1,2)])
        self.assertTrue(result.normalization_warnings)
        window=service._Window(c,1,3,result)
        sections=service._build_sections([c],[window])
        assigned=[i for sec in sections for span in sec.spans for i in range(span.start_line,span.end_line+1)]
        self.assertEqual(assigned,[1,2,3])

    def test_later_window_keeps_opening_context_without_selecting_it(self):
        c=chunk(0,'Table of Contents\n'+'Topic entry\n'*34)
        payload=service._line_payload(c,21,35)
        self.assertIn('Table of Contents',payload['chunk_opening_context'])
        self.assertTrue(all(start>=21 and end<=35 for start,end in service._heading_candidates(c,21,35).values()))

    def test_blank_window_does_not_call_provider(self):
        adapter=FakeAdapter()
        result=service._analyze(adapter,chunk(0,'\n   \n'),1,2)
        self.assertEqual(result.headings,[])
        self.assertEqual(adapter.calls,[])


if __name__ == '__main__':
    unittest.main()
