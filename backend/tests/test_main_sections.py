"""Main-section tests use immutable text fixtures and a fake summary adapter."""
import json
import unittest
from app.services import document_structure_service as s
from app.services import major_sections_service as m


def chunk(i,text,page=None):
    return s.StructureChunk(i+1,i,page if page is not None else i+1,text)


def fixture():
    return [chunk(0,'THE COMPLETE GUIDE\nAI AGENTS'),
            chunk(1,'Table of Contents\n\n1. Foundations\n• Details\n\n2. Frameworks\n• Examples\n'),
            chunk(2,'Executive Summary\nIntro.\n\n1. Foundations\n\nText.\n\n1. Follow instructions\nassert x\n'),
            chunk(3,'More explanation.\n\n2. Frameworks\n\nTool details.\n')]


class NoModel:
    def chat(self,*a,**k):
        raise AssertionError('Structure matching must not call a model')


class SummaryAdapter:
    def __init__(self):self.calls=[]
    def chat(self,message,**kwargs):
        p=json.loads(message);self.calls.append(p)
        if p['task']!='summarize_sections':raise AssertionError('Unexpected model structure call')
        return {'model':'qwen3:1.7b','content':json.dumps({'summary':'Tài liệu trình bày nền tảng và các framework cùng ví dụ thực hành.'}), 'done':True}


class MainSectionTests(unittest.TestCase):
    def test_only_toc_backed_chapters_are_selected(self):
        ss,_=s.detect_sections(fixture(),NoModel(),33)
        titles=[x.title for x in ss if x.title_origin=='source']
        self.assertEqual(titles,['1. Foundations','2. Frameworks'])
        self.assertTrue(all(x.parent_id is None for x in ss))

    def test_source_quotes_are_original(self):
        ss,_=s.detect_sections(fixture(),NoModel(),33)
        for sec in ss:
            for e in sec.heading_evidence:
                c=next(c for c in fixture() if c.chunk_id==e.chunk_id)
                self.assertIn(e.quote,c.text)

    def test_every_line_is_assigned_once(self):
        cs=fixture();ss,_=s.detect_sections(cs,NoModel(),33)
        for c in cs:
            assigned=[i for sec in ss for span in sec.spans if span.chunk_id==c.chunk_id
                      for i in range(span.start_line,span.end_line+1)]
            self.assertEqual(sorted(assigned),list(range(1,len(c.text.splitlines())+1)))

    def test_numbered_list_does_not_create_chapter(self):
        ss,_=s.detect_sections(fixture(),NoModel(),33)
        self.assertFalse(any('Follow instructions' in x.title for x in ss))

    def test_toc_and_cover_are_separate(self):
        ss,_=s.detect_sections(fixture(),NoModel(),33)
        self.assertEqual([x.kind for x in ss[:2]],['cover','table_of_contents'])

    def test_preamble_is_retained(self):
        ss,_=s.detect_sections(fixture(),NoModel(),33)
        generated=[x for x in ss if x.kind=='body' and x.title_origin=='generated']
        self.assertEqual(len(generated),1)
        self.assertEqual(generated[0].spans[0].start_line,1)

    def test_no_explicit_toc_fails(self):
        with self.assertRaisesRegex(s.StructureOutputError,'No explicit'):
            s.detect_sections([chunk(0,'1. Heading\nText.')],NoModel(),1)

    def test_toc_without_numbered_entries_fails(self):
        with self.assertRaisesRegex(s.StructureOutputError,'Fewer than two numbered'):
            s.detect_sections([chunk(0,'Contents\nTopic\nOther')],NoModel(),1)

    def test_repeated_toc_number_fails(self):
        with self.assertRaisesRegex(s.StructureOutputError,'repeated TOC'):
            m.read_toc_entries([chunk(0,'Contents\n1. Topic\n\n1. Other')])

    def test_out_of_order_toc_fails(self):
        with self.assertRaises(s.StructureOutputError):
            m.read_toc_entries([chunk(0,'Contents\n2. Topic\n\n1. Other')])

    def test_number_alone_does_not_match(self):
        e=m.TocEntry(1,'Foundations',1,2,1,1,'1. Foundations')
        self.assertEqual(m._match_score(e,'1. Completely different'),0)

    def test_wrong_number_does_not_match(self):
        e=m.TocEntry(1,'Foundations',1,2,1,1,'1. Foundations')
        self.assertEqual(m._match_score(e,'2. Foundations'),0)

    def test_ocr_ai_and_al_compared_without_rewriting(self):
        self.assertEqual(m._key('1. Al Agents'),m._key('1. AI Agents'))

    def test_wrapped_toc_title(self):
        es,_,_=m.read_toc_entries([chunk(0,'Contents\n1. Long heading\ncontinued here\n\n2. Other topic')])
        self.assertEqual(es[0].title,'Long heading continued here')
        self.assertIn('\n',es[0].quote)

    def test_wrapped_body_title(self):
        cs=fixture();cs[3]=chunk(3,'2.\nFrameworks\nText.')
        # A standalone number has no title; it can still be combined with its
        # adjacent wrapped title, preserving the actual quote.
        ss,_=s.detect_sections(cs,NoModel(),33)
        self.assertEqual([x.heading_evidence[0].quote for x in ss if x.title_origin=='source'][-1],'2.\nFrameworks')

    def test_missing_entry_is_reported_not_invented(self):
        cs=fixture();cs[1]=chunk(1,'Contents\n1. Foundations\n\n2. Frameworks\n\n3. Missing topic')
        ss,warnings=s.detect_sections(cs,NoModel(),33)
        self.assertTrue(any('not matched' in w and 'Missing topic' in w for w in warnings))
        self.assertFalse(any(x.title=='3. Missing topic' for x in ss))

    def test_only_one_match_fails(self):
        cs=fixture();cs[3]=chunk(3,'Other text.')
        with self.assertRaisesRegex(s.StructureOutputError,'Fewer than two main'):
            s.detect_sections(cs,NoModel(),33)

    def test_non_contiguous_chunks_fail(self):
        cs=fixture();cs[-1]=chunk(7,cs[-1].text)
        with self.assertRaisesRegex(s.StructureOutputError,'contiguous'):
            s.detect_sections(cs,NoModel(),33)

    def test_duplicate_chunk_ids_fail(self):
        cs=fixture();cs.append(cs[-1])
        with self.assertRaisesRegex(s.StructureOutputError,'Duplicate'):
            s.detect_sections(cs,NoModel(),33)

    def test_summaries_still_use_adapter(self):
        adapter=SummaryAdapter()
        summary,model,result=s.analyze_and_summarize(fixture(),adapter,33)
        self.assertTrue(summary);self.assertEqual(model,'qwen3:1.7b')
        self.assertGreater(len(adapter.calls),2)
        self.assertEqual(len([x for x in result.sections if x.title_origin=='source']),2)
        self.assertFalse(result.semantic_correctness_verified)

    def test_title_dot_leaders_are_removed_from_toc_comparison(self):
        es,_,_=m.read_toc_entries([chunk(0,'Contents\n1. Foundations ..... 3\n\n2. Frameworks .... 8')])
        self.assertEqual(es[0].title,'Foundations')


if __name__=='__main__':unittest.main()
