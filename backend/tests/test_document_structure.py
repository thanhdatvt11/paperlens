"""Summary safety tests retained after replacing the detailed-heading contract.

Main-section behavior is tested in test_main_sections.py. Older three-level
heading tests are superseded and backed up by the installer.
"""
import json
import unittest
from unittest.mock import patch
from app.services import document_structure_service as s


class SummarySafetyTests(unittest.TestCase):
    def test_truncated_output_rejected(self):
        class Truncated:
            def chat(self,*args,**kwargs):
                return {'model':'qwen3:1.7b','content':'{}','done':True,'done_reason':'length'}
        with self.assertRaisesRegex(s.StructureOutputError,'truncated'):
            s._summarize([s._Piece('Original.',frozenset({'A'}))],'A',600,Truncated(),1,{'A':'A'})

    def test_new_number_rejected(self):
        class Number:
            def chat(self,*args,**kwargs):
                return {'model':'qwen3:1.7b','content':json.dumps({'summary':'Revenue is 99999.'}),'done':True}
        with self.assertRaisesRegex(s.StructureOutputError,'introduced a number'):
            s._summarize([s._Piece('Original.',frozenset({'A'}))],'A',600,Number(),1,{'A':'A'})

    def test_no_progress_stops(self):
        pieces=[s._Piece('tiny',frozenset({'A'})),s._Piece('tiny',frozenset({'B'}))]
        class Echo:
            def chat(self,*args,**kwargs):
                return {'model':'test','content':json.dumps({'summary':'Một câu dài hơn đầu vào.'}),'done':True}
        real=s._fits
        def fit(prompt,payload,schema):
            return len(payload.get('items',[]))<=1 and real(prompt,payload,schema)
        with patch.object(s,'_fits',fit):
            with self.assertRaisesRegex(s.StructureOutputError,'no compression progress'):
                s._summarize(pieces,'Heading',600,Echo(),1,{'A':'A','B':'B'})

    def test_large_source_requests_stay_in_context(self):
        class Record:
            def __init__(self):self.calls=[]
            def chat(self,message,**kwargs):
                payload=json.loads(message);self.calls.append((payload,kwargs))
                return {'model':'test','content':json.dumps({'summary':'Tóm tắt ngắn gọn nội dung gốc.'}),'done':True}
        adapter=Record()
        text='Original factual source material. '*1200
        result,_=s._summarize([s._Piece(text,frozenset({'A'}))],'Heading',600,adapter,1,{'A':'Heading'})
        self.assertTrue(result)
        self.assertGreater(len(adapter.calls),1)
        for payload,kwargs in adapter.calls:
            self.assertTrue(s._fits(kwargs['system_prompt'],payload,kwargs['response_format']))


if __name__=='__main__':unittest.main()
