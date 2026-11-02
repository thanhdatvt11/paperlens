"""Regression tests for the observed 240-character schema cutoff."""
import json
import unittest
from app.services import document_structure_service as s


class Scripted:
    def __init__(self,outputs):self.outputs=outputs;self.calls=[]
    def chat(self,message,**kwargs):
        self.calls.append((json.loads(message),kwargs))
        value=self.outputs[min(len(self.calls)-1,len(self.outputs)-1)]
        return {'model':'qwen3:1.7b','content':json.dumps({'summary':value}), 'done':True}


def generate(adapter,limit=240):
    return s._generate_summary_group(adapter,[s._Piece('Agents perceive, reason, act and learn.',frozenset({'A'}))],
                                    'Agent foundations',limit,{'A':'Agent foundations'},35)


class SummaryRewriteTests(unittest.TestCase):
    def test_provider_schema_has_no_character_cutoff(self):
        self.assertNotIn('maxLength',s._summary_schema(240)['properties']['summary'])

    def test_observed_long_english_draft_is_rewritten(self):
        english=('AI agents are autonomous programs that perceive and act. '*12).strip()
        vietnamese='Agent AI nhận biết môi trường, suy luận và dùng công cụ để thực hiện mục tiêu.'
        adapter=Scripted([english,vietnamese])
        result,_=generate(adapter)
        self.assertEqual(result,vietnamese)
        self.assertEqual(len(adapter.calls),2)
        repair=adapter.calls[1][0]
        self.assertTrue(repair['rewrite_previous_draft'])
        self.assertEqual(repair['items'],[english])
        self.assertIn('tiếng Việt',repair['writing_instruction'])

    def test_short_english_draft_also_requires_rewrite(self):
        adapter=Scripted(['Agents use tools.','Agent sử dụng công cụ để thực hiện nhiệm vụ.'])
        out,_=generate(adapter)
        self.assertIn('sử dụng',out)
        self.assertIn('Vietnamese',adapter.calls[1][0]['validation_feedback'])

    def test_length_is_still_enforced_by_server(self):
        adapter=Scripted(['Một nội dung dài. '*30])
        with self.assertRaisesRegex(s.StructureOutputError,'after 3 attempts'):
            generate(adapter)
        self.assertEqual(len(adapter.calls),3)

    def test_repair_cannot_legitimize_invented_number(self):
        adapter=Scripted(['Agent sử dụng 999 công cụ.'])
        with self.assertRaisesRegex(s.StructureOutputError,'introduced a number'):
            generate(adapter)
        self.assertEqual(len(adapter.calls),3)

    def test_valid_vietnamese_names_and_complete_sentence_preserved(self):
        text='Tài liệu giới thiệu LangChain, LangGraph và FastAPI trong xây dựng agent.'
        adapter=Scripted([text]);result,_=generate(adapter)
        self.assertEqual(result,text)
        self.assertEqual(len(adapter.calls),1)

    def test_each_repair_request_stays_within_context_budget(self):
        adapter=Scripted(['Agents reason and act.','Các agent suy luận và hành động.'])
        generate(adapter)
        for payload,kwargs in adapter.calls:
            self.assertTrue(s._fits(kwargs['system_prompt'],payload,kwargs['response_format']))


if __name__=='__main__':unittest.main()
