"""Mechanical summary acceptance/retry checks. No semantic quality claim."""
import json
import unittest
from app.services import document_structure_service as s


class Scripted:
    def __init__(self,outputs):self.outputs=outputs;self.calls=[]
    def chat(self,message,**kwargs):
        self.calls.append((json.loads(message),kwargs))
        value=self.outputs[min(len(self.calls)-1,len(self.outputs)-1)]
        return {'model':'qwen3:1.7b','content':json.dumps({'summary':value},ensure_ascii=False),'done':True}


def generate(adapter,source='Agent frameworks and memory systems.',limit=600):
    return s._generate_summary_group(adapter,[s._Piece(source,frozenset({'A'}))],
                                     'Main section',limit,{'A':'Main section'},34)


class SummaryQualityTests(unittest.TestCase):
    def test_run34_dangling_tail_is_rejected(self):
        self.assertIsNotNone(s._summary_quality_error('Phần 6 c',''))

    def test_complete_summary_is_kept_exactly(self):
        text='Tài liệu trình bày các framework và hệ thống nhớ.'
        adapter=Scripted([text]);out,model=generate(adapter)
        self.assertEqual(out,text);self.assertEqual(model,'qwen3:1.7b')
        self.assertEqual(len(adapter.calls),1)

    def test_unfinished_summary_is_regenerated_once(self):
        adapter=Scripted(['Phần 6 c','Tài liệu trình bày bảo mật cho agent.'])
        result,_=generate(adapter,source='6. Security and compliance.')
        self.assertTrue(result.endswith('.'))
        self.assertEqual(len(adapter.calls),2)
        self.assertIn('unfinished',adapter.calls[1][0]['validation_feedback'])

    def test_repeated_bad_output_fails_after_three_calls(self):
        adapter=Scripted(['Đang viết dở'])
        with self.assertRaisesRegex(s.StructureOutputError,'after 3 attempts'):
            generate(adapter)
        self.assertEqual(len(adapter.calls),3)

    def test_new_cjk_characters_rejected_then_retried(self):
        adapter=Scripted(['Xử lý yêu cầu异步.','Hệ thống xử lý yêu cầu bất đồng bộ.'])
        result,_=generate(adapter)
        self.assertNotIn('异',result)
        self.assertEqual(len(adapter.calls),2)

    def test_original_cjk_term_is_not_globally_banned(self):
        text='Thuật ngữ gốc là 异步.'
        self.assertIsNone(s._summary_quality_error(text,'Original term: 异步'))

    def test_ellipsis_not_accepted_as_completion(self):
        self.assertIsNotNone(s._summary_quality_error('Các framework gồm...',''))
        self.assertIsNotNone(s._summary_quality_error('Các framework gồm…',''))

    def test_replacement_character_rejected(self):
        self.assertIsNotNone(s._summary_quality_error('Văn bản lỗi �.','Source'))

    def test_framework_names_and_closing_quotes_allowed(self):
        self.assertIsNone(s._summary_quality_error('Tài liệu giới thiệu LangChain và FastAPI.',''))
        self.assertIsNone(s._summary_quality_error('Nội dung nói về “agent.”',''))

    def test_target_below_hard_limit_and_retry_shorter(self):
        adapter=Scripted(['Không hoàn thành','Tài liệu trình bày nền tảng.'])
        generate(adapter)
        first,second=[x[0] for x in adapter.calls]
        self.assertLess(first['target_summary_characters'],first['max_summary_characters'])
        self.assertLess(second['target_summary_characters'],first['target_summary_characters'])
        self.assertEqual(len(first['validation_feedback']),len(second['validation_feedback']))
        for payload,kwargs in adapter.calls:
            self.assertTrue(s._fits(kwargs['system_prompt'],payload,kwargs['response_format']))

    def test_new_numbers_still_rejected_with_finite_retries(self):
        adapter=Scripted(['Thị trường đạt 99999 tỷ USD.'])
        with self.assertRaisesRegex(s.StructureOutputError,'introduced a number'):
            generate(adapter)
        self.assertEqual(len(adapter.calls),3)

    def test_hard_limit_failure_is_not_silently_sliced(self):
        adapter=Scripted(['Một câu quá dài. '*60,'Tài liệu trình bày nền tảng.'])
        result,_=generate(adapter)
        self.assertEqual(result,'Tài liệu trình bày nền tảng.')
        self.assertEqual(len(adapter.calls),2)


if __name__=='__main__':unittest.main()
