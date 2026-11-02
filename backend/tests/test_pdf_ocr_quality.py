"""Routing and failure tests; OCR accuracy is not asserted by fake text."""
import subprocess
import unittest
from unittest.mock import patch

from app import pdf_processing as pdf


class QualityTests(unittest.TestCase):
    def select(self,md,native='',images=False,ocr='Văn bản tiếng Việt đã đọc được.'):
        self.calls=0
        def read():self.calls+=1;return ocr
        return pdf._select_page_text(md,native,images,read,2)

    def test_cid_garbage_replaced_even_when_ocr_is_shorter(self):
        raw='(cid:77)(cid:40)'*200
        result,method,warn=self.select(raw,raw,ocr='Ngành thiết kế đồ họa là gì?')
        self.assertEqual(result,'Ngành thiết kế đồ họa là gì?')
        self.assertEqual(method,'ocr');self.assertTrue(warn)
        self.assertEqual(self.calls,1)

    def test_cids_do_not_inflate_useful_character_count(self):
        self.assertEqual(pdf._text_character_count('(cid:77)'*300),0)

    def test_clean_native_avoids_ocr_when_markdown_is_corrupt(self):
        text='Nội dung tiếng Việt rõ ràng và đầy đủ trong bản gốc.'
        result,method,_=self.select('(cid:77)'*10,text)
        self.assertEqual(result,text);self.assertEqual(method,'native_fallback')
        self.assertEqual(self.calls,0)

    def test_clean_markdown_retained_without_ocr(self):
        text='# Các nguyên lý thiết kế\nCân bằng và tương phản.'
        self.assertEqual(self.select(text,text),(text,'markitdown',False))
        self.assertEqual(self.calls,0)

    def test_nonempty_bad_native_encoding_triggers_ocr(self):
        text='MÏ¬éÔƎďÔÖÁďƎãÁƎ¼òƎÔí«\x01'
        _,method,_=self.select('',text)
        self.assertEqual(method,'ocr');self.assertEqual(self.calls,1)

    def test_valid_vietnamese_is_not_encoding_garbage(self):
        self.assertEqual(pdf._text_quality_issues('Đường nét, hình khối, không gian, tỷ lệ, màu sắc.'),[])

    def test_whitespace_not_treated_as_invalid_controls(self):
        self.assertEqual(pdf._text_quality_issues('Text\n\tMore\r\n'),[])

    def test_unicode_replacement_triggers_ocr(self):
        _,method,_=self.select('Bad � mapping','Bad � mapping')
        self.assertEqual(method,'ocr')

    def test_ocr_failure_does_not_save_corrupt_original(self):
        with self.assertRaisesRegex(pdf.PdfProcessingError,'unreadable encoded'):
            self.select('(cid:77)'*10,'\x01bad',ocr='')

    def test_bad_ocr_does_not_replace_clean_sparse_text(self):
        text='Clean heading'
        result,method,warn=self.select(text,text,True,ocr='(cid:77)')
        self.assertEqual(result,text);self.assertEqual(method,'markitdown');self.assertTrue(warn)

    def test_empty_page_can_be_skipped_with_warning(self):
        self.assertEqual(self.select('','',True,ocr=''),('','native_fallback',True))

    def test_sparse_scan_uses_longer_ocr(self):
        text='Footer'
        result,method,warn=self.select(text,text,True,ocr='Nội dung chính trên trang cùng chú thích cuối trang.')
        self.assertEqual(method,'ocr');self.assertTrue(warn);self.assertNotEqual(result,text)

    def test_sparse_page_keeps_clean_longer_text_with_warning(self):
        text='A complete heading and a detailed paragraph from extraction.'
        result,method,warn=self.select(text,'Logo',True,ocr='A heading')
        self.assertEqual(result,text);self.assertEqual(method,'markitdown');self.assertTrue(warn)

    def test_requested_missing_vietnamese_fails_before_ocr(self):
        response=subprocess.CompletedProcess([],0,'List of available languages (2):\neng\nosd\n','')
        with patch.object(pdf.subprocess,'run',return_value=response):
            with self.assertRaisesRegex(pdf.PdfProcessingError,'vie'):
                pdf._require_ocr_languages('tesseract','vie+eng')

    def test_available_bilingual_languages_accepted(self):
        response=subprocess.CompletedProcess([],0,'List of available languages (3):\neng\nosd\nvie\n','')
        with patch.object(pdf.subprocess,'run',return_value=response):
            pdf._require_ocr_languages('tesseract','vie+eng')

    def test_missing_executable_reports_clear_error(self):
        with patch.object(pdf.subprocess,'run',side_effect=FileNotFoundError()):
            with self.assertRaisesRegex(pdf.PdfProcessingError,'TESSERACT_CMD'):
                pdf._require_ocr_languages('missing','vie+eng')


if __name__=='__main__':unittest.main()
