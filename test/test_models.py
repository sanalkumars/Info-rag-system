import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import models
from app import vector


class ModelProviderTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            model_provider='gemini',
            google_api_key='test-key',
            google_embed_model='gemini-embedding-001',
            google_chat_model='gemini-3.5-flash-lite',
            embed_dim=1024,
        )

    @patch('app.models.settings')
    @patch('app.models.bedrock_embed')
    def test_bedrock_remains_the_default_embedding_route(self, bedrock_embed, mock_settings):
        mock_settings.return_value = SimpleNamespace(model_provider='bedrock')
        bedrock_embed.return_value = [0.1]

        result = models.embed('test text')

        self.assertEqual(result, [0.1])
        bedrock_embed.assert_called_once_with('test text')

    @patch('app.vector.settings')
    def test_vector_namespace_keeps_bedrock_data_and_separates_gemini(self, mock_settings):
        mock_settings.return_value = SimpleNamespace(model_provider='bedrock')
        self.assertEqual(vector.namespace('tenant-1'), 'tenant-tenant-1')

        mock_settings.return_value = SimpleNamespace(model_provider='gemini')
        self.assertEqual(vector.namespace('tenant-1'), 'tenant-tenant-1-gemini')

    @patch('app.models.httpx.post')
    def test_gemini_embedding_uses_retrieval_task_and_checks_dimension(self, post):
        response = Mock()
        response.json.return_value = {'embedding': {'values': [0.25] * 1024}}
        post.return_value = response
        with patch('app.models.settings', return_value=self.config):
            result = models.embed('test text', task_type='RETRIEVAL_DOCUMENT')

        self.assertEqual(len(result), 1024)
        post.assert_called_once_with(
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent',
            headers={'x-goog-api-key': 'test-key'},
            json={
                'taskType': 'RETRIEVAL_DOCUMENT',
                'content': {'parts': [{'text': 'test text'}]},
                'outputDimensionality': 1024,
            },
            timeout=120,
        )

    @patch('app.models.httpx.post')
    def test_gemini_chat_returns_model_response(self, post):
        response = Mock()
        response.json.return_value = {
            'candidates': [{'content': {'parts': [{'text': 'Test answer'}]}}]
        }
        post.return_value = response
        with patch('app.models.settings', return_value=self.config):
            result = models.answer('question', '[1] evidence')

        self.assertEqual(result, 'Test answer')
        self.assertEqual(
            post.call_args.args[0],
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent',
        )

    @patch('app.models.httpx.post')
    def test_gemini_embedding_rejects_wrong_dimension(self, post):
        response = Mock()
        response.json.return_value = {'embedding': {'values': [0.25] * 768}}
        post.return_value = response

        with self.assertRaisesRegex(ValueError, 'expected 1024'):
            models._gemini_embed('test text', 'RETRIEVAL_DOCUMENT', self.config)

    @patch('app.models.httpx.post')
    def test_gemini_requires_key_before_making_request(self, post):
        config = SimpleNamespace(google_api_key='')

        with self.assertRaisesRegex(RuntimeError, 'GOOGLE_API_KEY'):
            models._gemini_embed('test text', 'RETRIEVAL_DOCUMENT', config)

        post.assert_not_called()

    @patch('app.models.httpx.post')
    def test_scanned_pdf_uses_gemini_and_preserves_page_numbers(self, post):
        response = Mock()
        response.json.return_value = {
            'candidates': [{
                'finishReason': 'STOP',
                'content': {'parts': [{'text': '[[PAGE 1]]\nScanned text\n[[PAGE 2]]\nMore text'}]},
            }]
        }
        post.return_value = response
        with patch('app.models.settings', return_value=self.config):
            pages = models.extract_pdf_pages(b'%PDF-test')

        self.assertEqual(pages, [(1, 'Scanned text'), (2, 'More text')])
        request = post.call_args.kwargs
        self.assertEqual(request['headers'], {'x-goog-api-key': 'test-key'})
        self.assertEqual(
            request['json']['contents'][0]['parts'][0]['inlineData']['mimeType'],
            'application/pdf',
        )


if __name__ == '__main__':
    unittest.main()
