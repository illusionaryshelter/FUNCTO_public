"""Regression for the actual gateway's fenced JSON response and bad responses."""

import json
import unittest
from unittest.mock import patch

from local_models.gemini import Gemini


class GeminiOutputTest(unittest.TestCase):
    def test_plain_and_gateway_fenced_objects_have_identical_meaning(self):
        client = Gemini()
        for text in ('{"function_keypoint":"P7"}',
                     '```json\n{\n  "function_keypoint": "P7"\n}\n```'):
            with self.subTest(text=text), patch.object(client, "chat", return_value=text):
                self.assertEqual(client.chat_json("select a point"), {"function_keypoint": "P7"})

    def test_malformed_or_ambiguous_responses_raise_instead_of_retrying(self):
        client = Gemini()
        for text in ('Not a JSON answer', 'Explanation {"function_keypoint":"P7"}',
                     '```json\n{broken}\n```', '{"point":1}{"point":2}'):
            with self.subTest(text=text), patch.object(client, "chat", return_value=text) as request:
                with self.assertRaises(json.JSONDecodeError):
                    client.chat_json("select a point")
                request.assert_called_once()

    def test_non_object_responses_are_rejected(self):
        client = Gemini()
        for text in ('["P7"]', 'null', 'true', '"P7"'):
            with self.subTest(text=text), patch.object(client, "chat", return_value=text):
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    client.chat_json("select a point")


if __name__ == "__main__":
    unittest.main()
