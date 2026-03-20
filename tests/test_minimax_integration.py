"""Integration tests for MiniMax provider.

These tests make real API calls to the MiniMax API.
Set the MINIMAX_API_KEY environment variable to run them.

Usage:
    MINIMAX_API_KEY=your-key pytest tests/test_minimax_integration.py -v
"""

import os
import json
import unittest

from integuru.util.LLM import LLMSingleton, PROVIDER_PRESETS

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")


@unittest.skipUnless(MINIMAX_API_KEY, "MINIMAX_API_KEY not set")
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def setUp(self):
        LLMSingleton._instance = None
        LLMSingleton.set_provider("minimax")

    def tearDown(self):
        LLMSingleton._instance = None
        LLMSingleton._provider = "openai"
        LLMSingleton._default_model = "gpt-4o"
        LLMSingleton._alternate_model = "o1-preview"

    def test_basic_invoke(self):
        """Test that MiniMax responds to a simple prompt."""
        instance = LLMSingleton.get_instance()
        response = instance.invoke("Say 'hello' and nothing else.")
        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content.strip()), 0)

    def test_function_calling(self):
        """Test that MiniMax supports OpenAI-compatible function calling."""
        function_def = {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name",
                    }
                },
                "required": ["city"],
            },
        }

        instance = LLMSingleton.get_instance()
        response = instance.invoke(
            "What's the weather in Tokyo?",
            functions=[function_def],
            function_call={"name": "get_weather"},
        )

        function_call = response.additional_kwargs.get("function_call", {})
        self.assertEqual(function_call.get("name"), "get_weather")
        args = json.loads(function_call.get("arguments", "{}"))
        self.assertIn("city", args)

    def test_alternate_model(self):
        """Test that the highspeed alternate model works."""
        instance = LLMSingleton.switch_to_alternate_model()
        response = instance.invoke("Reply with just the word 'ok'.")
        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
