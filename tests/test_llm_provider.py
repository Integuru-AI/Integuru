"""Tests for multi-provider LLM support (OpenAI + MiniMax)."""

import os
import unittest
from unittest.mock import patch, MagicMock

from integuru.util.LLM import LLMSingleton, PROVIDER_PRESETS, _detect_provider, MiniMaxChatOpenAI


class TestProviderPresets(unittest.TestCase):
    """Verify provider preset definitions."""

    def test_openai_preset_exists(self):
        self.assertIn("openai", PROVIDER_PRESETS)

    def test_minimax_preset_exists(self):
        self.assertIn("minimax", PROVIDER_PRESETS)

    def test_minimax_preset_values(self):
        p = PROVIDER_PRESETS["minimax"]
        self.assertEqual(p["default_model"], "MiniMax-M2.7")
        self.assertEqual(p["alternate_model"], "MiniMax-M2.7-highspeed")
        self.assertEqual(p["api_key_env"], "MINIMAX_API_KEY")
        self.assertEqual(p["base_url"], "https://api.minimax.io/v1")

    def test_openai_preset_values(self):
        p = PROVIDER_PRESETS["openai"]
        self.assertEqual(p["default_model"], "gpt-4o")
        self.assertEqual(p["alternate_model"], "o1-preview")
        self.assertEqual(p["api_key_env"], "OPENAI_API_KEY")
        self.assertIsNone(p["base_url"])


class TestDetectProvider(unittest.TestCase):
    """Verify auto-detection of LLM provider from environment."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai"}, clear=False)
    def test_defaults_to_openai_when_openai_key_set(self):
        # Remove MINIMAX key if present
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(_detect_provider(), "openai")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-minimax"}, clear=True)
    def test_detects_minimax_when_only_minimax_key(self):
        self.assertEqual(_detect_provider(), "minimax")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai", "MINIMAX_API_KEY": "sk-minimax"}, clear=True)
    def test_prefers_openai_when_both_keys(self):
        self.assertEqual(_detect_provider(), "openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_openai_when_no_keys(self):
        self.assertEqual(_detect_provider(), "openai")


class TestLLMSingletonProvider(unittest.TestCase):
    """Verify LLMSingleton provider switching."""

    def setUp(self):
        # Reset singleton state before each test
        LLMSingleton._instance = None
        LLMSingleton._provider = "openai"
        LLMSingleton._default_model = "gpt-4o"
        LLMSingleton._alternate_model = "o1-preview"

    def test_set_provider_minimax(self):
        LLMSingleton.set_provider("minimax")
        self.assertEqual(LLMSingleton._provider, "minimax")
        self.assertEqual(LLMSingleton._default_model, "MiniMax-M2.7")
        self.assertEqual(LLMSingleton._alternate_model, "MiniMax-M2.7-highspeed")
        self.assertIsNone(LLMSingleton._instance)

    def test_set_provider_openai(self):
        LLMSingleton.set_provider("minimax")
        LLMSingleton.set_provider("openai")
        self.assertEqual(LLMSingleton._provider, "openai")
        self.assertEqual(LLMSingleton._default_model, "gpt-4o")

    def test_set_provider_invalid_raises(self):
        with self.assertRaises(ValueError) as ctx:
            LLMSingleton.set_provider("invalid_provider")
        self.assertIn("Unsupported provider", str(ctx.exception))

    @patch("integuru.util.LLM.MiniMaxChatOpenAI")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_minimax_creates_instance_with_base_url(self, mock_chat):
        LLMSingleton.set_provider("minimax")
        LLMSingleton.get_instance()
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(call_kwargs["api_key"], "test-key")
        self.assertEqual(call_kwargs["temperature"], 1)

    @patch("integuru.util.LLM.ChatOpenAI")
    def test_openai_creates_instance_without_base_url(self, mock_chat):
        LLMSingleton.set_provider("openai")
        LLMSingleton.get_instance()
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o")
        self.assertNotIn("base_url", call_kwargs)

    @patch("integuru.util.LLM.MiniMaxChatOpenAI")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_switch_to_alternate_model_minimax(self, mock_chat):
        LLMSingleton.set_provider("minimax")
        LLMSingleton.switch_to_alternate_model()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7-highspeed")
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")

    @patch("integuru.util.LLM.ChatOpenAI")
    def test_set_default_model_resets_instance(self, mock_chat):
        LLMSingleton.set_provider("minimax")
        LLMSingleton.set_default_model("custom-model")
        self.assertEqual(LLMSingleton._default_model, "custom-model")
        self.assertIsNone(LLMSingleton._instance)

    @patch("integuru.util.LLM.ChatOpenAI")
    def test_singleton_caches_instance(self, mock_chat):
        LLMSingleton.get_instance()
        LLMSingleton.get_instance()
        # Should only create once
        mock_chat.assert_called_once()


class TestLLMSingletonBackwardCompatibility(unittest.TestCase):
    """Ensure existing OpenAI-only behavior is preserved."""

    def setUp(self):
        LLMSingleton._instance = None
        LLMSingleton._provider = "openai"
        LLMSingleton._default_model = "gpt-4o"
        LLMSingleton._alternate_model = "o1-preview"

    @patch("integuru.util.LLM.ChatOpenAI")
    def test_default_behavior_unchanged(self, mock_chat):
        LLMSingleton.get_instance()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o")
        self.assertEqual(call_kwargs["temperature"], 1)

    @patch("integuru.util.LLM.ChatOpenAI")
    def test_set_default_model_still_works(self, mock_chat):
        LLMSingleton.set_default_model("gpt-4o-mini")
        LLMSingleton.get_instance()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o-mini")

    @patch("integuru.util.LLM.ChatOpenAI")
    def test_revert_to_default_model_still_works(self, mock_chat):
        LLMSingleton.revert_to_default_model()
        self.assertEqual(LLMSingleton._alternate_model, "gpt-4o")


if __name__ == "__main__":
    unittest.main()
