import json
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

# Provider configuration presets
PROVIDER_PRESETS = {
    "openai": {
        "default_model": "gpt-4o",
        "alternate_model": "o1-preview",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "minimax": {
        "default_model": "MiniMax-M2.7",
        "alternate_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/v1",
    },
}

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)


def _detect_provider() -> str:
    """Auto-detect the LLM provider from available API keys."""
    if os.environ.get("MINIMAX_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return "minimax"
    return "openai"


class MiniMaxChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that adapts deprecated function-calling kwargs
    to the tools/tool_choice format and strips <think> tags from responses."""

    def invoke(self, input, config=None, *, stop=None, **kwargs):
        # Convert deprecated functions/function_call to tools/tool_choice
        functions = kwargs.pop("functions", None)
        function_call = kwargs.pop("function_call", None)

        if functions:
            tools = [
                {"type": "function", "function": f} for f in functions
            ]
            kwargs["tools"] = tools
            if function_call and isinstance(function_call, dict):
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": function_call["name"]},
                }

        response = super().invoke(input, config=config, stop=stop, **kwargs)

        # Strip <think> tags from content
        if response.content:
            response.content = _THINK_RE.sub("", response.content).strip()

        # Convert tool_calls back to function_call format for compatibility
        if response.tool_calls:
            tc = response.tool_calls[0]
            response.additional_kwargs["function_call"] = {
                "name": tc["name"],
                "arguments": json.dumps(tc["args"]),
            }

        return response


class LLMSingleton:
    _instance = None
    _provider = "openai"
    _default_model = "gpt-4o"
    _alternate_model = "o1-preview"

    @classmethod
    def _create_instance(cls, model: str):
        """Create a ChatOpenAI instance with provider-specific configuration."""
        preset = PROVIDER_PRESETS.get(cls._provider, PROVIDER_PRESETS["openai"])
        kwargs = {"model": model, "temperature": 1}

        if preset["base_url"]:
            kwargs["base_url"] = preset["base_url"]

        api_key = os.environ.get(preset["api_key_env"])
        if api_key:
            kwargs["api_key"] = api_key

        chat_cls = MiniMaxChatOpenAI if cls._provider == "minimax" else ChatOpenAI
        return chat_cls(**kwargs)

    @classmethod
    def get_instance(cls, model: str = None):
        if model is None:
            model = cls._default_model

        if cls._instance is None:
            cls._instance = cls._create_instance(model)
        return cls._instance

    @classmethod
    def set_provider(cls, provider: str):
        """Set the LLM provider and update default models accordingly.

        Args:
            provider: Provider name ("openai" or "minimax").

        Raises:
            ValueError: If the provider is not supported.
        """
        if provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Choose from: {list(PROVIDER_PRESETS.keys())}"
            )
        cls._provider = provider
        preset = PROVIDER_PRESETS[provider]
        cls._default_model = preset["default_model"]
        cls._alternate_model = preset["alternate_model"]
        cls._instance = None  # Reset instance to force recreation

    @classmethod
    def set_default_model(cls, model: str):
        """Set the default model to use when no specific model is requested"""
        cls._default_model = model
        cls._instance = None  # Reset instance to force recreation with new model

    @classmethod
    def revert_to_default_model(cls):
        """Set the default model to use when no specific model is requested"""
        print("Reverting to default model: ", cls._default_model, "Performance will be degraded as Integuru is using non O1 model")
        cls._alternate_model = cls._default_model

    @classmethod
    def switch_to_alternate_model(cls):
        """Returns a ChatOpenAI instance configured for the alternate model"""
        cls._instance = cls._create_instance(cls._alternate_model)
        return cls._instance

llm = LLMSingleton()
