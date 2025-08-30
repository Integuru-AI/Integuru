from langchain_openai import ChatOpenAI
from ollama import chat
import json
from typing import Dict, List, Any, Optional

class OllamaWrapper:
    """Wrapper class to make Ollama compatible with ChatOpenAI interface"""

    def __init__(self, model: str = "llama3.1", temperature: float = 1.0):
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str, functions: Optional[List[Dict]] = None, function_call: Optional[Dict] = None, **kwargs):
        """
        Invoke Ollama with function calling support, maintaining ChatOpenAI interface compatibility
        """
        messages = [{'role': 'user', 'content': prompt}]

        # Convert functions to Ollama tools format if provided
        tools = []
        if functions:
            for func in functions:
                # Convert ChatOpenAI function format to Ollama tool format
                tool = {
                    'type': 'function',
                    'function': {
                        'name': func['name'],
                        'description': func['description'],
                        'parameters': func['parameters']
                    }
                }
                tools.append(tool)

        # Make the Ollama chat call
        if tools:
            response = chat(
                model=self.model,
                messages=messages,
                tools=tools
            )
        else:
            response = chat(
                model=self.model,
                messages=messages
            )

        # Create a response object that mimics ChatOpenAI's response format
        class OllamaResponse:
            def __init__(self, ollama_response):
                self.content = ollama_response.message.content or ""
                self.additional_kwargs = {}

                # Convert Ollama tool calls to ChatOpenAI format
                if hasattr(ollama_response.message, 'tool_calls') and ollama_response.message.tool_calls:
                    # Take the first tool call (matching current usage pattern)
                    tool_call = ollama_response.message.tool_calls[0]
                    self.additional_kwargs['function_call'] = {
                        'name': tool_call.function.name,
                        'arguments': json.dumps(tool_call.function.arguments)
                    }

        return OllamaResponse(response)

class LLMSingleton:
    _instance = None
    _default_model = "gpt-4o"
    _alternate_model = "o1-preview"
    _ollama_model = "ollama"

    @classmethod
    def get_instance(cls, model: str = None):
        if model is None:
            model = cls._default_model

        if cls._instance is None or (hasattr(cls._instance, 'model') and cls._instance.model != model):
            if model == cls._ollama_model:
                cls._instance = OllamaWrapper(model="llama3.1", temperature=1)
                cls._instance.model = model  # Add model attribute for consistency
            else:
                cls._instance = ChatOpenAI(model=model, temperature=1)
        return cls._instance

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
        """Returns a ChatOpenAI instance configured for o1-preview"""
        # Create a new instance only if we don't have one yet
        if cls._alternate_model == cls._ollama_model:
            cls._instance = OllamaWrapper(model="llama3.1", temperature=1)
            cls._instance.model = cls._alternate_model  # Add model attribute for consistency
        else:
            cls._instance = ChatOpenAI(model=cls._alternate_model, temperature=1)

        return cls._instance

    @classmethod
    def get_ollama_instance(cls):
        """Returns an Ollama instance"""
        cls._instance = OllamaWrapper(model="llama3.1", temperature=1)
        cls._instance.model = cls._ollama_model  # Add model attribute for consistency
        return cls._instance

llm = LLMSingleton()

