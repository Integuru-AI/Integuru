import json
import unittest
from unittest.mock import patch, MagicMock

import networkx as nx

from integuru.util.print import generate_code


def _build_graph(response_text: str, response_type: str = "application/json"):
    graph = nx.DiGraph()
    request = MagicMock()
    request.__str__.return_value = "curl 'https://example.com/api/data'"
    graph.add_node(
        "node_1",
        node_type="curl",
        content={
            "key": request,
            "value": {"type": response_type, "text": response_text},
        },
        dynamic_parts=[],
        extracted_parts=["needle-value"],
        input_variables=[],
    )
    return graph


class TestGenerateCodeJsonSizeGuard(unittest.TestCase):
    """
    Regression coverage for https://github.com/Integuru-AI/Integuru/issues/30:
    embedding an oversized JSON API response verbatim in the code-generation
    prompt can exceed the model's context window. generate_code() already
    guarded the HTML/JS response branch against this (a >100000 character
    threshold); this covers the same guard now applied to the JSON branch.
    """

    @patch("integuru.util.print.llm")
    def test_large_json_response_is_not_embedded_in_full(self, mock_llm):
        # given a JSON response so large it would blow past a typical
        # 128k-token context window if embedded verbatim
        large_value = "x" * 200_000
        response_text = json.dumps({"data": {"id": "needle-value"}, "blob": large_value})
        graph = _build_graph(response_text)

        mock_response = MagicMock()
        mock_response.content = "def fetch_data(cookie_string):\n    pass"
        mock_llm.switch_to_alternate_model.return_value.invoke.return_value = mock_response

        # when
        generate_code("node_1", graph)

        # then the prompt sent to the model must not contain the oversized
        # raw response text, but must still contain the resolved key path so
        # the model can navigate the parsed JSON itself
        sent_prompt = mock_llm.switch_to_alternate_model.return_value.invoke.call_args[0][0]
        self.assertNotIn(large_value, sent_prompt)
        self.assertIn("too large to include in full", sent_prompt)
        self.assertIn("'data'", sent_prompt)
        self.assertIn("'id'", sent_prompt)

    @patch("integuru.util.print.llm")
    def test_small_json_response_is_still_embedded(self, mock_llm):
        # given a small JSON response (the common case), behavior is
        # unchanged: the full response text is still included
        response_text = json.dumps({"data": {"id": "needle-value"}})
        graph = _build_graph(response_text)

        mock_response = MagicMock()
        mock_response.content = "def fetch_data(cookie_string):\n    pass"
        mock_llm.switch_to_alternate_model.return_value.invoke.return_value = mock_response

        # when
        generate_code("node_1", graph)

        # then
        sent_prompt = mock_llm.switch_to_alternate_model.return_value.invoke.call_args[0][0]
        self.assertIn("needle-value", sent_prompt)
        self.assertNotIn("too large to include in full", sent_prompt)


if __name__ == "__main__":
    unittest.main()
