import unittest
from integuru.agent import IntegrationAgent
from integuru.models.agent_state import AgentState
from unittest.mock import patch, MagicMock

class TestIntegrationAgent(unittest.TestCase):

    def setUp(self):
        self.prompt = "Test prompt"
        self.har_file_path = "test.har"
        self.cookie_path = "test_cookies.json"
        self.agent = IntegrationAgent(self.prompt, self.har_file_path, self.cookie_path)
        self.state = AgentState(
            master_node=None,
            in_process_node=None,
            to_be_processed_nodes=[],
            in_process_node_dynamic_parts=[],
            action_url="",
            input_variables={}
        )

    @patch('integuru.agent.llm.get_instance')
    def test_end_url_identify_agent(self, mock_llm_instance):
        mock_response = MagicMock()
        mock_response.additional_kwargs = {
            'function_call': {
                'arguments': '{"url": "http://example.com/action"}'
            }
        }
        mock_llm_instance.return_value.invoke.return_value = mock_response

        updated_state = self.agent.end_url_identify_agent(self.state)
        self.assertEqual(updated_state[self.agent.ACTION_URL_KEY], "http://example.com/action")

    @patch('integuru.agent.llm.get_instance')
    def test_input_variables_identifying_agent(self, mock_llm_instance):
        self.state[self.agent.IN_PROCESS_NODE_KEY] = "node_1"
        self.state[self.agent.INPUT_VARIABLES_KEY] = {"var1": "value1"}
        self.agent.dag_manager.graph.add_node("node_1", content={"key": MagicMock()})
        self.agent.dag_manager.graph.nodes["node_1"]["content"]["key"].to_curl_command.return_value = "curl command"

        mock_response = MagicMock()
        mock_response.additional_kwargs = {
            'function_call': {
                'arguments': '{"identified_variables": [{"variable_name": "var1", "variable_value": "value1"}]}'
            }
        }
        mock_llm_instance.return_value.invoke.return_value = mock_response

        updated_state = self.agent.input_variables_identifying_agent(self.state)
        self.assertEqual(updated_state[self.agent.INPUT_VARIABLES_KEY], {"var1": "value1"})

    @patch('integuru.agent.llm.get_instance')
    def test_dynamic_part_identifying_agent(self, mock_llm_instance):
        self.state[self.agent.TO_BE_PROCESSED_NODES_KEY] = ["node_1"]
        self.agent.dag_manager.graph.add_node("node_1", content={"key": MagicMock()})
        self.agent.dag_manager.graph.nodes["node_1"]["content"]["key"].to_minified_curl_command.return_value = "curl command"

        mock_response = MagicMock()
        mock_response.additional_kwargs = {
            'function_call': {
                'arguments': '{"dynamic_parts": ["dynamic_part1"]}'
            }
        }
        mock_llm_instance.return_value.invoke.return_value = mock_response

        updated_state = self.agent.dynamic_part_identifying_agent(self.state)
        self.assertEqual(updated_state[self.agent.IN_PROCESS_NODE_DYNAMIC_PARTS_KEY], ["dynamic_part1"])

if __name__ == '__main__':
    unittest.main()
