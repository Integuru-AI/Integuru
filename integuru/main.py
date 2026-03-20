from typing import Optional
from integuru.graph_builder import build_graph
from integuru.util.LLM import llm, _detect_provider, PROVIDER_PRESETS

agent = None

async def call_agent(
    model: Optional[str],
    prompt: str,
    har_file_path: str,
    cookie_path: str,
    input_variables: dict = None,
    max_steps: int = 15,
    to_generate_code: bool = False,
    llm_provider: Optional[str] = None,
):
    # Set provider first (before model, since it resets defaults)
    provider = llm_provider or _detect_provider()
    llm.set_provider(provider)

    # Set model (use provider default if not specified)
    if model is not None:
        llm.set_default_model(model)

    global agent
    graph, agent = build_graph(prompt, har_file_path, cookie_path, to_generate_code)
    event_stream = graph.astream(
        {
            "master_node": None,
            "in_process_node": None,
            "to_be_processed_nodes": [],
            "in_process_node_dynamic_parts": [],
            "action_url": "",
            "input_variables": input_variables or {},
        },
        {
            "recursion_limit": max_steps,
        },
    )
    async for event in event_stream:
        # print("+++", event)
        pass
