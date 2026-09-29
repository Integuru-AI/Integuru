from platform import node
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Set, Optional, Any
from integuru.util.LLM import llm
import json
from langchain_openai import ChatOpenAI
from typing import List
from openai import NotFoundError  # Add this import

def print_dag(
    graph: nx.DiGraph,
    current_node_id: str,
    prefix: str = "",
    is_last: bool = True,
    visited: Optional[Set[str]] = None,
    depth: int = 0,
    max_depth: Optional[int] = None,
) -> None:
    """
    Recursively prints the DAG structure with visual connectors and cUrl.
    """
    if visited is None:
        visited = set()

    connector = "└── " if is_last else "├── "
    new_prefix = prefix + ("    " if is_last else "│   ")

    node_attrs = graph.nodes[current_node_id]
    dynamic_parts = node_attrs.get("dynamic_parts", [])
    key = node_attrs.get("content", "").get("key", "")
    extracted_parts = node_attrs.get("extracted_parts", [])
    input_variables = node_attrs.get("input_variables", [])
    node_type = node_attrs.get("node_type", "")  # Get node type
    
    node_label = f"[{node_type}] [node_id: {current_node_id}]"
    if input_variables:
        node_label += f"\n{new_prefix}    [input_variables: {input_variables}]"
    node_label += f"\n{new_prefix}    [dynamic_parts: {dynamic_parts}]"
    node_label += f"\n{new_prefix}    [extracted_parts: {extracted_parts}]"
    node_label += f"\n{new_prefix}    [{key}]"

    print(f"{prefix}{connector}{node_label}")

    visited.add(current_node_id)

    if max_depth is not None and depth >= max_depth:
        return

    children = list(graph.successors(current_node_id))
    child_count = len(children)

    for i, child_id in enumerate(children):
        is_last_child = i == child_count - 1

        if child_id in visited:
            loop_connector = "└── " if is_last_child else "├── "
            print(f"{new_prefix}{loop_connector}(Already visited) [node_id: {child_id}]")
        else:
            print_dag(
                graph,
                child_id,
                prefix=new_prefix,
                is_last=is_last_child,
                visited=visited,
                depth=depth + 1,
                max_depth=max_depth,
            )


def visualize_dag(graph: nx.DiGraph) -> None:
    """
    Visualizes the DAG using Matplotlib with arrows indicating direction.
    """
    plt.switch_backend("Agg")

    pos = nx.spring_layout(graph) 

    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="lightblue")

    nx.draw_networkx_edges(
        graph, pos, edgelist=graph.edges, arrowstyle="->", arrowsize=20
    )

    labels = {node: f"{node}" for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=10)

    edge_labels = nx.get_edge_attributes(graph, "cUrl") 
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.title("Directed Acyclic Graph (DAG)")
    plt.savefig("dag_visualization.png")
    plt.close()


def find_json_path(json_obj, target_value, current_path=None):
    """
    Finds the path(s) to a given value in a JSON object.
    
    Args:
    json_obj (dict or list): The JSON object to search.
    target_value: The value to find in the JSON object.
    current_path (list): The current path being explored (used for recursion).

    Returns:
    list: A list of dictionaries, each containing 'key_path' and 'value' for each occurrence of the target value.
    """
    if current_path is None:
        current_path = []

    results = []

    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            new_path = current_path + [key]
            if value == target_value:
                results.append({
                    'key_path': new_path,
                    'value': value
                })
            if isinstance(value, (dict, list)):
                results.extend(find_json_path(value, target_value, new_path))
    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            new_path = current_path + [i]
            if item == target_value:
                results.append({
                    'key_path': new_path,
                    'value': item
                })
            if isinstance(item, (dict, list)):
                results.extend(find_json_path(item, target_value, new_path))
    
    return results



def generate_code(node_id: str, graph: nx.DiGraph) -> str:
    """
    Generates Python code for a given node in the graph based on its attributes.
    """

    node_attrs = graph.nodes[node_id]

    if node_attrs.get("node_type", "") == "cookie":
        cookie_value = node_attrs.get('content', {}).get('value', '')
        cookie_key = node_attrs.get('content', {}).get('key', '')
        return f"{cookie_value} = cookie_dict['{cookie_key}']"

    content = node_attrs.get("content", {})
    curl = content.get("key", "")
    response = content.get("value", {})
    response_type = response.get("type", "")
    response_text = response.get("text", "")

    dynamic_parts = node_attrs.get("dynamic_parts", "")
    extracted_parts = node_attrs.get("extracted_parts", "")
    input_variables = node_attrs.get("input_variables", "")
    to_parse_response = True

    parse_response_prompt = ""

    if response_type in ["application/octet-stream", "application/pdf", "application/zip", "image/jpeg", "image/png"]:
        parse_response_prompt = f"""
            The response is a downloadable file of type {response_type}.
            Include code to save the response content to a file with an appropriate extension.
        """

    if "application/json" in response_type:
        key_paths = []
        for extracted_part in extracted_parts:
            key_path = find_json_path(json.loads(response_text), extracted_part)
            key_paths.append(key_path)

        # Large JSON API responses (bulk data dumps, paginated listings, etc.)
        # can easily be hundreds of KB. Embedding them in full blows past the
        # model's context window (see issue #30). The HTML/JS branch above
        # already guards against this; mirror the same threshold here. The
        # key_paths computed above already pinpoint exactly where each
        # variable lives in the parsed JSON, so they're sufficient on their
        # own to write the extraction code without the raw response text.
        if len(response_text) > 100000:
            parse_response_prompt = f"""
                The JSON response is too large to include in full ({len(response_text)} characters).

                Each variable to extract has already been located in the parsed JSON as a key path
                (the sequence of keys/indices from the root of the response to reach that value):
                {key_paths}

                Parse out these variables by navigating the parsed JSON response using the key paths above.

                Through your judgement from analyzing the key paths, if polling is required to retrieve the variables above from the response. If so, implement polling else dont.
            """
        else:
            parse_response_prompt = f"""
                Response:
                {response_text}

                Parse out the following variables from the response using JSON keys:
                {key_paths}

                Through your judgement from analyzing the response, if polling is required to retrieve the variables above from the response. If so, implement polling else dont.
            """

    if "text/html" in response_type or "application/javascript" in response_type:
        if len(response_text) > 100000:
            context_snippets = []
            for part in extracted_parts:
                index = response_text.find(part)
                if index != -1:
                    start = max(0, index - 50)
                    end = min(len(response_text), index + len(part) + 50)
                    snippet = response_text[start:end]
                    context_snippets.append(f"{part}: {snippet}")
            
            parse_response_prompt = f"""
                The HTML response is too long to process entirely. 
                Here are the relevant sections for each variable to be extracted:

                {chr(10).join(context_snippets)}

            """
        else:
            parse_response_prompt = f"""
                Response:
                {response_text}
            """
        parse_response_prompt += f"""
            Parse out the variables following variables locations from the response using regex using locational context: 

            {extracted_parts}
            Do not include the variable in the regex filter as the variable will change. And do not be too specific with the regex.

        """

    dynamic_parts_prompt = ""
    if dynamic_parts:
        dynamic_parts_prompt = f"""
    Instead of hard coding, pass the following variables into the function as parameters in a dict. The dict should have keys thats the same as the value itself
    {dynamic_parts} 

    Keep everything else in the header hardcoded.
    """

    prompt = f"""
    Task:
    Write a Python function with a descriptive name that makes a request like the cURL below:
    {curl}


    Assume cookies are in a variable as parameter called "cookie_string".

    The parameters should be {"1. a dict of all the parameters and 2. Just the cookie string" if dynamic_parts else "only the cookie string"}.

    {dynamic_parts_prompt}

    {parse_response_prompt}
    
    Return a dictionary with the keys as the original parsed values content (needs to be hardcoded) and the values as the parsed values.

    Do not include pseudo-headers or any headers that start with a colon in the request.

    IMPORTANT! Do not include any backticks or markdown syntax AT ALL

    """

    # Make the API call using o1_llm

    llm_model = llm.switch_to_alternate_model()
    try:
        response = llm_model.invoke(prompt)
    except Exception as e:
        print("Switching to default model")
        llm.revert_to_default_model()
        response = llm.switch_to_alternate_model().invoke(prompt)

    # Extract the generated code from the response
    code = response.content.strip()

    # cannot get chatgpt to not return backticks
    if code.startswith("```python"):
        code = code[10:]
    if code.endswith("```"):
        code = code[:-3]

    return code

def aggregate_functions(txt_path, output_path):
    # Read the content of the file
    with open(txt_path, 'r') as file:
        content = file.read()

    # Initialize ChatGPT

    # Prepare the prompt for ChatGPT
    prompt = f"""
    The following text contains multiple Python functions:

    {content}

    Please generate Python code that does the following:    
    1. Fix up the functions if needed in the order they appear in the text.
    2. Leave everything that is hardcoded as is.
    3. Call each function in the order they appear in the text.
    4. The cookies will be hard coded in the file in a string format of key=value;key=value. You will need to convert them to a dict to retrieve values from them.
    5. Pass the return value of each function as an argument to the next function, if applicable.
    6. Ensure that the last function in the text is called last.
    7. Output the entire directly runnable code



    Only provide the Python code, without any explanations or markdown formatting.
    DO NOT include any backticks or markdown syntax AT ALL
    """

    # Get the response from ChatGPT

    llm_model = llm.switch_to_alternate_model()
    try:
        response = llm_model.invoke(prompt)
    except Exception as e:
        print("Switching to default model")
        llm.revert_to_default_model()
        response = llm.switch_to_alternate_model().invoke(prompt)
    # Extract the generated code
    generated_code = response.content.strip()

    # Save the generated code to the specified output file
    with open(output_path, 'w') as file:
        file.write(generated_code)

    print(f"Aggregated function calls have been saved to '{output_path}'")

    return output_path

def generate_obfuscation_map(dynamic_parts_list: List[str]) -> Dict[str, str]:
    obfuscation_map = {}
    for part in dynamic_parts_list:
        # Replace invalid characters with underscores and prepend with 'var_' to ensure it starts with a letter
        safe_key = f"var_{hash(part)}".replace('-', '_').replace('.', '_')
        obfuscation_map[part] = safe_key
    return obfuscation_map

def swap_string_using_obfuscation_map(input_string: str, obfuscation_map: Dict[str, str]) -> str:
    """
    Swaps all parts in the input string that match keys in the obfuscation map with their corresponding values.

    Args:
    input_string (str): The string to perform replacements on.
    obfuscation_map (Dict[str, str]): The obfuscation map with keys to be replaced by their values.

    Returns:
    str: The modified string with replacements made.
    """
    for key, value in obfuscation_map.items():
        input_string = input_string.replace(key, value)
    return input_string

def print_dag_in_reverse(graph: nx.DiGraph, max_depth: Optional[int] = None, to_generate_code: bool = False) -> None:
    """
    Generates the order of requests to be made based on the DAG.
    Prints the DAG starting from source nodes and ending at sink nodes, traversing successors.
    """
    if to_generate_code:
        print("--------------Generating code------------")

    generated_code = ""

    dynamic_parts_list = []

    def _print_dag_recursive(
        current_node_id: str,
        prefix: str = "",
        is_last: bool = True,
        visited: Optional[Set[str]] = None,
        fully_processed: Optional[Set[str]] = None,
        depth: int = 0,
    ) -> None:
        """
        Helper function to recursively print the DAG in reverse order.
        """
        nonlocal generated_code, dynamic_parts_list
        if visited is None:
            visited = set()
        if fully_processed is None:
            fully_processed = set()
    
        if current_node_id in fully_processed:
            return
    
        if current_node_id in visited:
            # Avoid infinite recursion in case of cycles
            return
    
        visited.add(current_node_id)
    
        if max_depth is not None and depth >= max_depth:
            visited.remove(current_node_id)
            return
    
        # Get child nodes (successors)
        children = list(graph.successors(current_node_id))
        child_count = len(children)
    
        # Recursively process child nodes first
        for i, child_id in enumerate(children):
            is_last_child = i == child_count - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            _print_dag_recursive(
                child_id,
                prefix=new_prefix,
                is_last=is_last_child,  # Ensure this argument is passed correctly
                visited=visited,
                fully_processed=fully_processed,
                depth=depth + 1,
            )
    
        # After all children have been processed, print the current node
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{get_node_label(graph, current_node_id)}")
        if to_generate_code:
            generated_code += generate_code(current_node_id, graph) + "\n\n"
        fully_processed.add(current_node_id)
        visited.remove(current_node_id)
    
    def get_node_label(graph: nx.DiGraph, node_id: str) -> str:
        """
        Generates a label for a node in the graph based on its attributes.
        """
        # Get node attributes
        node_attrs = graph.nodes[node_id]
        dynamic_parts = node_attrs.get("dynamic_parts", [])
        extracted_parts = node_attrs.get("extracted_parts", "")
        content = node_attrs.get("content", "")
        key = content.get("key", "")
        input_variables = node_attrs.get("input_variables", "")

        if dynamic_parts:
            dynamic_parts_list.extend(dynamic_parts)
        node_type = node_attrs.get("node_type", "")
        node_label = f"[{node_type}] "
        node_label += f"[node_id: {node_id}]"
        node_label += f" [dynamic_parts: {dynamic_parts}]"
        node_label += f" [extracted_parts: {extracted_parts}]"
        node_label += f" [input_variables: {input_variables}]"
        node_label += f" [{key}]"
        return node_label
    
    # Start from source nodes (nodes with no incoming edges)
    source_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    
    fully_processed = set()
    for idx, source_node in enumerate(source_nodes):
        is_last_source = idx == len(source_nodes) - 1
        _print_dag_recursive(
            source_node,
            prefix="",
            is_last=is_last_source,
            visited=set(),
            fully_processed=fully_processed,
            depth=0,
        )
    
    if to_generate_code:
        obfuscation_map = generate_obfuscation_map(dynamic_parts_list)
        generated_code = swap_string_using_obfuscation_map(generated_code, obfuscation_map)
        with open("generated_code.txt", "w") as f:
            f.write(generated_code)
        
        aggregate_functions("generated_code.txt", "generated_code.py")
        print("--------------Generated integration code in generated_code.py!!------------")



