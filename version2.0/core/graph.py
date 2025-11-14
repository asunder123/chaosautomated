from langgraph.graph import StateGraph, END
from langgraph.types import RunnableConfig

def build_graph(nodes: dict):
    """
    Build a simple linear graph for LangGraph execution.
    Ensures the workflow starts at 'codegen' and proceeds in order of nodes dict.
    """
    # Validate nodes dictionary
    if not nodes or "codegen" not in nodes:
        raise ValueError("Graph must include 'codegen' as the starting node.")

    # Define edges based on the order of keys in nodes
    steps_order = list(nodes.keys())
    edges = []
    for i in range(len(steps_order) - 1):
        edges.append((steps_order[i], steps_order[i + 1]))

    # Create graph configuration
    config = {
        "nodes": steps_order,
        "edges": edges,
        "start": "codegen",
        "end": steps_order[-1],
    }

    # Return a simple app-like object for streaming
    class GraphApp:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.edges = edges

        def stream(self, state, config):
            for step in steps_order:
                # Execute node function
                state = nodesstate
                yield {"current_step": step, "state": state}

    return GraphApp(nodes, edges), config