"""
Generate an interactive Pyvis visualization for the AI BOM graph, mirroring the
existing SBOM graph HTML output.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from pyvis.network import Network

from ai_bom.ai_graph_generator import generate_ai_bom_graph


# Color palette by node type to make layers easier to distinguish.
NODE_COLORS = {
    "code_package": "#1f77b4",
    "build_job": "#ff7f0e",
    "data_asset": "#2ca02c",
    "model_artifact": "#9467bd",
}


def build_ai_bom_visualization(
    distribution: str = "lognormal",
    n_nodes: int = 500,
    seed: int = 42,
    output_path: str = "ai_outputs/AI_BOM_graph.html",
) -> Path:
    """
    Create a Pyvis HTML visualization for the AI BOM graph.

    Args:
        distribution: "lognormal" or "pareto" degree sampling.
        n_nodes: Number of nodes in the generated graph.
        seed: Random seed for reproducibility.
        output_path: Where to write the HTML file.
    """
    G = generate_ai_bom_graph(n_nodes=n_nodes, distribution=distribution, seed=seed)

    net = Network(height="750px", width="100%", directed=True, notebook=False)
    positions = nx.spring_layout(G, seed=seed)

    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "component")
        color = NODE_COLORS.get(node_type, "#97c2fc")
        x, y = positions[node]
        net.add_node(
            node,
            label=node,
            title=f"Type: {node_type}",
            color=color,
            x=x * 1000,
            y=y * 1000,
            size=8,
        )

    for u, v in G.edges():
        net.add_edge(u, v, arrows="to")

    net.set_options(
        """
        var options = {
          "physics": {"enabled": false},
          "interaction": {"hover": true, "dragNodes": true, "zoomView": true}
        }
        """
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output))
    print(f"[AI BOM] graph visualization saved to {output}")
    return output


if __name__ == "__main__":
    build_ai_bom_visualization()

