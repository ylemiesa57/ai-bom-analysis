"""
Generate AI BOM unweighted centrality rankings (top-20) for both the lognormal
and pareto degree configurations. The outputs mirror what the existing software
SBOM scripts produce so that we can create the same eight plots/images.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from ai_bom.ai_graph_generator import generate_ai_bom_graph


OUTPUT_ROOT = Path("ai_outputs/unweighted")
TOP_K = 20


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_top20_bar_chart(
    scores: Dict[str, float],
    metric_name: str,
    distribution: str,
) -> Path:
    dist_dir = _ensure_dir(OUTPUT_ROOT / distribution)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    df = pd.DataFrame(sorted_scores, columns=["node", metric_name])
    csv_path = dist_dir / f"top20_{metric_name.lower()}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    fig = go.Figure(
        go.Bar(
            x=[node for node, _ in sorted_scores],
            y=[value for _, value in sorted_scores],
            marker_color="#1f77b4",
            marker_line_color="black",
            marker_line_width=1,
            hovertemplate="Node %{x}<br>"
            + f"{metric_name}: "+"%{y:.6f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"AI BOM Top {TOP_K} Nodes by {metric_name} ({distribution.title()})",
        xaxis_title="Node (ranked)",
        yaxis_title=metric_name,
        template="plotly_white",
        width=max(1400, TOP_K * 40),
        height=650,
    )

    # Overwrite deterministic filename so image_creation can reference it directly.
    html_path = dist_dir / f"top20_{metric_name.lower()}_bar.html"
    fig.write_html(html_path)
    return html_path


def run_unweighted_ai_analysis(distribution: str, n_nodes: int = 500, seed: int = 42) -> None:
    """Build the AI BOM graph, compute unweighted centralities, and store results."""
    G = generate_ai_bom_graph(n_nodes=n_nodes, distribution=distribution, seed=seed)
    betweenness = nx.betweenness_centrality(G, weight=None, normalized=True)
    pagerank = nx.pagerank(G, alpha=0.85)

    betw_path = _save_top20_bar_chart(betweenness, "Betweenness", distribution)
    pr_path = _save_top20_bar_chart(pagerank, "PageRank", distribution)
    print(f"[AI BOM][{distribution}] saved {betw_path}")
    print(f"[AI BOM][{distribution}] saved {pr_path}")


if __name__ == "__main__":
    for dist in ("lognormal", "pareto"):
        run_unweighted_ai_analysis(dist)

