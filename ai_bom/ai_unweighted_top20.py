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
import colorsys
import zlib

from ai_bom.ai_graph_generator import generate_ai_bom_graph


OUTPUT_ROOT = Path("ai_outputs/unweighted")
TOP_K = 20


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_node_label(node_name: str) -> str:
    """Convert node label from ai_node_XXX to N-XXX format."""
    if node_name.startswith("ai_node_"):
        node_id = node_name.replace("ai_node_", "")
        return f"N-{node_id}"
    return node_name


def _get_node_color(node_name: str, lighter: bool = False) -> str:
    """Get consistent color for a node based on its ID (deterministic mapping).
    
    Args:
        node_name: The node name (e.g., 'ai_node_123')
        lighter: If True, use lighter colors (for bar charts)
    """
    # Extract numeric ID from node name
    if node_name.startswith("ai_node_"):
        node_id = int(node_name.replace("ai_node_", ""))
    else:
        # Fallback: derive a stable id from the node name. Python's
        # built-in hash() is randomized per-process for str/bytes
        # (PYTHONHASHSEED), so node_id -- and therefore this node's
        # color -- would silently change between runs/processes despite
        # this function's docstring promising a "deterministic mapping".
        # zlib.crc32 is a fixed, portable, non-randomized hash, so the
        # same node name always maps to the same color.
        node_id = zlib.crc32(node_name.encode("utf-8")) % 1000
    
    # Use HSV color space for better color distribution
    # Use node_id to determine hue (0-360 degrees)
    hue = (node_id * 137.508) % 360  # Golden angle for better distribution
    # Lighter saturation for bar charts, normal for box plots
    saturation = 0.45 if lighter else 0.7
    value = 0.95 if lighter else 0.9
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
    
    # Convert to hex format
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"


def _get_metric_acronym(metric_name: str) -> str:
    """Get acronym for metric (PRC for PageRank, BC for Betweenness)."""
    if metric_name.lower() == "pagerank":
        return "PRC"
    elif metric_name.lower() == "betweenness":
        return "BC"
    return metric_name


def _get_metric_full_name(metric_name: str) -> str:
    """Get full display name for metric (for y-axis labels)."""
    if metric_name.lower() == "pagerank":
        return "PageRank"
    elif metric_name.lower() == "betweenness":
        return "Betweenness"
    return metric_name


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

    # Format node labels and get colors (lighter for bar charts)
    formatted_nodes = [_format_node_label(node) for node, _ in sorted_scores]
    node_colors = [_get_node_color(node, lighter=True) for node, _ in sorted_scores]
    values = [value for _, value in sorted_scores]

    fig = go.Figure()
    
    # Add bars with individual colors (no outlines)
    fig.add_trace(
        go.Bar(
            x=formatted_nodes,
            y=values,
            marker_color=node_colors,
            hovertemplate="Node %{x}<br>"
            + f"{_get_metric_full_name(metric_name)}: %{{y:.6f}}<extra></extra>",
        )
    )
    
    # Add trendline (median trend)
    fig.add_trace(
        go.Scatter(
            x=formatted_nodes,
            y=values,
            mode="lines",
            line=dict(color="black", width=6),
            name="Trend",
            hovertemplate="Trend: %{y:.6f}<extra></extra>",
        )
    )

    metric_acronym = _get_metric_acronym(metric_name)
    metric_full = _get_metric_full_name(metric_name)
    metric_acronym = _get_metric_acronym(metric_name)

    fig.update_layout(
        title=f"Top 20 Nodes by {metric_acronym} ({distribution.title()})",
        xaxis_title=None,
        yaxis_title=metric_acronym,
        template="plotly_white",
        width=max(1400, TOP_K * 40),
        height=650,
        showlegend=False,
        title_font=dict(size=59),
        font=dict(size=33),
        xaxis=dict(
            title_font=dict(size=53.8),
            tickfont=dict(size=35),
        ),
        yaxis=dict(
            title_font=dict(size=59.7),
            tickfont=dict(size=35),
        ),
    )

    # Overwrite deterministic filename so image_creation can reference it directly.
    html_path = dist_dir / f"top20_{metric_name.lower()}_bar.html"
    fig.write_html(html_path)
    return html_path


def _find_latest_csv(directory: Path, metric_name: str) -> Path | None:
    """Find the most recent CSV file matching the pattern."""
    pattern = f"top20_{metric_name.lower()}_*.csv"
    matching_files = list(directory.glob(pattern))
    if not matching_files:
        return None
    # Sort by modification time, return most recent
    return max(matching_files, key=lambda p: p.stat().st_mtime)


def regenerate_unweighted_plots_from_csv(distribution: str) -> None:
    """Regenerate plots from existing CSV files without recomputing centralities."""
    dist_dir = OUTPUT_ROOT / distribution
    
    if not dist_dir.exists():
        print(f"[AI BOM] Error: Directory not found for {distribution}")
        return
    
    # Find latest CSV files
    bc_csv_path = _find_latest_csv(dist_dir, "betweenness")
    pr_csv_path = _find_latest_csv(dist_dir, "pagerank")
    
    if bc_csv_path is None or pr_csv_path is None:
        print(f"[AI BOM] Error: Could not find CSV files for {distribution}")
        return
    
    # Read CSV files
    bc_df = pd.read_csv(bc_csv_path)
    pr_df = pd.read_csv(pr_csv_path)
    
    # Convert to dictionary format expected by _save_top20_bar_chart
    betweenness = dict(zip(bc_df["node"], bc_df["Betweenness"]))
    pagerank = dict(zip(pr_df["node"], pr_df["PageRank"]))
    
    # Regenerate plots
    betw_path = _save_top20_bar_chart(betweenness, "Betweenness", distribution)
    pr_path = _save_top20_bar_chart(pagerank, "PageRank", distribution)
    print(f"[AI BOM][{distribution}] regenerated {betw_path}")
    print(f"[AI BOM][{distribution}] regenerated {pr_path}")


def run_unweighted_ai_analysis(
    distribution: str,
    n_nodes: int = 500,
    seed: int = 42,
    generation_mode: str = "software_like",
    lognormal_mean: float = 1.0,
    lognormal_sigma: float = 0.7,
    pareto_alpha: float = 1.5,
    pareto_scale: float = 1.0,
    min_degree: int = 0,
    max_degree: int = 30,
) -> None:
    """Build the AI BOM graph, compute unweighted centralities, and store results."""
    G = generate_ai_bom_graph(
        n_nodes=n_nodes,
        distribution=distribution,
        seed=seed,
        generation_mode=generation_mode,
        lognormal_mean=lognormal_mean,
        lognormal_sigma=lognormal_sigma,
        pareto_alpha=pareto_alpha,
        pareto_scale=pareto_scale,
        min_degree=min_degree,
        max_degree=max_degree,
    )
    betweenness = nx.betweenness_centrality(G, weight=None, normalized=True)
    pagerank = nx.pagerank(G, alpha=0.85)

    betw_path = _save_top20_bar_chart(betweenness, "Betweenness", distribution)
    pr_path = _save_top20_bar_chart(pagerank, "PageRank", distribution)
    print(f"[AI BOM][{distribution}] saved {betw_path}")
    print(f"[AI BOM][{distribution}] saved {pr_path}")


if __name__ == "__main__":
    # Regenerate plots from existing CSV files (no recomputation)
    for dist in ("lognormal", "pareto"):
        regenerate_unweighted_plots_from_csv(dist)

