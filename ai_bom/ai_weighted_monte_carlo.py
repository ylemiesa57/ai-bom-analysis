"""
AI BOM weighted Monte Carlo analysis.

This script mirrors the software SBOM workflow but uses the AI-specific graph
generator and alpha weights. Running it produces:
  * Full Monte Carlo box plots (betweenness + PageRank)
  * Summary-stat CSVs for every node
  * Top-20 ranked box plots needed for PNG exports
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import colorsys

from ai_bom.ai_graph_generator import generate_ai_bom_graph


OUTPUT_ROOT = Path("ai_outputs/weighted")
TOP_K = 20
AI_ALPHA_WEIGHTS = {
    "alpha_C": 0.10,
    "alpha_M": 0.15,
    "alpha_W": 0.15,
    "alpha_D": 0.30,
    "alpha_E": 0.30,
}


def _format_node_label(node_name: str) -> str:
    """Convert node label from ai_node_XXX to N-XXX format."""
    if node_name.startswith("ai_node_"):
        node_id = node_name.replace("ai_node_", "")
        return f"N-{node_id}"
    return node_name


def _get_node_color(node_name: str) -> str:
    """Get consistent color for a node based on its ID (deterministic mapping)."""
    # Extract numeric ID from node name
    if node_name.startswith("ai_node_"):
        node_id = int(node_name.replace("ai_node_", ""))
    else:
        # Fallback: use hash of node name
        node_id = hash(node_name) % 1000
    
    # Use HSV color space for better color distribution
    # Use node_id to determine hue (0-360 degrees)
    hue = (node_id * 137.508) % 360  # Golden angle for better distribution
    saturation = 0.7
    value = 0.9
    
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


def squash_tanh(x, beta=0.6):
    return np.tanh(beta * x)


def squash_capped_linear(x, gamma=0.3):
    return np.minimum(1.0, gamma * x)


def squash_logistic(x, beta=1.0, mu=0.5):
    return 1.0 / (1.0 + np.exp(-beta * (x - mu)))


SQUASH_FUNCS = {
    "tanh": squash_tanh,
    "capped": squash_capped_linear,
    "logistic": squash_logistic,
}


def sample_node_parameters(n_nodes: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Sample normalized node risk attributes."""
    return {
        "CVEs": rng.beta(1.2, 4.0, n_nodes),
        "Misconfigs": np.clip(rng.beta(2.0, 3.5, n_nodes), 0, 1),
        "WeakControls": np.clip(rng.beta(2.0, 3.0, n_nodes), 0, 1),
        "DataQuality": np.clip(rng.beta(3.0, 2.0, n_nodes), 0, 1),
        "Exploitability": np.clip(rng.beta(1.5, 3.0, n_nodes), 0, 1),
    }


def compute_base_risk(params: Dict[str, np.ndarray], alpha: Dict[str, float]) -> np.ndarray:
    """b_v = Σ α_i * feature_i (DataQuality enters as 1 - dq)."""
    return (
        alpha["alpha_C"] * params["CVEs"]
        + alpha["alpha_M"] * params["Misconfigs"]
        + alpha["alpha_W"] * params["WeakControls"]
        + alpha["alpha_D"] * (1 - params["DataQuality"])
        + alpha["alpha_E"] * params["Exploitability"]
    )


def sample_edge_weights(G: nx.DiGraph, rng: np.random.Generator) -> Dict[Tuple[str, str], float]:
    """Transmission weights mimic the software experiment (0.05–1.5)."""
    weights: Dict[Tuple[str, str], float] = {}
    for edge in G.edges():
        roll = rng.random()
        if roll < 0.6:
            weights[edge] = rng.uniform(0.4, 0.9)
        elif roll < 0.9:
            weights[edge] = rng.uniform(0.05, 0.4)
        else:
            weights[edge] = rng.uniform(0.9, 1.5)
    return weights


def montecarlo_on_graph(
    G: nx.DiGraph,
    n_simulations: int,
    squash: str,
    squash_params: Dict[str, float],
    alpha_weights: Dict[str, float],
    seed: int | None,
    save_dir_bc: Path,
    save_dir_pr: Path,
    verbose: bool = True,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    all_bc = {node: [] for node in nodes}
    all_pr = {node: [] for node in nodes}
    squash_fn = SQUASH_FUNCS[squash]

    save_dir_bc.mkdir(parents=True, exist_ok=True)
    save_dir_pr.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[AI BOM] Monte Carlo ({squash}) → {n_simulations} sims on {len(nodes)} nodes")

    for sim in range(n_simulations):
        params = sample_node_parameters(len(nodes), rng)
        base_risk = compute_base_risk(params, alpha_weights)
        weights = sample_edge_weights(G, rng)

        for u, v in G.edges():
            x_val = base_risk[node_index[u]] * weights[(u, v)]
            G.edges[u, v]["weight"] = max(1e-6, squash_fn(x_val, **squash_params))

        bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
        pr = nx.pagerank(G, weight="weight")

        for node in nodes:
            all_bc[node].append(bc[node])
            all_pr[node].append(pr[node])

        if verbose and (sim + 1) % max(1, n_simulations // 10) == 0:
            print(f"    → simulation {sim + 1}/{n_simulations}")

    _write_samples_csv(all_bc, save_dir_bc / "betweenness_samples.csv")
    _write_samples_csv(all_pr, save_dir_pr / "pagerank_samples.csv")
    return all_bc, all_pr


def _write_samples_csv(data: Dict[str, list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["node"] + [f"sim_{i}" for i in range(len(next(iter(data.values()))))]
        writer.writerow(header)
        for node, vals in data.items():
            writer.writerow([node] + vals)
    print(f"[AI BOM] saved samples → {path}")


def plot_interactive_boxplot(
    all_values: Dict[str, list],
    metric_name: str,
    save_dir: Path,
    show: bool = False,
) -> pd.DataFrame:
    """Create the giant per-node box plot and return the summary dataframe."""
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    nodes = list(all_values.keys())
    fig = go.Figure()
    summary_stats = []

    for node in nodes:
        arr = np.array(all_values[node])
        formatted_node = _format_node_label(node)
        node_color = _get_node_color(node)

        summary_stats.append(
            {
                "node": node,
                "median": np.median(arr),
                "q1": np.percentile(arr, 25),
                "q3": np.percentile(arr, 75),
                "min": np.min(arr),
                "max": np.max(arr),
                "mean": np.mean(arr),
            }
        )

        fig.add_trace(
            go.Box(
                y=arr,
                name=formatted_node,
                boxpoints=False,
                line=dict(width=1),
                marker_color=node_color,
                width=0.7,
            )
        )

    stats_df = pd.DataFrame(summary_stats)
    metric_full = _get_metric_full_name(metric_name)
    metric_acronym = _get_metric_acronym(metric_name)
    html_path = save_dir / f"{metric_name.lower()}_boxplot_{timestamp}.html"
    csv_path = save_dir / f"{metric_name.lower()}_summary_stats_{timestamp}.csv"

    fig.update_layout(
        title=f"Weighted {metric_full} Distributions (Monte Carlo)",
        xaxis=dict(title=None, automargin=True, tickangle=45),
        yaxis=dict(title=metric_acronym),
        width=max(1400, len(nodes) * 20),
        height=800,
        template="plotly_white",
        showlegend=False,
        title_font=dict(size=59),
        font=dict(size=33),
        xaxis_title_font=dict(size=53.8),
        xaxis_tickfont=dict(size=35),
        yaxis_title_font=dict(size=59.7),
        yaxis_tickfont=dict(size=35),
    )

    fig.write_html(html_path)
    stats_df.to_csv(csv_path, index=False)
    if show:
        fig.show()

    print(f"[AI BOM] saved {metric_full} box plot → {html_path}")
    print(f"[AI BOM] saved {metric_full} summary CSV → {csv_path}")
    return stats_df


def save_top20_ranked_plot(
    stats_df: pd.DataFrame,
    metric_name: str,
    file_prefix: str,
    output_dir: Path,
    distribution: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_top = stats_df.sort_values("median", ascending=False).head(TOP_K).copy()
    df_top["rank"] = range(1, len(df_top) + 1)
    
    # Format node labels
    df_top["formatted_node"] = df_top["node"].apply(_format_node_label)

    fig = go.Figure()
    for _, row in df_top.iterrows():
        formatted_node = row["formatted_node"]
        node_color = _get_node_color(row["node"])

        fig.add_trace(
            go.Box(
                y=[row["min"], row["q1"], row["median"], row["q3"], row["max"]],
                name=formatted_node,
                boxpoints=False,
                marker_color=node_color,
                hovertemplate=(
                    f"Rank: {row['rank']}<br>"
                    f"Node: {formatted_node}<br>"
                    f"Min: {row['min']:.4f}<br>"
                    f"Q1: {row['q1']:.4f}<br>"
                    f"Median: {row['median']:.4f}<br>"
                    f"Q3: {row['q3']:.4f}<br>"
                    f"Max: {row['max']:.4f}<extra></extra>"
                ),
            )
        )

    # Add trendline with thicker width
    fig.add_trace(
        go.Scatter(
            x=df_top["formatted_node"],
            y=df_top["median"],
            mode="lines",
            line=dict(color="black", width=6),
            name="Median Trend",
            hovertemplate="Median Trend: %{y:.4f}<extra></extra>",
        )
    )

    metric_acronym = _get_metric_acronym(metric_name)
    metric_full = _get_metric_full_name(metric_name)
    
    # Format title with distribution if provided
    if distribution:
        title = f"Top 20 Nodes by {metric_acronym} ({distribution.title()})"
    else:
        title = f"Top 20 Nodes by {metric_acronym}"

    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=metric_acronym,
        showlegend=False,
        width=max(1400, TOP_K * 60),
        height=700,
        template="plotly_white",
        title_font=dict(size=59),
        font=dict(size=33),
        xaxis_title_font=dict(size=53.8),
        xaxis_tickfont=dict(size=35),
        yaxis_title_font=dict(size=59.7),
        yaxis_tickfont=dict(size=35),
    )

    html_path = output_dir / f"{file_prefix}_top20_ranked.html"
    fig.write_html(html_path)
    print(f"[AI BOM] saved top20 {metric_full} plot → {html_path}")
    return html_path


def _read_samples_csv(path: Path) -> Dict[str, list]:
    """Read samples CSV file and return dictionary of node -> list of values."""
    data: Dict[str, list] = {}
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row["node"]
            # Get all simulation columns (sim_0, sim_1, ..., sim_N)
            sim_values = [float(row[key]) for key in row.keys() if key.startswith("sim_")]
            data[node] = sim_values
    return data


def _find_latest_summary_stats_csv(directory: Path, metric_prefix: str) -> Path | None:
    """Find the most recent summary_stats CSV file in the directory."""
    pattern = f"{metric_prefix}_summary_stats_*.csv"
    matching_files = list(directory.glob(pattern))
    if not matching_files:
        return None
    # Sort by modification time, return most recent
    return max(matching_files, key=lambda p: p.stat().st_mtime)


def regenerate_plots_from_csv(
    distribution: str,
    n_simulations: int = 25000,
) -> None:
    """Regenerate plots from existing CSV files without running simulations."""
    dist_root = OUTPUT_ROOT / distribution
    bc_dir = dist_root / f"montecarlo_bc_{n_simulations}"
    pr_dir = dist_root / f"montecarlo_pr_{n_simulations}"

    # Check if directories exist
    if not bc_dir.exists() or not pr_dir.exists():
        print(f"[AI BOM] Error: Directories not found for {distribution}")
        return

    # Try to read from samples CSV first (for full boxplot), fallback to summary_stats
    bc_samples_path = bc_dir / "betweenness_samples.csv"
    pr_samples_path = pr_dir / "pagerank_samples.csv"
    
    if bc_samples_path.exists() and pr_samples_path.exists():
        print(f"[AI BOM] Reading samples CSV files for {distribution}...")
        all_bc = _read_samples_csv(bc_samples_path)
        all_pr = _read_samples_csv(pr_samples_path)
        
        bc_stats = plot_interactive_boxplot(all_bc, metric_name="Betweenness", save_dir=bc_dir, show=False)
        pr_stats = plot_interactive_boxplot(all_pr, metric_name="PageRank", save_dir=pr_dir, show=False)
    else:
        # Fallback: read from summary_stats CSV
        print(f"[AI BOM] Reading summary_stats CSV files for {distribution}...")
        bc_stats_path = _find_latest_summary_stats_csv(bc_dir, "betweenness")
        pr_stats_path = _find_latest_summary_stats_csv(pr_dir, "pagerank")
        
        if bc_stats_path is None or pr_stats_path is None:
            print(f"[AI BOM] Error: Could not find summary_stats CSV files for {distribution}")
            return
        
        bc_stats = pd.read_csv(bc_stats_path)
        pr_stats = pd.read_csv(pr_stats_path)
        print(f"[AI BOM] Loaded stats from {bc_stats_path.name} and {pr_stats_path.name}")

    # Regenerate top20 plots
    save_top20_ranked_plot(bc_stats, metric_name="Betweenness", file_prefix="betweenness", output_dir=bc_dir, distribution=distribution)
    save_top20_ranked_plot(pr_stats, metric_name="PageRank", file_prefix="pagerank", output_dir=pr_dir, distribution=distribution)
    print(f"[AI BOM] Regenerated plots for {distribution}")


def run_ai_weighted_analysis(
    distribution: str,
    n_nodes: int = 500,
    n_simulations: int = 25000,
    seed: int = 42,
    squash: str = "tanh",
    squash_params: Dict[str, float] | None = None,
    generation_mode: str = "software_like",
    lognormal_mean: float = 1.0,
    lognormal_sigma: float = 0.7,
    pareto_alpha: float = 1.5,
    pareto_scale: float = 1.0,
    min_degree: int = 0,
    max_degree: int = 30,
) -> None:
    """Run full Monte Carlo analysis (regenerates simulations)."""
    if squash_params is None:
        squash_params = {"beta": 0.6}

    dist_root = OUTPUT_ROOT / distribution
    bc_dir = dist_root / f"montecarlo_bc_{n_simulations}"
    pr_dir = dist_root / f"montecarlo_pr_{n_simulations}"

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
    all_bc, all_pr = montecarlo_on_graph(
        G,
        n_simulations=n_simulations,
        squash=squash,
        squash_params=squash_params,
        alpha_weights=AI_ALPHA_WEIGHTS,
        seed=seed,
        save_dir_bc=bc_dir,
        save_dir_pr=pr_dir,
        verbose=True,
    )

    bc_stats = plot_interactive_boxplot(all_bc, metric_name="Betweenness", save_dir=bc_dir, show=False)
    pr_stats = plot_interactive_boxplot(all_pr, metric_name="PageRank", save_dir=pr_dir, show=False)

    save_top20_ranked_plot(bc_stats, metric_name="Betweenness", file_prefix="betweenness", output_dir=bc_dir, distribution=distribution)
    save_top20_ranked_plot(pr_stats, metric_name="PageRank", file_prefix="pagerank", output_dir=pr_dir, distribution=distribution)


if __name__ == "__main__":
    # Regenerate plots from existing CSV files (no simulations)
    for dist in ("lognormal", "pareto"):
        regenerate_plots_from_csv(dist)
