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

from ai_bom.ai_graph_generator import generate_ai_bom_graph


OUTPUT_ROOT = Path("ai_outputs/weighted")
TOP_K = 20
AI_ALPHA_WEIGHTS = {
    "alpha_C": 0.25,
    "alpha_M": 0.20,
    "alpha_W": 0.20,
    "alpha_D": 0.15,
    "alpha_E": 0.20,
}


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
                name=node,
                boxpoints=False,
                line=dict(width=1),
                marker_color="rgba(0,123,255,0.6)",
                width=0.7,
            )
        )

    stats_df = pd.DataFrame(summary_stats)
    html_path = save_dir / f"{metric_name.lower()}_boxplot_{timestamp}.html"
    csv_path = save_dir / f"{metric_name.lower()}_summary_stats_{timestamp}.csv"

    fig.update_layout(
        title=f"{metric_name} Distributions (AI BOM Monte Carlo)",
        xaxis=dict(title="Node Index", automargin=True, tickangle=45),
        yaxis=dict(title=metric_name),
        width=max(1400, len(nodes) * 20),
        height=800,
        template="plotly_white",
        showlegend=False,
    )

    fig.write_html(html_path)
    stats_df.to_csv(csv_path, index=False)
    if show:
        fig.show()

    print(f"[AI BOM] saved {metric_name} box plot → {html_path}")
    print(f"[AI BOM] saved {metric_name} summary CSV → {csv_path}")
    return stats_df


def save_top20_ranked_plot(
    stats_df: pd.DataFrame,
    metric_name: str,
    file_prefix: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_top = stats_df.sort_values("median", ascending=False).head(TOP_K).copy()
    df_top["rank"] = range(1, len(df_top) + 1)

    fig = go.Figure()
    for _, row in df_top.iterrows():
        fig.add_trace(
            go.Box(
                y=[row["min"], row["q1"], row["median"], row["q3"], row["max"]],
                name=f"Node {row['node']}",
                boxpoints=False,
                hovertemplate=(
                    f"Rank: {row['rank']}<br>"
                    f"Node: {row['node']}<br>"
                    f"Min: {row['min']:.4f}<br>"
                    f"Q1: {row['q1']:.4f}<br>"
                    f"Median: {row['median']:.4f}<br>"
                    f"Q3: {row['q3']:.4f}<br>"
                    f"Max: {row['max']:.4f}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[f"Node {node}" for node in df_top["node"]],
            y=df_top["median"],
            mode="lines",
            line=dict(color="black", width=3),
            name="Median Trend",
            hovertemplate="Median Trend: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"AI BOM Top {TOP_K} Nodes by {metric_name}",
        xaxis_title="Node (ranked)",
        yaxis_title=metric_name,
        showlegend=False,
        width=max(1400, TOP_K * 60),
        height=700,
        template="plotly_white",
    )

    html_path = output_dir / f"{file_prefix}_top20_ranked.html"
    fig.write_html(html_path)
    print(f"[AI BOM] saved top20 {metric_name} plot → {html_path}")
    return html_path


def run_ai_weighted_analysis(
    distribution: str,
    n_nodes: int = 500,
    n_simulations: int = 5000,
    seed: int = 42,
    squash: str = "tanh",
    squash_params: Dict[str, float] | None = None,
) -> None:
    if squash_params is None:
        squash_params = {"beta": 0.6}

    dist_root = OUTPUT_ROOT / distribution
    bc_dir = dist_root / f"montecarlo_bc_{n_simulations}"
    pr_dir = dist_root / f"montecarlo_pr_{n_simulations}"

    G = generate_ai_bom_graph(n_nodes=n_nodes, distribution=distribution, seed=seed)
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

    save_top20_ranked_plot(bc_stats, metric_name="Betweenness", file_prefix="betweenness", output_dir=bc_dir)
    save_top20_ranked_plot(pr_stats, metric_name="PageRank", file_prefix="pagerank", output_dir=pr_dir)


if __name__ == "__main__":
    for dist in ("lognormal", "pareto"):
        run_ai_weighted_analysis(dist)

