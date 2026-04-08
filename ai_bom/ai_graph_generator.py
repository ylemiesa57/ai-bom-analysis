"""
AI BOM Graph Generator.

Creates directed graphs that approximate an AI/ML supply-chain (code -> build ->
data -> model) with configurable in/out-degree distributions. The generator is
used by the AI-specific unweighted and weighted simulations to keep node
structure consistent with the risk model requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np


# Default layer mix roughly matches the description used elsewhere in the repo.
NODE_TYPE_DISTRIBUTION: Dict[str, float] = {
    "code_package": 0.40,
    "build_job": 0.25,
    "data_asset": 0.20,
    "model_artifact": 0.15,
}


@dataclass(frozen=True)
class DegreeBand:
    dist: str
    max_degree: int
    mean: float | None = None
    sigma: float | None = None
    alpha: float | None = None
    scale: float | None = None


# Degree settings derived from the research notes in the repo. We keep separate
# configs for lognormal vs pareto to emulate both scenarios that the software
# SBOM scripts already support.
DEGREE_CONFIG: Dict[str, Dict[str, Dict[str, DegreeBand]]] = {
    "code_package": {
        "lognormal": {
            "in": DegreeBand(dist="lognormal", mean=0.5, sigma=0.7, max_degree=10),
            "out": DegreeBand(dist="lognormal", mean=1.5, sigma=0.9, max_degree=12),
        },
        "pareto": {
            "in": DegreeBand(dist="pareto", alpha=2.0, scale=1.5, max_degree=10),
            "out": DegreeBand(dist="pareto", alpha=2.0, scale=2.5, max_degree=12),
        },
    },
    "build_job": {
        "lognormal": {
            "in": DegreeBand(dist="lognormal", mean=0.4, sigma=0.6, max_degree=8),
            "out": DegreeBand(dist="lognormal", mean=1.0, sigma=0.7, max_degree=10),
        },
        "pareto": {
            "in": DegreeBand(dist="pareto", alpha=2.2, scale=1.2, max_degree=8),
            "out": DegreeBand(dist="pareto", alpha=2.2, scale=2.0, max_degree=10),
        },
    },
    "data_asset": {
        "lognormal": {
            "in": DegreeBand(dist="lognormal", mean=0.3, sigma=0.5, max_degree=6),
            "out": DegreeBand(dist="lognormal", mean=0.9, sigma=0.6, max_degree=8),
        },
        "pareto": {
            "in": DegreeBand(dist="pareto", alpha=2.5, scale=1.0, max_degree=6),
            "out": DegreeBand(dist="pareto", alpha=2.5, scale=1.8, max_degree=8),
        },
    },
    "model_artifact": {
        "lognormal": {
            "in": DegreeBand(dist="lognormal", mean=0.2, sigma=0.4, max_degree=5),
            "out": DegreeBand(dist="lognormal", mean=0.8, sigma=0.5, max_degree=7),
        },
        "pareto": {
            "in": DegreeBand(dist="pareto", alpha=2.8, scale=0.8, max_degree=5),
            "out": DegreeBand(dist="pareto", alpha=2.8, scale=1.5, max_degree=7),
        },
    },
}


def _sample_degree(rng: np.random.Generator, band: DegreeBand) -> int:
    """Sample a degree from the provided configuration."""
    if band.dist == "lognormal":
        value = rng.lognormal(mean=band.mean, sigma=band.sigma)
    elif band.dist == "pareto":
        value = rng.pareto(band.alpha) * (band.scale or 1.0)
    else:
        raise ValueError(f"Unsupported distribution '{band.dist}'")

    return int(np.clip(round(value), 0, band.max_degree))


def _assign_node_types(n_nodes: int, rng: np.random.Generator) -> Dict[str, str]:
    names = [f"ai_node_{i}" for i in range(n_nodes)]
    types = list(NODE_TYPE_DISTRIBUTION.keys())
    probs = np.array(list(NODE_TYPE_DISTRIBUTION.values()))
    probs = probs / probs.sum()

    node_types = {}
    for name in names:
        node_types[name] = rng.choice(types, p=probs)
    return node_types


def _connect_components(G: nx.DiGraph, rng: np.random.Generator) -> None:
    """Ensure the graph is weakly connected (needed for eigenvector/pagerank)."""
    components = list(nx.weakly_connected_components(G))
    if len(components) <= 1:
        return

    for idx in range(len(components) - 1):
        comp_a = list(components[idx])
        comp_b = list(components[idx + 1])
        u = rng.choice(comp_a)
        v = rng.choice(comp_b)
        G.add_edge(u, v)
        if rng.random() < 0.5:
            G.add_edge(v, u)


def _generate_layered_ai_graph(
    n_nodes: int,
    distribution: str,
    rng: np.random.Generator,
) -> nx.DiGraph:
    """Generate the original AI-typed layered graph."""
    node_types = _assign_node_types(n_nodes, rng)
    node_order: List[str] = list(node_types.keys())

    G = nx.DiGraph()
    for node, node_type in node_types.items():
        G.add_node(node, node_type=node_type)

    for idx, node in enumerate(node_order):
        type_config = DEGREE_CONFIG[node_types[node]][distribution]
        in_degree = _sample_degree(rng, type_config["in"])
        out_degree = _sample_degree(rng, type_config["out"])

        possible_predecessors = node_order[:idx]
        if possible_predecessors and in_degree > 0:
            preds = rng.choice(
                possible_predecessors,
                size=min(in_degree, len(possible_predecessors)),
                replace=False,
            )
            for pred in preds:
                G.add_edge(pred, node)

        possible_successors = node_order[idx + 1 :]
        if possible_successors and out_degree > 0:
            succs = rng.choice(
                possible_successors,
                size=min(out_degree, len(possible_successors)),
                replace=False,
            )
            for succ in succs:
                G.add_edge(node, succ)

    _connect_components(G, rng)
    return G


def _generate_software_like_ai_graph(
    n_nodes: int,
    distribution: str,
    rng: np.random.Generator,
    lognormal_mean: float,
    lognormal_sigma: float,
    pareto_alpha: float,
    pareto_scale: float,
    min_degree: int,
    max_degree: int,
) -> nx.DiGraph:
    """
    Generate AI graph with the same structure logic as software SBOM generation:
    sample in-degree targets globally, then attach random sources.
    """
    G = nx.DiGraph()
    nodes = [f"ai_node_{idx}" for idx in range(n_nodes)]
    G.add_nodes_from(nodes)
    for node in nodes:
        G.nodes[node]["node_type"] = "component"

    if distribution == "lognormal":
        target_in_degree = np.clip(
            np.round(rng.lognormal(mean=lognormal_mean, sigma=lognormal_sigma, size=n_nodes)).astype(int),
            min_degree,
            max_degree,
        )
    else:
        target_in_degree = np.clip(
            np.round(rng.pareto(pareto_alpha, size=n_nodes) * pareto_scale).astype(int),
            min_degree,
            max_degree,
        )

    for tgt_idx, n_edges in enumerate(target_in_degree):
        tgt = nodes[tgt_idx]
        if n_edges <= 0:
            continue
        possible_sources = [node for node in nodes if node != tgt]
        sources = rng.choice(possible_sources, size=min(n_edges, len(possible_sources)), replace=False)
        for src in sources:
            G.add_edge(src, tgt)

    # Mirror software generator rule: the first node has no incoming edges.
    root = nodes[0]
    for pred in list(G.predecessors(root)):
        G.remove_edge(pred, root)

    return G


def generate_ai_bom_graph(
    n_nodes: int = 500,
    distribution: str = "lognormal",
    seed: int | None = 42,
    generation_mode: str = "software_like",
    lognormal_mean: float = 1.0,
    lognormal_sigma: float = 0.7,
    pareto_alpha: float = 1.5,
    pareto_scale: float = 1.0,
    min_degree: int = 0,
    max_degree: int = 30,
) -> nx.DiGraph:
    """
    Build an AI BOM graph with the requested degree distribution.

    Args:
        n_nodes: Number of nodes to generate.
        distribution: "lognormal" or "pareto".
        seed: RNG seed for reproducibility.
        generation_mode: "software_like" for software-matching structure logic,
            or "layered" for the original AI typed-layer generator.
        lognormal_mean: Global lognormal mean for software_like mode.
        lognormal_sigma: Global lognormal sigma for software_like mode.
        pareto_alpha: Global pareto alpha for software_like mode.
        pareto_scale: Global pareto scale for software_like mode.
        min_degree: Minimum sampled in-degree for software_like mode.
        max_degree: Maximum sampled in-degree for software_like mode.
    """
    if distribution not in {"lognormal", "pareto"}:
        raise ValueError("distribution must be 'lognormal' or 'pareto'")

    if generation_mode not in {"software_like", "layered"}:
        raise ValueError("generation_mode must be 'software_like' or 'layered'")

    rng = np.random.default_rng(seed)
    if generation_mode == "layered":
        return _generate_layered_ai_graph(n_nodes=n_nodes, distribution=distribution, rng=rng)

    return _generate_software_like_ai_graph(
        n_nodes=n_nodes,
        distribution=distribution,
        rng=rng,
        lognormal_mean=lognormal_mean,
        lognormal_sigma=lognormal_sigma,
        pareto_alpha=pareto_alpha,
        pareto_scale=pareto_scale,
        min_degree=min_degree,
        max_degree=max_degree,
    )


if __name__ == "__main__":
    out_dir = Path("ai_outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    for dist in ("lognormal", "pareto"):
        graph = generate_ai_bom_graph(n_nodes=1000, distribution=dist, seed=7)
        print(f"[AI BOM] {dist}: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

