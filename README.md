# AI BOM Analysis Toolkit

End-to-end toolkit for analyzing AI/ML supply-chain bills of materials (AI BOMs).
It mirrors the SBOM workflow from `canon_urop/node_exploitability` but swaps in
an AI-specific graph generator, Monte Carlo risk simulator, and image pipeline
tuned to the α-weight vector `[0.10, 0.15, 0.15, 0.30, 0.30]` (CVEs, Misconfigs,
WeakControls, 1 - DataQuality, Exploitability).

---

## Project Layout

```
ai_bom/                # Core Python modules
  ├─ ai_graph_generator.py        # Builds AI/ML dependency graphs
  ├─ ai_unweighted_top20.py       # Unweighted betweenness + PageRank plots
  ├─ ai_weighted_monte_carlo.py   # Weighted Monte Carlo simulations (25k trials)
  ├─ ai_image_creation.py         # HTML → PNG conversion via Playwright
  └─ ai_graph_visualization.py    # Pyvis network view (AI_BOM_graph.html)
ai_outputs/            # HTML + CSV outputs (graph viz, plots, summaries)
ai_images/             # Final PNG snapshots (8 total: weighted/unweighted x 2 metrics)
README.md              # You are here
LICENSE                # MIT
```

---

## Environment Setup

```bash
cd /Users/yaphetlemiesa/UROP_2025_fall/ai-bom-analysis
python3 -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
```

`requirements.txt` tracks the packages this toolkit actually imports (including
`pillow`, needed by `ai_bom/create_multisize_2x8_grid.py` and
`ai_bom/create_plot_grid.py` for the PNG grid assembly, which the old inline
pip command here omitted). `tqdm` was listed previously but isn't imported
anywhere in `ai_bom/`, so it's been dropped.

- Playwright is required for `ai_image_creation.py`. Running `playwright install chromium`
  once per machine resolves the `sync_playwright` import error.
- If you plan to rerun Monte Carlo simulations, be sure the environment has enough RAM;
  5,000 weighted runs on a 500-node graph take ~5 minutes on a modern laptop.

---

## Generating Artifacts

1. **Interactive Graph (AI vs SBOM parity)**
   ```bash
   python -m ai_bom.ai_graph_visualization \
       --distribution lognormal  # edit inside file if you prefer pareto
   ```
   Output: `ai_outputs/AI_BOM_graph.html`

2. **Unweighted Top-20 Rankings (lognormal + pareto)**
   ```bash
   python -m ai_bom.ai_unweighted_top20
   ```
   Outputs per distribution:
   - `top20_betweenness_bar.html` + CSV snapshot
   - `top20_pagerank_bar.html` + CSV snapshot

3. **Weighted Monte Carlo (25k sims, α=[0.10,0.15,0.15,0.30,0.30])**
   ```bash
   python -m ai_bom.ai_weighted_monte_carlo
   ```
   Creates for each distribution (lognormal & pareto):
   - Betweenness/PageRank box plots (`montecarlo_*_25000/*.html`)
   - Samples + summary CSVs
   - Ranked top-20 box plots (`*_top20_ranked.html`)

4. **Screenshot Pipeline (8 PNGs)**
   ```bash
   python -m ai_bom.ai_image_creation
   ```
   Converts the deterministic HTML filenames above into PNGs under `ai_images/`
   (unweighted & weighted × betweenness & PageRank).

---

## Expected Outputs

- `ai_outputs/AI_BOM_graph.html` – interactive Pyvis visualization of the AI BOM network.
- `ai_outputs/unweighted/<dist>/top20_*` – bar charts + CSVs for betweenness/PageRank.
- `ai_outputs/weighted/<dist>/montecarlo_*_25000/` – Monte Carlo diagnostics, summary stats,
  and top-20 ranked box plots.
- `ai_images/*.png` – final eight images ready for reports:
  - `unweighted_betweenness_lognormal.png`
  - `unweighted_betweenness_pareto.png`
  - `unweighted_pagerank_lognormal.png`
  - `unweighted_pagerank_pareto.png`
  - `weighted_betweenness_lognormal.png`
  - `weighted_betweenness_pareto.png`
  - `weighted_pagerank_lognormal.png`
  - `weighted_pagerank_pareto.png`

---

## Troubleshooting

- **Playwright import error** → run `pip install playwright` and `playwright install chromium`.
- **HTML not found during screenshot step** → ensure the unweighted/weighted scripts were
  run first; the screenshot script expects fixed filenames.
- **Long Monte Carlo runtimes** → lower `n_simulations` inside `ai_weighted_monte_carlo.py`
  or run one distribution at a time by editing the `for dist in (...)` loop.

---

## Next Steps

- Add CLI arguments (e.g., `--distribution`, `--n-simulations`) for finer-grained control.
- Wire the outputs into downstream reporting notebooks or dashboards.

Questions? Ping email yaphet.lemiesa@gmail.com
