#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Runtime knobs (override via env vars).
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODE="${MODE:-software_like}"
N_NODES="${N_NODES:-500}"
SEED="${SEED:-42}"
N_SIMS="${N_SIMS:-25000}"
RUN_WEIGHTED="${RUN_WEIGHTED:-1}"
LOGNORMAL_MEAN="${LOGNORMAL_MEAN:-1.0}"
LOGNORMAL_SIGMA="${LOGNORMAL_SIGMA:-0.7}"
PARETO_ALPHA="${PARETO_ALPHA:-1.5}"
PARETO_SCALE="${PARETO_SCALE:-1.0}"
MIN_DEGREE="${MIN_DEGREE:-0}"
MAX_DEGREE="${MAX_DEGREE:-30}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-0}"

export MODE N_NODES SEED N_SIMS RUN_WEIGHTED LOGNORMAL_MEAN LOGNORMAL_SIGMA PARETO_ALPHA PARETO_SCALE MIN_DEGREE MAX_DEGREE AUTO_INSTALL_DEPS

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

echo "[AI BOM] Starting pipeline with:"
echo "  python=${PYTHON_BIN}"
echo "  mode=${MODE}, n_nodes=${N_NODES}, seed=${SEED}, n_sims=${N_SIMS}, run_weighted=${RUN_WEIGHTED}"
echo "  lognormal(mean=${LOGNORMAL_MEAN}, sigma=${LOGNORMAL_SIGMA})"
echo "  pareto(alpha=${PARETO_ALPHA}, scale=${PARETO_SCALE}), degree_clip=[${MIN_DEGREE}, ${MAX_DEGREE}]"

echo "[AI BOM] Checking Python dependencies..."
missing_pkgs="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
required = ["networkx", "numpy", "pandas", "plotly", "scipy", "pyvis", "playwright", "PIL"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(" ".join(missing))
PY
)"

if [[ -n "${missing_pkgs}" ]]; then
  echo "[AI BOM] Missing packages: ${missing_pkgs}"
  if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
    echo "[AI BOM] Installing missing dependencies..."
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install networkx numpy scipy pandas plotly pyvis pillow playwright
    "${PYTHON_BIN}" -m playwright install chromium
  else
    echo "[AI BOM] Aborting. Set AUTO_INSTALL_DEPS=1 to auto-install dependencies."
    exit 1
  fi
fi

start_ts="$(date +%s)"

echo "[AI BOM] Running unweighted analyses..."
"${PYTHON_BIN}" - <<'PY'
import os
from ai_bom.ai_unweighted_top20 import run_unweighted_ai_analysis

kwargs = dict(
    n_nodes=int(os.environ["N_NODES"]),
    seed=int(os.environ["SEED"]),
    generation_mode=os.environ["MODE"],
    lognormal_mean=float(os.environ["LOGNORMAL_MEAN"]),
    lognormal_sigma=float(os.environ["LOGNORMAL_SIGMA"]),
    pareto_alpha=float(os.environ["PARETO_ALPHA"]),
    pareto_scale=float(os.environ["PARETO_SCALE"]),
    min_degree=int(os.environ["MIN_DEGREE"]),
    max_degree=int(os.environ["MAX_DEGREE"]),
)

for dist in ("lognormal", "pareto"):
    print(f"[AI BOM] Unweighted -> {dist}")
    run_unweighted_ai_analysis(distribution=dist, **kwargs)
PY

if [[ "${RUN_WEIGHTED}" == "1" ]]; then
  echo "[AI BOM] Running weighted Monte Carlo analyses..."
  "${PYTHON_BIN}" - <<'PY'
import os
from ai_bom.ai_weighted_monte_carlo import run_ai_weighted_analysis

kwargs = dict(
    n_nodes=int(os.environ["N_NODES"]),
    n_simulations=int(os.environ["N_SIMS"]),
    seed=int(os.environ["SEED"]),
    generation_mode=os.environ["MODE"],
    lognormal_mean=float(os.environ["LOGNORMAL_MEAN"]),
    lognormal_sigma=float(os.environ["LOGNORMAL_SIGMA"]),
    pareto_alpha=float(os.environ["PARETO_ALPHA"]),
    pareto_scale=float(os.environ["PARETO_SCALE"]),
    min_degree=int(os.environ["MIN_DEGREE"]),
    max_degree=int(os.environ["MAX_DEGREE"]),
)

for dist in ("lognormal", "pareto"):
    print(f"[AI BOM] Weighted -> {dist}")
    run_ai_weighted_analysis(distribution=dist, **kwargs)
PY
else
  echo "[AI BOM] Skipping weighted Monte Carlo (RUN_WEIGHTED=${RUN_WEIGHTED})."
fi

echo "[AI BOM] Exporting plot PNGs to final_images..."
"${PYTHON_BIN}" -m ai_bom.create_plot_grid

end_ts="$(date +%s)"
elapsed="$((end_ts - start_ts))"
mins="$((elapsed / 60))"
secs="$((elapsed % 60))"
echo "[AI BOM] Pipeline complete in ${mins}m ${secs}s."
echo "[AI BOM] Updated images: ${SCRIPT_DIR}/final_images/"
