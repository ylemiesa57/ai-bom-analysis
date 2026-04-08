#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Common runtime knobs (override via env vars).
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODE="${MODE:-software_like}"
SEED="${SEED:-42}"
N_SIMS="${N_SIMS:-500}"
RUN_WEIGHTED="${RUN_WEIGHTED:-1}"
LOGNORMAL_MEAN="${LOGNORMAL_MEAN:-1.0}"
LOGNORMAL_SIGMA="${LOGNORMAL_SIGMA:-0.7}"
PARETO_ALPHA="${PARETO_ALPHA:-1.5}"
PARETO_SCALE="${PARETO_SCALE:-1.0}"
MIN_DEGREE="${MIN_DEGREE:-0}"
MAX_DEGREE="${MAX_DEGREE:-30}"
NODE_SIZES="${NODE_SIZES:-100 500 1000}"

# Snapshot directories to preserve each size-run.
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-${SCRIPT_DIR}/run_snapshots}"
mkdir -p "${SNAPSHOT_ROOT}"

# Dedicated multi-size image output (kept separate from existing final_images).
BATCH_FINAL_IMAGES_DIR="${BATCH_FINAL_IMAGES_DIR:-${SCRIPT_DIR}/final_images_multisize}"
rm -rf "${BATCH_FINAL_IMAGES_DIR}"
mkdir -p "${BATCH_FINAL_IMAGES_DIR}"

echo "[AI BOM] Multi-size setup"
echo "  node_sizes=${NODE_SIZES}"
echo "  n_sims=${N_SIMS}"
echo "  mode=${MODE}, seed=${SEED}"
echo "  pareto(alpha=${PARETO_ALPHA}, scale=${PARETO_SCALE})"
echo "  batch_final_images_dir=${BATCH_FINAL_IMAGES_DIR}"

for N_NODES in ${NODE_SIZES}; do
  echo ""
  echo "[AI BOM] ===== Run start: n_nodes=${N_NODES} ====="

  MODE="${MODE}" \
  N_NODES="${N_NODES}" \
  SEED="${SEED}" \
  N_SIMS="${N_SIMS}" \
  RUN_WEIGHTED="${RUN_WEIGHTED}" \
  LOGNORMAL_MEAN="${LOGNORMAL_MEAN}" \
  LOGNORMAL_SIGMA="${LOGNORMAL_SIGMA}" \
  PARETO_ALPHA="${PARETO_ALPHA}" \
  PARETO_SCALE="${PARETO_SCALE}" \
  MIN_DEGREE="${MIN_DEGREE}" \
  MAX_DEGREE="${MAX_DEGREE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  ./run_ai_pipeline_software_like.sh

  SNAPSHOT_DIR="${SNAPSHOT_ROOT}/n${N_NODES}_sims${N_SIMS}"
  mkdir -p "${SNAPSHOT_DIR}/unweighted/lognormal" "${SNAPSHOT_DIR}/unweighted/pareto"
  mkdir -p "${SNAPSHOT_DIR}/weighted/lognormal" "${SNAPSHOT_DIR}/weighted/pareto"
  mkdir -p "${SNAPSHOT_DIR}/final_images"

  # Preserve deterministic unweighted chart HTML outputs.
  cp "ai_outputs/unweighted/lognormal/top20_betweenness_bar.html" "${SNAPSHOT_DIR}/unweighted/lognormal/" || true
  cp "ai_outputs/unweighted/lognormal/top20_pagerank_bar.html" "${SNAPSHOT_DIR}/unweighted/lognormal/" || true
  cp "ai_outputs/unweighted/pareto/top20_betweenness_bar.html" "${SNAPSHOT_DIR}/unweighted/pareto/" || true
  cp "ai_outputs/unweighted/pareto/top20_pagerank_bar.html" "${SNAPSHOT_DIR}/unweighted/pareto/" || true

  # Preserve weighted outputs for the configured simulation count.
  cp -R "ai_outputs/weighted/lognormal/montecarlo_bc_${N_SIMS}" "${SNAPSHOT_DIR}/weighted/lognormal/" || true
  cp -R "ai_outputs/weighted/lognormal/montecarlo_pr_${N_SIMS}" "${SNAPSHOT_DIR}/weighted/lognormal/" || true
  cp -R "ai_outputs/weighted/pareto/montecarlo_bc_${N_SIMS}" "${SNAPSHOT_DIR}/weighted/pareto/" || true
  cp -R "ai_outputs/weighted/pareto/montecarlo_pr_${N_SIMS}" "${SNAPSHOT_DIR}/weighted/pareto/" || true

  # Preserve exported PNGs.
  shopt -s nullglob
  for png in final_images/*.png; do
    cp "${png}" "${SNAPSHOT_DIR}/final_images/"
  done
  shopt -u nullglob

  # Add this batch's images into a clean, dedicated folder (no mixing with old final_images).
  mkdir -p "${BATCH_FINAL_IMAGES_DIR}/n${N_NODES}_sims${N_SIMS}"
  shopt -s nullglob
  for png in final_images/*.png; do
    cp "${png}" "${BATCH_FINAL_IMAGES_DIR}/n${N_NODES}_sims${N_SIMS}/"
  done
  shopt -u nullglob

  echo "[AI BOM] Snapshot saved -> ${SNAPSHOT_DIR}"
  echo "[AI BOM] Batch images added -> ${BATCH_FINAL_IMAGES_DIR}/n${N_NODES}_sims${N_SIMS}"
done

echo ""
echo "[AI BOM] All multi-size runs complete."
echo "[AI BOM] Combined multi-size images -> ${BATCH_FINAL_IMAGES_DIR}"

echo "[AI BOM] Creating 2x8 node-size comparison grids..."
"${PYTHON_BIN}" -m ai_bom.create_multisize_2x8_grid \
  --batch-root "${BATCH_FINAL_IMAGES_DIR}" \
  --n-sims "${N_SIMS}" \
  --output-dir "${BATCH_FINAL_IMAGES_DIR}/grids_2x8"
