"""
Build 2x8 comparison grids from multi-size AI run outputs.

Each grid contains 16 plots total:
- Row 1: 8 standard plots for size A
- Row 2: 8 standard plots for size B

Input folders are expected under final_images_multisize/, e.g.:
  final_images_multisize/n100_sims500/
  final_images_multisize/n500_sims500/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


EXPECTED_PLOT_ORDER = [
    "unweighted_bc_lognormal.png",
    "unweighted_prc_lognormal.png",
    "unweighted_bc_pareto.png",
    "unweighted_prc_pareto.png",
    "weighted_bc_lognormal.png",
    "weighted_prc_lognormal.png",
    "weighted_bc_pareto.png",
    "weighted_prc_pareto.png",
]

TILE_WIDTH = 700
TILE_HEIGHT = 500
LABEL_BAND_HEIGHT = 50


def _parse_size_dir_name(name: str) -> Tuple[int, int] | None:
    m = re.fullmatch(r"n(\d+)_sims(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _find_size_dirs(batch_root: Path, n_sims: int) -> List[Tuple[int, Path]]:
    candidates: List[Tuple[int, Path]] = []
    for child in batch_root.iterdir():
        if not child.is_dir():
            continue
        parsed = _parse_size_dir_name(child.name)
        if parsed is None:
            continue
        n_nodes, sims = parsed
        if sims == n_sims:
            candidates.append((n_nodes, child))
    return sorted(candidates, key=lambda x: x[0])


def _pair_consecutive(items: Iterable[Tuple[int, Path]]) -> List[Tuple[Tuple[int, Path], Tuple[int, Path]]]:
    items_list = list(items)
    return list(zip(items_list, items_list[1:]))


def _load_plot_or_placeholder(path: Path, fallback_label: str) -> Image.Image:
    if path.exists():
        img = Image.open(path).convert("RGB")
        return img.resize((TILE_WIDTH, TILE_HEIGHT), Image.Resampling.LANCZOS)

    placeholder = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), color="#f0f0f0")
    draw = ImageDraw.Draw(placeholder)
    text = f"Missing:\n{fallback_label}"
    draw.multiline_text((20, 20), text, fill="#555555", spacing=8)
    return placeholder


def _render_row_label(text: str, width: int) -> Image.Image:
    band = Image.new("RGB", (width, LABEL_BAND_HEIGHT), color="#1f2937")
    draw = ImageDraw.Draw(band)
    font = ImageFont.load_default()
    draw.text((12, 16), text, fill="#ffffff", font=font)
    return band


def _build_pair_grid(
    size_a: int,
    dir_a: Path,
    size_b: int,
    dir_b: Path,
    out_path: Path,
) -> None:
    cols = 8
    row_width = cols * TILE_WIDTH
    row_height = TILE_HEIGHT
    total_height = (LABEL_BAND_HEIGHT + row_height) * 2

    canvas = Image.new("RGB", (row_width, total_height), color="white")

    label_a = _render_row_label(f"n={size_a}", row_width)
    label_b = _render_row_label(f"n={size_b}", row_width)
    canvas.paste(label_a, (0, 0))
    canvas.paste(label_b, (0, LABEL_BAND_HEIGHT + row_height))

    for col, filename in enumerate(EXPECTED_PLOT_ORDER):
        img_a = _load_plot_or_placeholder(dir_a / filename, filename)
        img_b = _load_plot_or_placeholder(dir_b / filename, filename)

        x = col * TILE_WIDTH
        y_a = LABEL_BAND_HEIGHT
        y_b = LABEL_BAND_HEIGHT * 2 + row_height

        canvas.paste(img_a, (x, y_a))
        canvas.paste(img_b, (x, y_b))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[AI BOM] 2x8 grid saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 2x8 multi-size comparison grids.")
    parser.add_argument("--batch-root", default="final_images_multisize", help="Root dir containing n*_sims* folders")
    parser.add_argument("--n-sims", type=int, required=True, help="Simulation count used in folder names")
    parser.add_argument("--output-dir", default="final_images_multisize/grids_2x8", help="Where to save 2x8 grids")
    args = parser.parse_args()

    batch_root = Path(args.batch_root)
    output_dir = Path(args.output_dir)
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch root not found: {batch_root}")

    size_dirs = _find_size_dirs(batch_root, args.n_sims)
    if len(size_dirs) < 2:
        print("[AI BOM] Need at least two node-size folders to build 2x8 grids.")
        return

    for (size_a, dir_a), (size_b, dir_b) in _pair_consecutive(size_dirs):
        out_name = f"grid_2x8_n{size_a}_vs_n{size_b}_sims{args.n_sims}.png"
        _build_pair_grid(size_a, dir_a, size_b, dir_b, output_dir / out_name)


if __name__ == "__main__":
    main()

