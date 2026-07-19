"""
Create a 2x4 grid (8 plots total) of AI BOM plots and save to ai_image folder.
Also saves individual images to final_images folder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple
from glob import glob

from PIL import Image
from playwright.sync_api import sync_playwright  # type: ignore[import]


OUTPUT_DIR = Path("ai_image1")
OUTPUT_DIR.mkdir(exist_ok=True)

FINAL_IMAGES_DIR = Path("final_images")
FINAL_IMAGES_DIR.mkdir(exist_ok=True)


def html_to_image(html_path: str, output_path: str, width: int = 1920, height: int = 1080, scale: int = 4):
    """Convert HTML plot to PNG image."""
    abs_html_path = "file://" + os.path.abspath(html_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=scale)
        page.set_viewport_size({"width": width, "height": height})
        page.goto(abs_html_path, wait_until="networkidle")
        page.screenshot(path=output_path, full_page=True)
        browser.close()


def get_plot_paths() -> List[Tuple[str, str, str, str]]:
    """
    Get all plot paths in order for 2x4 grid (8 plots total).
    Returns list of (html_path_pattern, temp_png_path, label, final_image_name) tuples.

    Layout (columns: Lognormal BC, Lognormal PRC, Pareto BC, Pareto PRC):
      Row 1: Unweighted (top-20 bar charts)
      Row 2: Weighted (top-20 ranked box plots)
    """
    plots: List[Tuple[str, str, str, str]] = []

    # Row 1: Unweighted plots (4)
    plots.append((
        "ai_outputs/unweighted/lognormal/top20_betweenness_bar.html",
        "ai_image/temp_unweighted_bc_lognormal.png",
        "Unweighted BC (Lognormal)",
        "unweighted_bc_lognormal.png"
    ))
    plots.append((
        "ai_outputs/unweighted/lognormal/top20_pagerank_bar.html",
        "ai_image/temp_unweighted_pr_lognormal.png",
        "Unweighted PRC (Lognormal)",
        "unweighted_prc_lognormal.png"
    ))
    plots.append((
        "ai_outputs/unweighted/pareto/top20_betweenness_bar.html",
        "ai_image/temp_unweighted_bc_pareto.png",
        "Unweighted BC (Pareto)",
        "unweighted_bc_pareto.png"
    ))
    plots.append((
        "ai_outputs/unweighted/pareto/top20_pagerank_bar.html",
        "ai_image/temp_unweighted_pr_pareto.png",
        "Unweighted PRC (Pareto)",
        "unweighted_prc_pareto.png"
    ))
    
    # Row 2: Weighted top20 plots (4)
    plots.append((
        "ai_outputs/weighted/lognormal/montecarlo_bc_*/betweenness_top20_ranked.html",
        "ai_image/temp_weighted_top20_bc_lognormal.png",
        "Weighted BC Top20 (Lognormal)",
        "weighted_bc_lognormal.png"
    ))
    plots.append((
        "ai_outputs/weighted/lognormal/montecarlo_pr_*/pagerank_top20_ranked.html",
        "ai_image/temp_weighted_top20_pr_lognormal.png",
        "Weighted PRC Top20 (Lognormal)",
        "weighted_prc_lognormal.png"
    ))
    plots.append((
        "ai_outputs/weighted/pareto/montecarlo_bc_*/betweenness_top20_ranked.html",
        "ai_image/temp_weighted_top20_bc_pareto.png",
        "Weighted BC Top20 (Pareto)",
        "weighted_bc_pareto.png"
    ))
    plots.append((
        "ai_outputs/weighted/pareto/montecarlo_pr_*/pagerank_top20_ranked.html",
        "ai_image/temp_weighted_top20_pr_pareto.png",
        "Weighted PRC Top20 (Pareto)",
        "weighted_prc_pareto.png"
    ))
    
    return plots


def find_latest_html(pattern: str) -> Path | None:
    """Find the most recent HTML file matching the pattern."""
    if "*" not in pattern:
        path = Path(pattern)
        return path if path.exists() else None

    matching_files = [Path(p) for p in glob(pattern)]
    if not matching_files:
        return None

    return max(matching_files, key=lambda p: p.stat().st_mtime)


def create_grid(plot_paths: List[Tuple[str, str, str, str]], grid_size: Tuple[int, int] = (2, 4)) -> None:
    """Create a 2x4 grid from the plot images and save individual images to final_images folder."""
    rows, cols = grid_size
    
    # Convert HTML to PNG if needed
    temp_images = []
    final_image_names = []
    max_plots = rows * cols
    for html_pattern, temp_png, label, final_name in plot_paths[:max_plots]:
        html_path = find_latest_html(html_pattern)
        
        if html_path is None:
            print(f"[WARNING] Could not find HTML file matching: {html_pattern}")
            # Create a blank placeholder
            Path(temp_png).parent.mkdir(parents=True, exist_ok=True)
            temp_img = Image.new('RGB', (800, 600), color='white')
            temp_img.save(temp_png)
            temp_images.append(temp_png)
            final_image_names.append(final_name)
            continue
        
        print(f"[Converting] {html_path.name} -> {temp_png}")
        html_to_image(str(html_path), temp_png, width=1600, height=900, scale=2)
        temp_images.append(temp_png)
        final_image_names.append(final_name)
        
        # Also save high-quality version to final_images folder
        final_path = FINAL_IMAGES_DIR / final_name
        html_to_image(str(html_path), str(final_path), width=1920, height=1080, scale=4)
        print(f"[Saved] {final_path}")
    
    # Load all images
    images = []
    for img_path in temp_images:
        if Path(img_path).exists():
            img = Image.open(img_path)
            # Resize to consistent size
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            images.append(img)
        else:
            # Create placeholder
            img = Image.new('RGB', (800, 600), color='lightgray')
            images.append(img)
    
    # Create grid
    grid_width = cols * 800
    grid_height = rows * 600
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images into grid
    for idx, img in enumerate(images[:rows * cols]):
        row = idx // cols
        col = idx % cols
        x = col * 800
        y = row * 600
        grid_image.paste(img, (x, y))
    
    # Save grid
    output_path = OUTPUT_DIR / "all_plots_grid.png"
    grid_image.save(output_path)
    print(f"\n[SUCCESS] Created grid image: {output_path}")
    print(f"[SUCCESS] Individual images saved to: {FINAL_IMAGES_DIR}/")


if __name__ == "__main__":
    print("Creating 4x4 grid of all plots...")
    plot_paths = get_plot_paths()
    create_grid(plot_paths)
    print("\nDone!")

