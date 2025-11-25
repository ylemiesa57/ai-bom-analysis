"""
Convert the AI BOM HTML outputs into PNGs (mirrors image_creation.py used for the
software SBOM workflow). Requires `playwright install` + chromium.
"""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright  # type: ignore[import]


def html_to_image(html_path: str, output_path: str, width: int = 1920, height: int = 1080, scale: int = 4):
    abs_html_path = "file://" + os.path.abspath(html_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=scale)
        page.set_viewport_size({"width": width, "height": height})
        page.goto(abs_html_path, wait_until="networkidle")
        page.screenshot(path=output_path, full_page=True)
        browser.close()


HTML_IMAGE_PAIRS = [
    # Unweighted
    ("ai_outputs/unweighted/lognormal/top20_betweenness_bar.html", "ai_images/unweighted_betweenness_lognormal.png"),
    ("ai_outputs/unweighted/lognormal/top20_pagerank_bar.html", "ai_images/unweighted_pagerank_lognormal.png"),
    ("ai_outputs/unweighted/pareto/top20_betweenness_bar.html", "ai_images/unweighted_betweenness_pareto.png"),
    ("ai_outputs/unweighted/pareto/top20_pagerank_bar.html", "ai_images/unweighted_pagerank_pareto.png"),
    # Weighted
    ("ai_outputs/weighted/lognormal/montecarlo_bc_5000/betweenness_top20_ranked.html", "ai_images/weighted_betweenness_lognormal.png"),
    ("ai_outputs/weighted/lognormal/montecarlo_pr_5000/pagerank_top20_ranked.html", "ai_images/weighted_pagerank_lognormal.png"),
    ("ai_outputs/weighted/pareto/montecarlo_bc_5000/betweenness_top20_ranked.html", "ai_images/weighted_betweenness_pareto.png"),
    ("ai_outputs/weighted/pareto/montecarlo_pr_5000/pagerank_top20_ranked.html", "ai_images/weighted_pagerank_pareto.png"),
]


if __name__ == "__main__":
    for html_path, image_path in HTML_IMAGE_PAIRS:
        if not Path(html_path).exists():
            print(f"[skip] {html_path} not found yet")
            continue
        html_to_image(html_path, image_path, scale=4)
        print(f"[saved] {image_path}")

