"""Regression test for create_plot_grid.py's OUTPUT_DIR typo.

The OUTPUT_DIR was incorrectly set to "ai_image1" instead of "ai_images",
which is the directory referenced in the README and used by the pipeline.
This caused grid output to go to the wrong location.

See: https://github.com/ylemiesa57/ai-bom-analysis/issues/...
(The typo was introduced in commit 8d6eb80 and persisted through multiple
subsequent commits, only caught because the existing ai_images/ directory
already had the 8 PNG files in it from prior manual runs.)
"""
import sys
import unittest
from pathlib import Path

# Add ai_bom to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestCreatePlotGridOutputDir(unittest.TestCase):
    def test_output_dir_is_ai_images_not_ai_image1(self):
        """Verify OUTPUT_DIR points to ai_images, not the typo ai_image1."""
        # Import here to avoid top-level import of playwright when we just
        # want to check the constant without actually running code.
        try:
            from ai_bom.create_plot_grid import OUTPUT_DIR
            # The bug: OUTPUT_DIR was set to Path("ai_image1")
            # The fix: OUTPUT_DIR should be Path("ai_images")
            self.assertEqual(OUTPUT_DIR, Path("ai_images"),
                           msg=f"OUTPUT_DIR is '{OUTPUT_DIR}' but should be 'ai_images'")
        except ImportError as e:
            # playwright might not be installed in CI; that's okay for this test
            if "playwright" in str(e):
                self.skipTest("playwright not installed (expected in CI)")
            raise


if __name__ == "__main__":
    unittest.main()
