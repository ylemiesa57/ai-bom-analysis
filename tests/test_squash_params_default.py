"""Regression test for the run_ai_weighted_analysis() default squash_params bug.

Before this fix, run_ai_weighted_analysis() defaulted squash_params to
{"beta": 0.6} whenever the caller left it unset, regardless of which
squash function was selected. That hardcoded dict only matches
squash_tanh's own parameter name ("beta"); calling with squash="capped"
(parameter name "gamma") raised a TypeError, and squash="logistic" ran
but silently used beta=0.6 instead of its own declared default of
beta=1.0. This only ever went unnoticed because every call site in this
repo uses the default squash="tanh".

These tests exercise the actual defaulting logic in
run_ai_weighted_analysis() (rather than re-implementing it) by monkey-
patching montecarlo_on_graph() to capture the squash_params it receives,
so a regression in the fix would be caught here even if the surrounding
function is refactored.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_bom import ai_weighted_monte_carlo as awmc


class TestSquashParamsDefault(unittest.TestCase):
    def _captured_squash_params(self, squash):
        """Run run_ai_weighted_analysis() with a tiny graph and capture the
        squash_params it actually passes down, without running a real
        Monte Carlo simulation (mocks out generate_ai_bom_graph and
        montecarlo_on_graph, which are the only side-effecting calls)."""
        captured = {}

        def fake_montecarlo_on_graph(G, n_simulations, squash, squash_params, **kwargs):
            captured["squash"] = squash
            captured["squash_params"] = squash_params
            # Still exercise the real squash function with the params
            # run_ai_weighted_analysis() decided on, so a bad default is
            # caught here even if montecarlo_on_graph's own signature
            # changes shape.
            awmc.SQUASH_FUNCS[squash](0.7, **squash_params)
            return {}, {}

        with mock.patch.object(awmc, "generate_ai_bom_graph", return_value=mock.Mock(nodes=lambda: [])), \
             mock.patch.object(awmc, "montecarlo_on_graph", side_effect=fake_montecarlo_on_graph), \
             mock.patch.object(awmc, "plot_interactive_boxplot", return_value=mock.Mock()), \
             mock.patch.object(awmc, "save_top20_ranked_plot", return_value=None):
            awmc.run_ai_weighted_analysis(
                distribution="lognormal", n_nodes=5, n_simulations=1, squash=squash,
            )
        return captured["squash_params"]

    def test_tanh_default_unaffected(self):
        params = self._captured_squash_params("tanh")
        # tanh's own declared default (beta=0.6) must still be what's used.
        self.assertEqual(awmc.squash_tanh(0.7), awmc.squash_tanh(0.7, **params))

    def test_capped_no_longer_crashes(self):
        # Previously: TypeError, unexpected keyword argument 'beta'.
        params = self._captured_squash_params("capped")
        result = awmc.squash_capped_linear(0.7, **params)
        self.assertEqual(result, awmc.squash_capped_linear(0.7))  # matches its own default (gamma=0.3)

    def test_logistic_uses_its_own_default_not_tanhs(self):
        params = self._captured_squash_params("logistic")
        # Previously silently got beta=0.6 (tanh's default) instead of
        # logistic's own declared default of beta=1.0.
        self.assertNotIn("beta", params)
        self.assertEqual(awmc.squash_logistic(0.7), awmc.squash_logistic(0.7, **params))


if __name__ == "__main__":
    unittest.main()
