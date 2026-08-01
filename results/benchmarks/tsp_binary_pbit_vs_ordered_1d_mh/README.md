# Benchmark Result Bundle

This directory packages the published comparison outputs for TSP, Max-Cut,
and Portfolio.

## Comparison scope

- **TSP:** binary p-bit + one-hot Ising versus direct-permutation encoding +
  ordered 1D empirical-transition MH.
- **Max-Cut and Portfolio:** native QUBO/Ising encoding is fixed; Gray binary
  p-bit and ordered 1D P-dit device-state interpretations are compared. The
  TSP-only permutation representation is not used for these two problems.

## TSP

- [Method comparison](tsp_method_comparison.png)
- [Best 12-city routes](tsp_best_routes_comparison.png)
- [Per-run results](tsp_comparison_runs.csv)
- [Summary CSV](tsp_comparison_summary.csv)
- [Strict JSON summary](tsp_comparison_summary.json)

Key results: ordered MH reaches the exact 8-city reference in 24/24 runs and
the 12-city heuristic reference in 13/24 runs. Every ordered route is a valid
permutation.

## 10-node Max-Cut

- [Metric comparison](maxcut_method_comparison.png)
- [Success curves](maxcut_success_curves.png)
- [Per-run results](maxcut_comparison_runs.csv)
- [Summary CSV](maxcut_comparison_summary.csv)
- [Paired comparison](maxcut_paired_comparison.csv)
- [Success-curve data](maxcut_success_curves.csv)

Both methods reach the exact reference in 24/24 runs. The instance is in a
ceiling regime and does not distinguish final solution quality.

## 8-asset Portfolio

- [Metric comparison](portfolio_method_comparison.png)
- [Success curves](portfolio_success_curves.png)
- [Per-run results](portfolio_comparison_runs.csv)
- [Summary CSV](portfolio_comparison_summary.csv)
- [Paired comparison](portfolio_paired_comparison.csv)
- [Success-curve data](portfolio_success_curves.csv)

Ordered 1D reaches the exact feasible reference in 18/24 runs (75.0%), versus
16/24 runs (66.7%) for Gray binary. Its successful runs have a later median
first hit (831 versus 405 updates), so the result is higher final coverage,
not uniformly faster convergence.

## Combined metadata

[Max-Cut and Portfolio strict JSON summary](maxcut_portfolio_comparison_summary.json)
records the comparison design, source data, run-file references, method
summaries, and paired statistics.

Regenerate the Max-Cut and Portfolio views with:

```bash
python tools/results/export_maxcut_portfolio_views.py
```
