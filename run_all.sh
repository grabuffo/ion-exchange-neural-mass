#!/usr/bin/env bash
# Reproduce every simulation-based panel. Population runs (N = 3000) take
# 5-10 min each; the whole script needs ~1-2 h on a laptop. Results are cached
# in cache/ so re-running only re-plots. Add --quick for a fast smoke test.
set -e
cd "$(dirname "$0")/scripts"
python fig2_single_neuron.py "$@"
python fig3_population_vs_meanfield.py "$@"
python fig4_slow_potassium.py "$@"
python fig6_network.py "$@"
python figS1_bimodal_and_psd.py "$@"
python figS2_first_moment_error.py "$@"
