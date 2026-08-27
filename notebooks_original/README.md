# Original analysis notebooks (provenance only)

These are the Jupyter notebooks that were used to produce the figures of the
paper, copied **unchanged** from the authors' working directory. They are kept
for provenance: they contain hard-coded paths to the authors' machines and to
a computing cluster, load intermediate pickles that are not distributed, and
depend on an old software stack (Python 3.6, TVB 2.x, `pickle5`). They are
**not** meant to be re-run; use the scripts in `../scripts/` instead, which
re-implement the same models and simulations in a self-contained way.

| notebook | what it produced (final figure numbering) |
|---|---|
| `Fig2-TimeSeries.ipynb` | Fig. 2a-b (single neuron regimes, 3-D trajectory), Fig. 3b-c (population vs mean field, reparametrization) |
| `Suppl.Fig1.ipynb` | Fig. 2c-d (dV/dt curves, parabola fit -> c-, c+, R-, R+) |
| `POPcluster.ipynb`, `POPcluster-Copy1.ipynb`, `Fig3-population-emergent.ipynb` | Brian2 population simulations: Fig. 3a raster/histogram, Fig. 2-figure supplement 1a |
| `MFcluster.ipynb`, `MFcluster-slower.ipynb` | mean-field simulations: Fig. 3b, Fig. 4a |
| `Fig4-parabolas-parameter-explore.ipynb` | Fig. 2-figure supplement 1b (PSD while sweeping c-, c+, R-, R+) |
| `Fig5-BifurcationI.ipynb` | mean-field regime with isolated bursts (Fig. 4e) |
| `Hexagon-HH.ipynb`, `Single_node-HH.ipynb` | Fig. 6 (TVB network of six neural masses; uses `tvb_model/model_HH_ABH.py`) |

The notebook names refer to the figure numbering of the preprint; the
two-parameter bifurcation diagram (Fig. 5 of the final version) was computed
with numerical continuation software and is documented in `../continuation/`.
