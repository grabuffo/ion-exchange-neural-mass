# In vitro recordings (Fig. 4c-d)

Fig. 4c-d shows simultaneous recordings of the local field potential
(AC-coupled) and of the extracellular potassium concentration in mouse
hippocampal slices (seven recordings from four mice, see Methods).

The traces are distributed here as plain csv files, one per recording:

    time_s, lfp, k_ext_mM

`scripts/fig4_slow_potassium.py` plots every `*.csv` found in this folder
(panels c-d); if the folder is empty those panels are skipped.
