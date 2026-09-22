# Two-parameter bifurcation diagram (Fig. 5)

Fig. 5 shows the bifurcation diagram of the fast subsystem of the mean-field
model (Eq. 25 with the slow variables Delta[K+]_int and [K+]_g treated as
parameters): saddle-node, saddle-homoclinic, fold-of-limit-cycles and Hopf
curves, with the Bogdanov-Takens and SNL codimension-2 points.

The curves were computed by numerical continuation (not by direct
simulation). The continuation files are collected in this folder; the fast
subsystem itself is `ionmf.mean_field.rhs` with `dDKi = dKg = 0`, i.e. the
first three equations (x, V, n) with DKi and Kg fixed.

**Status:** the continuation files are not yet in this folder; they are being
collected and will be added. Until then they are available on request from
the corresponding author, Giovanni Rabuffo (giovanni.rabuffo@univ-amu.fr).
