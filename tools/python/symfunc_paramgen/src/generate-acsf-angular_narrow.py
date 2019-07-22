#!/usr/bin/env python

import numpy as np
import sys

# Settings
elements = ["S", "Cu"]
mode     = "center"
r_0      = 1.5
r_c      = 6.0
r_N      = r_c - 0.5
N        = 3
zetas    = [1.0, 6.0]

grid = np.linspace(r_0, r_N, N)
dr = (r_N - r_0) / (N - 1)

sys.stdout.write("# Generating narrow angular symmetry function set:\n")
sys.stdout.write("# mode  = {0:9s}\n".format(mode))
sys.stdout.write("# r_0   = {0:9.3E}\n".format(r_0))
sys.stdout.write("# r_c   = {0:9.3E}\n".format(r_c))
sys.stdout.write("# r_N   = {0:9.3E}\n".format(r_N))
sys.stdout.write("# N     = {0:9d}\n".format(N))
sys.stdout.write("# grid  = " + " ".join(str(r) for r in grid) + "\n")
sys.stdout.write("# zetas = " + " ".join(str(z) for z in zetas) + "\n")

if mode == "center":
    eta_grid = 1.0 / (2.0 * grid**2)
    rs_grid = [0.0] * N
elif mode == "shift":
    eta_grid = [1.0 / (2.0 * dr * dr)] * N
    rs_grid = grid

for e in elements:
    sys.stdout.write("# Narrow angular symmetry functions for element {0:2s}\n".format(e))
    for e1 in elements:
        elements_reduced = elements[elements.index(e1):]
        for e2 in elements_reduced:
            for (eta, rs) in zip(eta_grid, rs_grid):
                for zeta in zetas:
                    for lambd in [-1.0, 1.0]:
                        sys.stdout.write("symfunction_short {0:2s} 3 {1:2s} {2:2s} {3:9.3E} {4:2.0f} {5:9.3E} {6:9.3E} {7:9.3E}\n".format(e, e1, e2, eta, lambd, zeta, r_c, rs))
            sys.stdout.write("\n")
