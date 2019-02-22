#!/usr/bin/env python

import numpy as np
import sys

# Settings
elements = ["S", "Cu"]
mode     = "shift"
r_0      = 1.5
r_c      = 6.0
r_N      = r_c - 0.5
N        = 9

grid = np.linspace(r_0, r_N, N)
dr = (r_N - r_0) / (N - 1)

sys.stdout.write("# Generating radial symmetry function set:\n")
sys.stdout.write("# mode  = {0:9s}\n".format(mode))
sys.stdout.write("# r_0   = {0:9.3E}\n".format(r_0))
sys.stdout.write("# r_c   = {0:9.3E}\n".format(r_c))
sys.stdout.write("# r_N   = {0:9.3E}\n".format(r_N))
sys.stdout.write("# N     = {0:9d}\n".format(N))
sys.stdout.write("# grid  = " + " ".join(str(r) for r in grid) + "\n")

if mode == "center":
    eta_grid = 1.0 / (2.0 * grid**2)
    rs_grid = [0.0] * N
elif mode == "shift":
    eta_grid = [1.0 / (2.0 * dr * dr)] * N
    rs_grid = grid

for e in elements:
    sys.stdout.write("# Radial symmetry functions for element {0:2s}\n".format(e))
    for e1 in elements:
        for (eta, rs) in zip(eta_grid, rs_grid):
            sys.stdout.write("symfunction_short {0:2s} 2 {1:2s} {2:9.3E} {3:9.3E} {4:9.3E}\n".format(e, e1, eta, rs, r_c))
        sys.stdout.write("\n")
