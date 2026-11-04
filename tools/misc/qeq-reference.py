#!/usr/bin/env python3
"""Independent reference for the 4G-HDNNP charge equilibration (test values).

Written from the published equations of the 4G-HDNNP (Ko et al., Nat. Commun.
12, 398 (2021), Eqs. 2-4), not from the n2p2 code, in atomic units:

    E_elec = sum_{i<j} Q_i Q_j erf(r_ij / (sqrt(2) gamma_ij)) / r_ij
             + sum_i Q_i^2 / (2 sigma_i sqrt(pi)),  gamma_ij = sqrt(sigma_i^2 + sigma_j^2)
    E_Qeq  = E_elec + sum_i (chi_i Q_i + J_i Q_i^2 / 2),  sum_i Q_i = Q_tot

with periodic images summed by Ewald (splitting Gaussian of width eta):

    real   1/2 sum'_{i,j,n} Q_i Q_j [erfc(r / (sqrt(2) eta)) - erfc(r / (sqrt(2) gamma_ij))] / r
    recip  (2 pi / V) sum_{k != 0} exp(-eta^2 k^2 / 2) / k^2 |sum_j Q_j exp(i k.r_j)|^2
    self   sum_i Q_i^2 [1 / (2 sqrt(pi) sigma_i) - 1 / (sqrt(2 pi) eta)]

The real-space sum runs over all images within rcut, so rcut may exceed half
the box. The structure is a disordered rock-salt NaCl cell built from a fixed
formula, so the C++ test can rebuild it without stored coordinates.

Prints the charges and E_elec for two Ewald splittings (their difference shows
the convergence) and the values for test/cpp/test_StructureElectrostatics.cpp.
n2p2 reports E_elec without the Gaussian self-energy sum_i Q_i^2 / (2 sigma_i
sqrt(pi)); both forms are printed.

Usage: qeq-reference.py
"""
import numpy as np
from scipy.special import erfc

# Element parameters (bohr, Ha, e): Na, Cl.
SIGMA = {"Na": 1.6, "Cl": 2.0}
CHI = {"Na": -0.35, "Cl": 0.35}
J = {"Na": 0.40, "Cl": 0.40}
NCELL = 2          # rock-salt conventional cells per direction (8 atoms each)
A = 11.9           # lattice constant (bohr)


def structure():
    """Disordered rock salt: lattice sites shifted by 0.3 sin(...) bohr."""
    L = NCELL * A
    X, T = [], []
    n = 2 * NCELL
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = len(X)
                d = [0.3 * np.sin(1.3 * idx + 0.7 * c + 0.1) for c in range(3)]
                X.append([i * A / 2 + d[0], j * A / 2 + d[1], k * A / 2 + d[2]])
                T.append("Na" if (i + j + k) % 2 == 0 else "Cl")
    return np.array(X), T, L


def coulomb_matrix(X, s, L, eta, rcut, kcut):
    """B with E_elec = Q^T B Q / 2 (Gaussian self-energy included)."""
    N = len(X)
    g = np.sqrt(2.0 * (s[:, None] ** 2 + s[None, :] ** 2))   # sqrt(2) gamma_ij
    se = np.sqrt(2.0) * eta
    B = np.zeros((N, N))
    nmax = int(np.ceil(rcut / L)) + 1
    for a in range(-nmax, nmax + 1):
        for b in range(-nmax, nmax + 1):
            for c in range(-nmax, nmax + 1):
                d = X[:, None, :] - X[None, :, :] + np.array([a, b, c]) * L
                r = np.sqrt((d ** 2).sum(-1))
                if (a, b, c) == (0, 0, 0):
                    np.fill_diagonal(r, np.inf)
                m = r < rcut
                with np.errstate(divide="ignore", invalid="ignore"):
                    B += np.where(m, (erfc(r / se) - erfc(r / g)) / r, 0.0)
    kn = int(np.ceil(kcut * L / (2.0 * np.pi)))
    G = np.array([(a, b, c) for a in range(-kn, kn + 1) for b in range(-kn, kn + 1)
                  for c in range(-kn, kn + 1) if (a, b, c) != (0, 0, 0)], float)
    k = 2.0 * np.pi * G / L
    k2 = (k ** 2).sum(1)
    k, k2 = k[k2 <= kcut ** 2], k2[k2 <= kcut ** 2]
    w = 4.0 * np.pi / L ** 3 * np.exp(-0.5 * eta ** 2 * k2) / k2
    ph = X @ k.T
    B += (np.cos(ph) * w) @ np.cos(ph).T + (np.sin(ph) * w) @ np.sin(ph).T
    B += np.diag(1.0 / (np.sqrt(np.pi) * s) - 2.0 / (np.sqrt(2.0 * np.pi) * eta))
    return B


def qeq(B, chi, hard, qtot=0.0):
    N = len(chi)
    M = np.zeros((N + 1, N + 1))
    M[:N, :N] = B + np.diag(hard)
    M[:N, N] = 1.0
    M[N, :N] = 1.0
    sol = np.linalg.solve(M, np.append(-chi, qtot))
    return sol[:N]


def main():
    X, T, L = structure()
    s = np.array([SIGMA[t] for t in T])
    chi = np.array([CHI[t] for t in T])
    hard = np.array([J[t] for t in T])
    res = []
    for eta, rcut, kcut in ((3.0, 32.0, 3.4), (3.6, 38.0, 2.9)):
        B = coulomb_matrix(X, s, L, eta, rcut, kcut)
        Q = qeq(B, chi, hard)
        E = 0.5 * Q @ B @ Q
        res.append((Q, E))
        print("eta %.2f rcut %.1f kcut %.2f: E_elec %.15e Ha, sum Q %.1e"
              % (eta, rcut, kcut, E, Q.sum()))
    (Q, E), (Q2, E2) = res
    print("difference between splittings: max |dQ| %.1e e, |dE| %.1e Ha"
          % (np.abs(Q - Q2).max(), abs(E - E2)))
    Eself = (Q ** 2 / (2.0 * np.sqrt(np.pi) * s)).sum()
    print("box %.4f bohr, %d atoms" % (L, len(X)))
    print("E_elec (Ko et al. Eq. 3)            %.15e" % E)
    print("E_elec - Gaussian self-energy (n2p2) %.15e" % (E - Eself))
    print("charges:")
    print(",\n".join("    %.15e" % q for q in Q))


if __name__ == "__main__":
    main()
