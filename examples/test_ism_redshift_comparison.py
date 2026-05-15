import numpy as np
import matplotlib.pyplot as plt

from atomic_data import parse_atom_file
from solver import build_Q_matrix, solve_populations


atom = parse_atom_file("CI.dat")

T_gas = 100.0
nH_grid = np.logspace(-4, 6, 100)

cases = {
    "MW ISM, z=0": {
        "z": 0.0,
        "uv_field": "draine",
        "uv_scale": 1.0,
    },
    "galaxy ISM, z=0.01": {
        "z": 0.01,
        "uv_field": "draine",
        "uv_scale": 1.0,
    },
    "galaxy ISM, z=2": {
        "z": 2.0,
        "uv_field": "draine",
        "uv_scale": 1.0,
    },
}


def make_densities(nH):
    return {
        "electron": 1e-4 * nH,
        "proton": 0.0,
        "H0": nH,
        "helium": 0.1 * nH,
        "p-H2": 0.25 * nH,
        "o-H2": 0.75 * nH,
    }


results_21 = {}
results_31 = {}

for label, cfg in cases.items():
    z = cfg["z"]
    T_rad = 2.728 * (1.0 + z)

    r21 = []
    r31 = []

    for nH in nH_grid:
        Q = build_Q_matrix(
            atom,
            make_densities(nH),
            temperature=T_gas,
            use_detailed_balance=True,
            include_radiation=True,
            T_rad=T_rad,
            include_fluorescence=True,
            uv_field=cfg["uv_field"],
            uv_scale=cfg["uv_scale"],
        )

        pop = solve_populations(Q)

        r21.append(pop[1] / pop[0])
        r31.append(pop[2] / pop[0])

    results_21[label] = np.array(r21)
    results_31[label] = np.array(r31)


plt.figure(figsize=(8, 5))
for label in cases:
    plt.loglog(nH_grid, results_21[label], label=label)

plt.xlabel(r"$n_{\rm H}$ [cm$^{-3}$]")
plt.ylabel(r"$n_2/n_1$")
plt.title(r"C I: redshift comparison, $n_2/n_1$")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
for label in cases:
    plt.loglog(nH_grid, results_31[label], label=label)

plt.xlabel(r"$n_{\rm H}$ [cm$^{-3}$]")
plt.ylabel(r"$n_3/n_1$")
plt.title(r"C I: redshift comparison, $n_3/n_1$")
plt.legend()
plt.tight_layout()
plt.show()


for nH_test in [1e-2, 1.0, 100.0]:
    idx = np.argmin(np.abs(nH_grid - nH_test))

    print(f"\nAt nH ~ {nH_grid[idx]:.3g} cm^-3")
    for label in cases:
        print(
            label,
            "n2/n1 =", results_21[label][idx],
            "n3/n1 =", results_31[label][idx],
        )
