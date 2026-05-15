import numpy as np
import matplotlib.pyplot as plt

from atomic_data import parse_atom_file
from solver import build_Q_matrix, solve_populations


# --------------------------------------------------
# Observed Jenkins & Tripp-like column densities
# --------------------------------------------------

HD = np.array([
    "108", "3827", "15137", "69106", "88115", "93843", "94493",
    "99857A", "103779", "106343", "109399", "116781A", "120086",
    "122879", "124314A", "203374A", "206267A", "208947",
    "209339A", "210839", "224151"
])

logN_CI = np.array([
    14.810, 13.514, 14.513, 14.179, 13.956, 13.969, 14.144,
    14.418, 14.152, 14.210, 14.155, 14.188, 13.076,
    14.309, 14.540, 14.857, 15.160, 14.502,
    14.608, 14.798, 14.501
])

logN_CI1 = np.array([
    14.320, 12.415, 13.862, 13.555, 13.161, 13.542, 13.504,
    13.952, 13.365, 13.638, 13.584, 13.457, 12.472,
    13.682, 13.970, 14.252, 14.616, 13.978,
    14.104, 14.337, 13.898
])

# CI** for HD 120086 is listed as consistent with zero; use NaN
logN_CI2 = np.array([
    13.926, 12.444, 13.219, 13.033, 12.766, 13.112, 12.921,
    13.397, 12.322, 13.059, 13.049, 12.935, np.nan,
    13.042, 13.402, 13.763, 14.131, 13.418,
    13.517, 14.013, 13.437
])


# --------------------------------------------------
# Convert to f1 and f2
# --------------------------------------------------

N0 = 10**logN_CI
N1 = 10**logN_CI1
N2 = 10**logN_CI2

Ntot = N0 + N1 + np.nan_to_num(N2, nan=0.0)

f1_obs = N1 / Ntot
f2_obs = N2 / Ntot

good = np.isfinite(f2_obs)


# --------------------------------------------------
# Model curve at T = 100 K
# --------------------------------------------------

atom = parse_atom_file("CI.dat")

T_gas = 100.0
z = 0.0
T_rad = 2.728 * (1 + z)

nH_grid = np.logspace(-1, 5, 200)

f1_model = []
f2_model = []

for nH in nH_grid:
    densities = {
        "electron": 1e-4 * nH,
        "proton": 0.0,
        "H0": nH,
        "helium": 0.1 * nH,
        "p-H2": 0.25 * nH,
        "o-H2": 0.75 * nH,
    }

    Q = build_Q_matrix(
        atom,
        densities,
        temperature=T_gas,
        use_detailed_balance=True,
        include_radiation=True,
        T_rad=T_rad,
        include_fluorescence=True,
        uv_field="draine",
        uv_scale=1.0,
    )

    pop = solve_populations(Q)

    # CI, CI*, CI** are first three fine-structure levels
    total_3 = pop[0] + pop[1] + pop[2]

    f1_model.append(pop[1] / total_3)
    f2_model.append(pop[2] / total_3)

f1_model = np.array(f1_model)
f2_model = np.array(f2_model)


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(7, 6))

plt.plot(f1_model, f2_model, color="black", lw=2, label=r"Model, $T=100$ K")


plt.scatter(
    f1_obs[good],
    f2_obs[good],
    s=50,
    facecolors="none",
    edgecolors="tab:blue",
    label="Observed sight lines"
)

# Mark a few densities
for nmark in [10, 100, 1000, 10000]:
    idx = np.argmin(np.abs(nH_grid - nmark))

    plt.scatter(
        f1_model[idx],
        f2_model[idx],
        color="black",
        s=35,
        zorder=5
    )

    plt.text(
        f1_model[idx] + 0.01,
        f2_model[idx] + 0.01,
        rf"$n_H={nmark:g}$",
        fontsize=10
    )

plt.xlabel(r"$f_1 = N({\rm C\,I^*})/N({\rm C\,I_{total}})$")
plt.ylabel(r"$f_2 = N({\rm C\,I^{**}})/N({\rm C\,I_{total}})$")

plt.xlim(0, 0.55)
plt.ylim(0, 0.55)

plt.legend()
plt.tight_layout()
plt.show()
