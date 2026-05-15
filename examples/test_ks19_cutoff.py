import numpy as np
import matplotlib.pyplot as plt

from atomic_data import parse_atom_file
from solver import build_Q_matrix, solve_populations
from radiation_fields import build_ks19_interpolator


z = 1.0

# Use the exact filename you have. Yours seems to include a space before 1.0:
ks19_file = f"Fiducial_Q18/EBL_KS18_Q18_z_ {z:.1f}.txt"

atom = parse_atom_file("CI.dat")

T_gas = 100.0
T_rad = 2.728 * (1 + z)

nH_grid = np.logspace(-4, 6, 80)

ks19_interp = build_ks19_interpolator(ks19_file)


def make_densities(nH):
    return {
        "electron": 1e-4 * nH,
        "proton": 0.0,
        "H0": nH,
        "helium": 0.1 * nH,
        "p-H2": 0.25 * nH,
        "o-H2": 0.75 * nH,
    }


def run_curve(uv_Emin_eV=None, uv_Emax_eV=None):
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
            uv_field="ks19",
            uv_scale=1.0,
            ks19_interp=ks19_interp,
            uv_Emin_eV=uv_Emin_eV,
            uv_Emax_eV=uv_Emax_eV,
        )

        pop = solve_populations(Q)

        r21.append(pop[1] / pop[0])
        r31.append(pop[2] / pop[0])

    return np.array(r21), np.array(r31)


print("Running KS19 FULL...")
r21_full, r31_full = run_curve(
    uv_Emin_eV=None,
    uv_Emax_eV=None,
)

print("Running KS19 UV-CUTOFF...")
r21_cut, r31_cut = run_curve(
    uv_Emin_eV=5.0,
    uv_Emax_eV=13.6,
)


plt.figure(figsize=(8, 5))
plt.loglog(nH_grid, r21_full, label="KS19 full")
plt.loglog(nH_grid, r21_cut, "--", label="KS19 UV-only 5–13.6 eV")
plt.xlabel(r"$n_{\rm H}$")
plt.ylabel(r"$n_2/n_1$")
plt.title("KS19 full vs UV-cutoff")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.loglog(nH_grid, r31_full, label="KS19 full")
plt.loglog(nH_grid, r31_cut, "--", label="KS19 UV-only 5–13.6 eV")
plt.xlabel(r"$n_{\rm H}$")
plt.ylabel(r"$n_3/n_1$")
plt.title("KS19 full vs UV-cutoff")
plt.legend()
plt.tight_layout()
plt.show()


diff21 = np.max(np.abs(r21_full - r21_cut))
diff31 = np.max(np.abs(r31_full - r31_cut))

reldiff21 = np.max(np.abs((r21_full - r21_cut) / r21_full))
reldiff31 = np.max(np.abs((r31_full - r31_cut) / r31_full))

print("\nMax absolute difference:")
print("n2/n1:", diff21)
print("n3/n1:", diff31)

print("\nMax relative difference:")
print("n2/n1:", reldiff21)
print("n3/n1:", reldiff31)
