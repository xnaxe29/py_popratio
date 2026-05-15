import os
import numpy as np
import matplotlib.pyplot as plt

from atomic_data import parse_atom_file
from solver import build_Q_matrix, solve_populations
from radiation_fields import build_ks19_interpolator, radiation_u_nu


# --------------------------------------------------
# Settings
# --------------------------------------------------

z = 1.0
ks19_file = f"Fiducial_Q18/EBL_KS18_Q18_z_ {z:.1f}.txt"

atom = parse_atom_file("CI.dat")

T_gas = 100.0
T_rad = 2.728 * (1.0 + z)

nH_grid = np.logspace(-4, 6, 80)


# --------------------------------------------------
# Check file exists
# --------------------------------------------------

if not os.path.exists(ks19_file):
    raise FileNotFoundError(
        f"Could not find KS19 file:\n{ks19_file}\n\n"
        "Check the filename inside Fiducial_Q18."
    )

print("Using KS19 file:")
print(ks19_file)


# --------------------------------------------------
# Load KS19 interpolator
# --------------------------------------------------

ks19_interp = build_ks19_interpolator(ks19_file)

print("KS19 interpolator loaded successfully")


# --------------------------------------------------
# Quick radiation-field sanity check
# --------------------------------------------------

nu_grid = np.logspace(12, 18, 300)

u_cmb = []
u_draine = []
u_popratio = []
u_ks19 = []

for nu in nu_grid:
    u_cmb.append(
        radiation_u_nu(nu, T_rad, uv_field="none")
    )

    u_draine.append(
        radiation_u_nu(nu, T_rad, uv_field="draine", uv_scale=1.0)
    )

    u_popratio.append(
        radiation_u_nu(nu, T_rad, uv_field="popratio", uv_scale=1.0)
    )

    u_ks19.append(
        radiation_u_nu(
            nu,
            T_rad,
            uv_field="ks19",
            uv_scale=1.0,
            ks19_interp=ks19_interp,
        )
    )

plt.figure(figsize=(8, 5))
plt.loglog(nu_grid, u_cmb, label="CMB only")
plt.loglog(nu_grid, u_draine, label="CMB + Draine")
plt.loglog(nu_grid, u_popratio, label="CMB + PopRatio")
plt.loglog(nu_grid, u_ks19, label="CMB + KS19 Q18")

plt.xlabel(r"$\nu$ [Hz]")
plt.ylabel(r"$u_\nu$ [erg cm$^{-3}$ Hz$^{-1}$]")
plt.title("Radiation field comparison")
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def make_densities(nH):
    return {
        "electron": 1e-4 * nH,
        "proton": 0.0,
        "H0": nH,
        "helium": 0.1 * nH,
        "p-H2": 0.25 * nH,
        "o-H2": 0.75 * nH,
    }


def compute_curve(uv_field, uv_scale=1.0):
    ratio_21 = []
    ratio_31 = []

    for nH in nH_grid:
        Q = build_Q_matrix(
            atom,
            make_densities(nH),
            temperature=T_gas,
            use_detailed_balance=True,
            include_radiation=True,
            T_rad=T_rad,
            include_fluorescence=True,
            uv_field=uv_field,
            uv_scale=uv_scale,
            ks19_interp=ks19_interp,
        )

        pop = solve_populations(Q)

        if abs(np.sum(pop) - 1.0) > 1e-10:
            print("Warning: population sum =", np.sum(pop), "nH =", nH)

        ratio_21.append(pop[1] / pop[0])
        ratio_31.append(pop[2] / pop[0])

    return np.array(ratio_21), np.array(ratio_31)


# --------------------------------------------------
# Compute curves
# --------------------------------------------------

r21_none, r31_none = compute_curve("none")
r21_draine, r31_draine = compute_curve("draine")
r21_popratio, r31_popratio = compute_curve("popratio")
r21_ks19, r31_ks19 = compute_curve("ks19")


# --------------------------------------------------
# Plot n2/n1
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.loglog(nH_grid, r21_none, label="CMB only")
plt.loglog(nH_grid, r21_draine, label="Draine + fluorescence")
plt.loglog(nH_grid, r21_popratio, label="PopRatio + fluorescence")
plt.loglog(nH_grid, r21_ks19, label="KS19 Q18 + fluorescence")

plt.xlabel(r"$n_{\rm H}$ [cm$^{-3}$]")
plt.ylabel(r"$n_2/n_1$")
plt.title(r"C I: $n_2/n_1$ with KS19 test")
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Plot n3/n1
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.loglog(nH_grid, r31_none, label="CMB only")
plt.loglog(nH_grid, r31_draine, label="Draine + fluorescence")
plt.loglog(nH_grid, r31_popratio, label="PopRatio + fluorescence")
plt.loglog(nH_grid, r31_ks19, label="KS19 Q18 + fluorescence")

plt.xlabel(r"$n_{\rm H}$ [cm$^{-3}$]")
plt.ylabel(r"$n_3/n_1$")
plt.title(r"C I: $n_3/n_1$ with KS19 test")
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Print representative values
# --------------------------------------------------

print("\nRepresentative values at nH ~ 100 cm^-3:")

idx = np.argmin(np.abs(nH_grid - 100.0))

print("nH =", nH_grid[idx])
print("CMB only       :", r21_none[idx], r31_none[idx])
print("Draine         :", r21_draine[idx], r31_draine[idx])
print("PopRatio       :", r21_popratio[idx], r31_popratio[idx])
print("KS19 Q18       :", r21_ks19[idx], r31_ks19[idx])

print("\nKS19 test completed.")
