import numpy as np
import matplotlib.pyplot as plt

from atomic_data import parse_atom_file
from solver import (
    build_Q_matrix,
    solve_populations,
    compute_population_ratios,
    k_B_eV,
)

from radiation_fields import build_ks19_interpolator


# ==================================================
# User settings
# ==================================================

atom_file = "CI.dat"

# Optional KS19 file
# Change this to your actual path, or set to None
ks19_file = None
# ks19_file = "Fiducial_Q18/EBL_KS18_Q18_z_1.000.txt"

z = 1.0
T_gas_default = 100.0
T_rad_default = 2.728 * (1 + z)

nH_grid = np.logspace(-4, 6, 80)

density_scaling = {
    "electron": 1e-4,
    "proton": 0.0,
    "H0": 1.0,
    "helium": 0.1,
    "p-H2": 0.25,
    "o-H2": 0.75,
}


# ==================================================
# Load atom
# ==================================================

atom = parse_atom_file(atom_file)

print("\nLoaded atom:")
print("species =", atom["species_name"])
print("levels =", len(atom["levels"]))
print("A transitions =", len(atom["A"]))
print("fluorescence transitions =", len(atom["fluorescence"]))
print("collision partners =", [p["name"] for p in atom["collision_partners"]])


# ==================================================
# Optional KS19 interpolator
# ==================================================

ks19_interp = None
if ks19_file is not None:
    ks19_interp = build_ks19_interpolator(ks19_file)
    print("\nLoaded KS19 field:", ks19_file)


# ==================================================
# Helper
# ==================================================

def make_densities(nH):
    return {
        name: factor * nH
        for name, factor in density_scaling.items()
    }


def run_single_model(
    nH,
    T_gas=100.0,
    T_rad=T_rad_default,
    include_radiation=True,
    include_fluorescence=True,
    uv_field="none",
    uv_scale=1.0,
):
    Q = build_Q_matrix(
        atom,
        make_densities(nH),
        temperature=T_gas,
        use_detailed_balance=True,
        include_radiation=include_radiation,
        T_rad=T_rad,
        include_fluorescence=include_fluorescence,
        uv_field=uv_field,
        uv_scale=uv_scale,
        ks19_interp=ks19_interp,
    )

    pop = solve_populations(Q)

    if abs(np.sum(pop) - 1.0) > 1e-10:
        print("WARNING: population sum =", np.sum(pop))

    return pop


# ==================================================
# Test 1: population normalization
# ==================================================

print("\nTEST 1: population normalization")

pop = run_single_model(
    nH=100.0,
    T_gas=T_gas_default,
    T_rad=T_rad_default,
    include_radiation=True,
    include_fluorescence=True,
    uv_field="draine",
    uv_scale=1.0,
)

print("pop =", pop)
print("sum(pop) =", np.sum(pop))


# ==================================================
# Test 2: LTE high-density limit
# ==================================================

print("\nTEST 2: LTE high-density limit")

T_lte = 100.0
nH_lte = 1e10

pop_lte = run_single_model(
    nH=nH_lte,
    T_gas=T_lte,
    T_rad=T_rad_default,
    include_radiation=True,
    include_fluorescence=True,
    uv_field="draine",
    uv_scale=1.0,
)

energies = np.array([lvl["energy"] for lvl in atom["levels"]])
g = np.array([lvl["g"] for lvl in atom["levels"]])

E_eV = (energies - energies[0]) * 1.23981e-4

boltz_ratio = (g / g[0]) * np.exp(-E_eV / (k_B_eV * T_lte))
boltz_pop = boltz_ratio / np.sum(boltz_ratio)

print("solver LTE pop =", pop_lte)
print("Boltzmann pop  =", boltz_pop)
print("max abs diff   =", np.max(np.abs(pop_lte - boltz_pop)))


# ==================================================
# Test 3: low-density radiation limit
# ==================================================

print("\nTEST 3: low-density radiation limit")

nH_low = 1e-12
T_gas = 100.0
T_rad_high = 2.728 * (1 + 5.0)

pop_no = run_single_model(
    nH=nH_low,
    T_gas=T_gas,
    T_rad=T_rad_high,
    include_radiation=False,
    include_fluorescence=False,
    uv_field="none",
)

pop_cmb = run_single_model(
    nH=nH_low,
    T_gas=T_gas,
    T_rad=T_rad_high,
    include_radiation=True,
    include_fluorescence=False,
    uv_field="none",
)

pop_full = run_single_model(
    nH=nH_low,
    T_gas=T_gas,
    T_rad=T_rad_high,
    include_radiation=True,
    include_fluorescence=True,
    uv_field="draine",
    uv_scale=1.0,
)

print("No radiation ratio n2/n1 =", pop_no[1] / pop_no[0])
print("CMB only ratio n2/n1     =", pop_cmb[1] / pop_cmb[0])
print("Full ratio n2/n1         =", pop_full[1] / pop_full[0])


# ==================================================
# Test 4: density dependence
# ==================================================

print("\nTEST 4: density dependence")

cases = {
    "collisions only": dict(
        include_radiation=False,
        include_fluorescence=False,
        uv_field="none",
        uv_scale=1.0,
    ),
    "CMB only": dict(
        include_radiation=True,
        include_fluorescence=False,
        uv_field="none",
        uv_scale=1.0,
    ),
    "Draine direct": dict(
        include_radiation=True,
        include_fluorescence=False,
        uv_field="draine",
        uv_scale=1.0,
    ),
    "Draine + fluorescence": dict(
        include_radiation=True,
        include_fluorescence=True,
        uv_field="draine",
        uv_scale=1.0,
    ),
    "PopRatio + fluorescence": dict(
        include_radiation=True,
        include_fluorescence=True,
        uv_field="popratio",
        uv_scale=1.0,
    ),
}

if ks19_interp is not None:
    cases["KS19 + fluorescence"] = dict(
        include_radiation=True,
        include_fluorescence=True,
        uv_field="ks19",
        uv_scale=1.0,
    )

results_21 = {name: [] for name in cases}
results_31 = {name: [] for name in cases}

for nH in nH_grid:
    for name, opts in cases.items():
        pop = run_single_model(
            nH=nH,
            T_gas=T_gas_default,
            T_rad=T_rad_default,
            **opts,
        )

        results_21[name].append(pop[1] / pop[0])
        results_31[name].append(pop[2] / pop[0])


plt.figure(figsize=(8, 6))
for name in cases:
    plt.loglog(nH_grid, results_21[name], label=name, lw=2, alpha=0.5)

plt.xlabel(r"$n_{\rm H}$ [cm$^{-3}$]")
plt.ylabel(r"$n_2/n_1$")
plt.title(r"C I density test: $n_2/n_1$")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
for name in cases:
    plt.loglog(nH_grid, results_31[name], label=name, lw=2, alpha=0.5)

plt.xlabel(r"$n_{\rm H}$ [cm$^{-3}$]")
plt.ylabel(r"$n_3/n_1$")
plt.title(r"C I density test: $n_3/n_1$")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# ==================================================
# Test 5: gas temperature dependence
# ==================================================

print("\nTEST 5: gas temperature dependence")

T_grid = np.array([10, 30, 100, 300, 1000, 3000, 10000])
nH_fixed = 100.0

ratio_T_21 = []
ratio_T_31 = []

for T_gas in T_grid:
    pop = run_single_model(
        nH=nH_fixed,
        T_gas=T_gas,
        T_rad=T_rad_default,
        include_radiation=True,
        include_fluorescence=True,
        uv_field="draine",
        uv_scale=1.0,
    )

    ratio_T_21.append(pop[1] / pop[0])
    ratio_T_31.append(pop[2] / pop[0])

    print(
        f"T_gas={T_gas:8.1f} K  "
        f"n2/n1={pop[1]/pop[0]:.4e}  "
        f"n3/n1={pop[2]/pop[0]:.4e}"
    )

plt.figure(figsize=(7, 5))
plt.loglog(T_grid, ratio_T_21, marker="o", label=r"$n_2/n_1$", lw=2, alpha=0.5)
plt.loglog(T_grid, ratio_T_31, marker="o", label=r"$n_3/n_1$", lw=2, alpha=0.5)
plt.xlabel(r"$T_{\rm gas}$ [K]")
plt.ylabel("Population ratio")
plt.title(r"Gas temperature dependence at fixed $n_{\rm H}$")
plt.legend()
plt.tight_layout()
plt.show()


# ==================================================
# Test 6: radiation temperature dependence
# ==================================================

print("\nTEST 6: radiation temperature dependence")

T_rad_grid = np.array([2.728, 5, 10, 20, 50, 100, 300, 1000])
nH_fixed = 1e-4
T_gas_fixed = 100.0

ratio_Trad_21 = []
ratio_Trad_31 = []

for T_rad in T_rad_grid:
    pop = run_single_model(
        nH=nH_fixed,
        T_gas=T_gas_fixed,
        T_rad=T_rad,
        include_radiation=True,
        include_fluorescence=True,
        uv_field="none",
        uv_scale=1.0,
    )

    ratio_Trad_21.append(pop[1] / pop[0])
    ratio_Trad_31.append(pop[2] / pop[0])

    print(
        f"T_rad={T_rad:8.2f} K  "
        f"n2/n1={pop[1]/pop[0]:.4e}  "
        f"n3/n1={pop[2]/pop[0]:.4e}"
    )

plt.figure(figsize=(7, 5))
plt.loglog(T_rad_grid, ratio_Trad_21, marker="o", label=r"$n_2/n_1$", lw=2, alpha=0.5)
plt.loglog(T_rad_grid, ratio_Trad_31, marker="o", label=r"$n_3/n_1$", lw=2, alpha=0.5)
plt.xlabel(r"$T_{\rm rad}$ [K]")
plt.ylabel("Population ratio")
plt.title(r"Radiation temperature dependence at very low density")
plt.legend()
plt.tight_layout()
plt.show()


# ==================================================
# Test 7: UV-field scaling dependence
# ==================================================

print("\nTEST 7: UV-field scaling dependence")

uv_grid = np.logspace(-1, 2, 50)
nH_fixed = 30.0
T_gas_fixed = 100.0

uv_fields_to_test = ["draine", "popratio"]
if ks19_interp is not None:
    uv_fields_to_test.append("ks19")

plt.figure(figsize=(8, 6))

for uv_field in uv_fields_to_test:
    vals = []

    for uv_scale in uv_grid:
        pop = run_single_model(
            nH=nH_fixed,
            T_gas=T_gas_fixed,
            T_rad=T_rad_default,
            include_radiation=True,
            include_fluorescence=True,
            uv_field=uv_field,
            uv_scale=uv_scale,
        )

        vals.append(pop[1] / pop[0])

    plt.loglog(uv_grid, vals, label=uv_field, lw=2, alpha=0.5)

plt.xlabel("UV scale")
plt.ylabel(r"$n_2/n_1$")
plt.title(r"UV-field scaling at fixed density")
plt.legend()
plt.tight_layout()
plt.show()




'''
# ==================================================
# Test 8: likelihood map like old C I figure
# ==================================================

print("\nTEST 8: C I likelihood map")

logN0, err0 = 14.82, 0.18
logN1, err1 = 14.60, 0.06
logN2, err2 = 14.03, 0.06

obs_log21 = logN1 - logN0
obs_log31 = logN2 - logN0

sig_log21 = np.sqrt(err1**2 + err0**2)
sig_log31 = np.sqrt(err2**2 + err0**2)

nH_like = np.logspace(0, 3, 50)
IUV_like = np.logspace(-1, 2, 50)

prob = np.zeros((len(nH_like), len(IUV_like)))
chi2_grid = np.zeros_like(prob)

for i, nH in enumerate(nH_like):
    if i % 10 == 0:
        print(f"Likelihood row {i+1}/{len(nH_like)}")

    for j, IUV in enumerate(IUV_like):
        pop = run_single_model(
            nH=nH,
            T_gas=100.0,
            T_rad=2.728,
            include_radiation=True,
            include_fluorescence=True,
            uv_field="draine",
            uv_scale=IUV,
        )

        model_log21 = np.log10(pop[1] / pop[0])
        model_log31 = np.log10(pop[2] / pop[0])

        chi2 = (
            ((model_log21 - obs_log21) / sig_log21) ** 2
            + ((model_log31 - obs_log31) / sig_log31) ** 2
        )

        chi2_grid[i, j] = chi2
        prob[i, j] = np.exp(-0.5 * chi2)

prob /= np.sum(prob)

ii, jj = np.unravel_index(np.argmax(prob), prob.shape)

print("\nBest-fit likelihood point:")
print("log nH  =", np.log10(nH_like[ii]))
print("log IUV =", np.log10(IUV_like[jj]))
print("chi2    =", chi2_grid[ii, jj])

flat = prob.ravel()
idx = np.argsort(flat)[::-1]
cumsum = np.cumsum(flat[idx])

p68 = flat[idx][np.where(cumsum >= 0.68)[0][0]]
p30 = flat[idx][np.where(cumsum >= 0.30)[0][0]]

plt.figure(figsize=(8, 6))

plt.contourf(
    np.log10(nH_like),
    np.log10(IUV_like),
    prob.T,
    levels=50,
    cmap="viridis",
)

plt.contour(
    np.log10(nH_like),
    np.log10(IUV_like),
    prob.T,
    levels=[p68],
    colors="lime",
    linewidths=2,
)

plt.contour(
    np.log10(nH_like),
    np.log10(IUV_like),
    prob.T,
    levels=[p30],
    colors="lime",
    linestyles="dashed",
    linewidths=2,
)

plt.scatter(
    np.log10(nH_like[ii]),
    np.log10(IUV_like[jj]),
    color="red",
    marker="x",
    s=80,
    label="best fit",
)

plt.xlabel(r"$\log n_{\rm H}$ [cm$^{-3}$]")
plt.ylabel(r"$\log I_{\rm UV}$")
plt.title("C I likelihood map")
plt.legend()
plt.colorbar(label="Probability")
plt.tight_layout()
plt.show()
'''


print("\nAll tests completed.")
