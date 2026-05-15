import numpy as np
import matplotlib.pyplot as plt

from atomic_data import parse_atom_file
from solver import build_Q_matrix, solve_populations

from radiation_fields import build_ks19_interpolator, draine_uv_u_nu, ks19_uv_u_nu
from astropy import constants as const
from astropy import units as u


def integrated_uvb_floor_vs_draine(ks19_interp, Emin_eV=5.0, Emax_eV=13.6):
    h_eV = const.h.to(u.eV * u.s).value

    E_grid = np.linspace(Emin_eV, Emax_eV, 1000)
    nu_grid = E_grid / h_eV

    u_draine = np.array([draine_uv_u_nu(nu, scale=1.0) for nu in nu_grid])
    u_ks19 = np.array([ks19_uv_u_nu(nu, ks19_interp, scale=1.0) for nu in nu_grid])

    # integrate over frequency
    int_draine = np.trapz(u_draine, nu_grid)
    int_ks19 = np.trapz(u_ks19, nu_grid)

    IUV_floor = int_ks19 / int_draine
    return IUV_floor, np.log10(IUV_floor)
   

'''
z = 2.3
# --- UVB lower limit relative to Draine at 1000 Å ---
ks19_file = f"Fiducial_Q18/EBL_KS18_Q18_z_ {z:.1f}.txt"
ks19_interp = build_ks19_interpolator(ks19_file)

IUV_floor, log_IUV_floor = integrated_uvb_floor_vs_draine(ks19_interp)

print("IUV floor =", IUV_floor)
print("log IUV floor =", log_IUV_floor)

#for lam in [912, 1000, 1200, 1500, 2000]:
for lam in [1656]:
    nu = const.c.to(u.AA / u.s).value / lam

    ratio = ks19_uv_u_nu(nu, ks19_interp) / draine_uv_u_nu(nu)

    print(lam, np.log10(ratio))
    

quit()
'''


atom = parse_atom_file("CI.dat")

# Observed columns
logN0, err0 = 14.82, 0.18
logN1, err1 = 14.60, 0.06
logN2, err2 = 14.03, 0.06

obs_log21 = logN1 - logN0
obs_log31 = logN2 - logN0

sig_log21 = np.sqrt(err1**2 + err0**2)
sig_log31 = np.sqrt(err2**2 + err0**2)

T_gas = 100.0

nH_grid = np.logspace(0, 3, 60)      # 1–1000 cm^-3
IUV_grid = np.logspace(-1, 2, 60)    # 0.1–100 Draine

redshifts = [0.0, 0.5, 2.3]


def make_densities(nH):
    return {
        "electron": 1e-4 * nH,
        "proton": 0.0,
        "H0": nH,
        "helium": 0.1 * nH,
        "p-H2": 0.25 * nH,
        "o-H2": 0.75 * nH,
    }


def compute_likelihood_map(z):
    T_rad = 2.728 * (1.0 + z)

    prob = np.zeros((len(nH_grid), len(IUV_grid)))
    chi2_grid = np.zeros_like(prob)

    for i, nH in enumerate(nH_grid):
        if i % 10 == 0:
            print(f"z={z}: row {i+1}/{len(nH_grid)}")

        densities = make_densities(nH)

        for j, IUV in enumerate(IUV_grid):
            Q = build_Q_matrix(
                atom,
                densities,
                temperature=T_gas,
                use_detailed_balance=True,
                include_radiation=True,
                T_rad=T_rad,
                include_fluorescence=True,
                uv_field="draine",
                uv_scale=IUV,
            )

            pop = solve_populations(Q)

            model_log21 = np.log10(pop[1] / pop[0])
            model_log31 = np.log10(pop[2] / pop[0])

            chi2 = (
                ((model_log21 - obs_log21) / sig_log21) ** 2
                + ((model_log31 - obs_log31) / sig_log31) ** 2
            )

            chi2_grid[i, j] = chi2
            prob[i, j] = np.exp(-0.5 * chi2)

    prob /= np.sum(prob)

    return prob, chi2_grid


def contour_levels_from_probability(prob):
    flat = prob.ravel()
    idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx])

    p68 = flat[idx][np.where(cumsum >= 0.68)[0][0]]
    p30 = flat[idx][np.where(cumsum >= 0.30)[0][0]]

    return p68, p30


fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

for ax, z in zip(axes, redshifts):
    prob, chi2_grid = compute_likelihood_map(z)

    ii, jj = np.unravel_index(np.argmax(prob), prob.shape)

    best_log_nH = np.log10(nH_grid[ii])
    best_log_IUV = np.log10(IUV_grid[jj])
    best_chi2 = chi2_grid[ii, jj]

    p68, p30 = contour_levels_from_probability(prob)

    # --- UVB lower limit relative to Draine at 1000 Å ---
    ks19_file = f"Fiducial_Q18/EBL_KS18_Q18_z_ {z:.1f}.txt"
    ks19_interp = build_ks19_interpolator(ks19_file)

    #lambda_ref_A = 1000.0
    #For CI1656
    lambda_ref_A = 1656.0

    IUV_floor, log_IUV_floor = integrated_uvb_floor_vs_draine(ks19_interp)

    print("IUV floor =", IUV_floor)
    print("log IUV floor =", log_IUV_floor)
    nu = const.c.to(u.AA / u.s).value / lambda_ref_A
    ratio = ks19_uv_u_nu(nu, ks19_interp) / draine_uv_u_nu(nu)
    print(lambda_ref_A, np.log10(ratio))
    log_IUV_floor_rev = np.log10(ratio)

    im = ax.contourf(
        np.log10(nH_grid),
        np.log10(IUV_grid),
        prob.T,
        levels=50,
        cmap="viridis",
    )

    ax.contour(
        np.log10(nH_grid),
        np.log10(IUV_grid),
        prob.T,
        levels=[p68],
        colors="lime",
        linewidths=2,
    )

    ax.contour(
        np.log10(nH_grid),
        np.log10(IUV_grid),
        prob.T,
        levels=[p30],
        colors="lime",
        linestyles="dashed",
        linewidths=2,
    )

    ax.scatter(
        best_log_nH,
        best_log_IUV,
        marker="x",
        s=80,
        color="red",
    )


    if (np.log10(np.nanmin(IUV_grid))<log_IUV_floor_rev):
        ax.axhline(
            log_IUV_floor_rev,
            color="white",
            linestyle="-",
            linewidth=2,
        )

        ax.text(
            0.05,
            log_IUV_floor_rev + 0.05,
            "UVB floor",
            transform=ax.get_yaxis_transform(),
            color="white",
            fontsize=9,
            va="bottom",
        )

        x_arrow = np.linspace(np.log10(nH_grid[5]), np.log10(nH_grid[-5]), 6)

        for x in x_arrow:
            ax.annotate(
                "",
                xy=(x, log_IUV_floor_rev + 0.32),
                xytext=(x, log_IUV_floor_rev),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
            )

    ax.set_title(
        rf"$z={z}$" + "\n"
        + rf"best: log $n_H$={best_log_nH:.2f}, log $I_{{UV}}$={best_log_IUV:.2f}"
    )

    print(
        f"z={z}: best log nH={best_log_nH:.3f}, "
        f"log IUV={best_log_IUV:.3f}, chi2={best_chi2:.3f}"
    )

axes[0].set_ylabel(r"$\log I_{\rm UV}$")
for ax in axes:
    ax.set_xlabel(r"$\log n_{\rm H}$ [cm$^{-3}$]")

fig.colorbar(im, ax=axes, label="Probability", shrink=0.9)

#plt.tight_layout()
plt.savefig('test_ci_likelihood_redshift.pdf', dpi=100)
print ('test_ci_likelihood_redshift.pdf saved')
plt.show()
