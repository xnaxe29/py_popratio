import numpy as np

from astropy import constants as const
from astropy import units as u

from radiation_fields import radiation_u_nu


# --------------------------------------------------
# Constants
# --------------------------------------------------

k_B_eV = const.k_B.to(u.eV / u.K).value
c = const.c.to(u.cm / u.s).value
h = const.h.to(u.erg * u.s).value


# --------------------------------------------------
# Einstein B coefficients
# --------------------------------------------------

def compute_B_matrix(atom):
    n = len(atom["levels"])
    B = np.zeros((n, n))

    energies = np.array([lvl["energy"] for lvl in atom["levels"]])
    g = np.array([lvl["g"] for lvl in atom["levels"]])

    for t in atom["A"]:
        i = t["upper"] - 1
        j = t["lower"] - 1

        delta_E = energies[i] - energies[j]

        if delta_E <= 0:
            continue

        nu = delta_E * c
        Aij = t["Aij"]

        B_down = Aij * c**3 / (8.0 * np.pi * h * nu**3)
        B_up = (g[i] / g[j]) * B_down

        B[i, j] = B_down
        B[j, i] = B_up

    return B


# --------------------------------------------------
# Collision-rate interpolation
# --------------------------------------------------

def interpolate_rate(temps, rates, T, partner_name):
    temps = np.asarray(temps, dtype=float)
    rates = np.asarray(rates, dtype=float)

    rates = rates.copy()
    rates[rates <= 0] = 1e-30

    if T <= temps.min():
        return rates[0]

    if T >= temps.max():
        return rates[-1]

    idx = np.searchsorted(temps, T) - 1

    T1, T2 = temps[idx], temps[idx + 1]
    R1, R2 = rates[idx], rates[idx + 1]

    logT1, logT2 = np.log10(T1), np.log10(T2)
    logR1, logR2 = np.log10(R1), np.log10(R2)
    logT = np.log10(T)

    logR = logR1 + (logT - logT1) * (logR2 - logR1) / (logT2 - logT1)

    return 10.0**logR


# --------------------------------------------------
# Fluorescence
# --------------------------------------------------

def compute_fluorescence_matrix(
    atom,
    T_rad,
    uv_field="none",
    uv_scale=1.0,
    ks19_interp=None,
    uv_Emin_eV=None,
    uv_Emax_eV=None,
):
    n_levels = len(atom["levels"])
    Gamma = np.zeros((n_levels, n_levels))

    energies = np.array([lvl["energy"] for lvl in atom["levels"]])
    g_levels = np.array([lvl["g"] for lvl in atom["levels"]])

    upper_energies = sorted(set(f["energy"] for f in atom["fluorescence"]))

    for E_upper in upper_energies:
        lines = [
            f for f in atom["fluorescence"]
            if np.isclose(f["energy"], E_upper, rtol=0.0, atol=1e-6)
        ]

        if len(lines) == 0:
            continue

        g_upper = lines[0]["g"]
        A_total = np.sum([f["Aij"] for f in lines])

        if A_total <= 0:
            continue

        for f_abs in lines:
            i = f_abs["lower"] - 1

            delta_E = E_upper - energies[i]

            if delta_E <= 0:
                continue

            nu = delta_E * c

            u_rad = radiation_u_nu(
                nu,
                T_rad,
                uv_field=uv_field,
                uv_scale=uv_scale,
                ks19_interp=ks19_interp,
                uv_Emin_eV=uv_Emin_eV,
                uv_Emax_eV=uv_Emax_eV,
            )

            A_upper_to_i = f_abs["Aij"]

            B_upper_to_i = A_upper_to_i * c**3 / (8.0 * np.pi * h * nu**3)
            B_i_to_upper = (g_upper / g_levels[i]) * B_upper_to_i

            K_i_upper = B_i_to_upper * u_rad

            for f_emit in lines:
                j = f_emit["lower"] - 1

                if i == j:
                    continue

                A_upper_to_j = f_emit["Aij"]

                Gamma[i, j] += K_i_upper * A_upper_to_j / A_total

    return Gamma


# --------------------------------------------------
# Q matrix
# --------------------------------------------------

def build_Q_matrix(
    atom,
    densities,
    temperature,
    use_detailed_balance=True,
    include_radiation=True,
    T_rad=2.73,
    include_fluorescence=True,
    uv_field="none",
    uv_scale=1.0,
    ks19_interp=None,
    uv_Emin_eV=None,
    uv_Emax_eV=None,
):
    n_levels = len(atom["levels"])
    Q = np.zeros((n_levels, n_levels))

    energies = np.array([lvl["energy"] for lvl in atom["levels"]])
    g = np.array([lvl["g"] for lvl in atom["levels"]])

    # --------------------------------------------------
    # 1. Spontaneous radiative transitions A_ij
    # --------------------------------------------------
    for t in atom["A"]:
        i = t["upper"] - 1
        j = t["lower"] - 1
        Q[i, j] += t["Aij"]

    # --------------------------------------------------
    # 2. Collisions
    # --------------------------------------------------
    densities_norm = {
        key.strip().lower(): value
        for key, value in densities.items()
    }

    for partner in atom["collision_partners"]:
        partner_name = partner["name"].strip()
        partner_key = partner_name.lower()

        n_k = densities_norm.get(partner_key, 0.0)

        if n_k == 0.0:
            continue

        for table in partner["tables"]:
            q = interpolate_rate(
                table["temperature"],
                table["rate"],
                temperature,
                partner_name,
            )

            i = table["level1"] - 1
            j = table["level2"] - 1

            if not use_detailed_balance:
                Q[i, j] += n_k * q
                continue

            if energies[i] > energies[j]:
                Eij_eV = (energies[i] - energies[j]) * 1.23981e-4

                q_down = q
                q_up = (g[i] / g[j]) * np.exp(
                    -Eij_eV / (k_B_eV * temperature)
                ) * q_down

                Q[i, j] += n_k * q_down
                Q[j, i] += n_k * q_up

            else:
                Eji_eV = (energies[j] - energies[i]) * 1.23981e-4

                q_down = q
                q_up = (g[j] / g[i]) * np.exp(
                    -Eji_eV / (k_B_eV * temperature)
                ) * q_down

                Q[j, i] += n_k * q_down
                Q[i, j] += n_k * q_up

    # --------------------------------------------------
    # 3. Direct radiation B_ij u_ij
    # --------------------------------------------------
    if include_radiation:
        B = compute_B_matrix(atom)

        for i in range(n_levels):
            for j in range(n_levels):
                if i == j:
                    continue

                if B[i, j] == 0.0:
                    continue

                delta_E = abs(energies[i] - energies[j])

                if delta_E <= 0:
                    continue

                nu = delta_E * c

                u_rad = radiation_u_nu(
                    nu,
                    T_rad,
                    uv_field=uv_field,
                    uv_scale=uv_scale,
                    ks19_interp=ks19_interp,
                    uv_Emin_eV=uv_Emin_eV,
                    uv_Emax_eV=uv_Emax_eV,
                )

                Q[i, j] += B[i, j] * u_rad

    # --------------------------------------------------
    # 4. Fluorescence / UV pumping
    # --------------------------------------------------
    if include_fluorescence:
        Gamma = compute_fluorescence_matrix(
            atom,
            T_rad,
            uv_field=uv_field,
            uv_scale=uv_scale,
            ks19_interp=ks19_interp,
            uv_Emin_eV=uv_Emin_eV,
            uv_Emax_eV=uv_Emax_eV,
        )

        Q += Gamma

    return Q


# --------------------------------------------------
# Solve statistical equilibrium
# --------------------------------------------------

def solve_populations(Q):
    n = Q.shape[0]

    M = np.zeros((n - 1, n - 1))
    I = np.zeros(n - 1)

    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                M[i - 1, j - 1] = -np.sum(Q[i, :])
            else:
                M[i - 1, j - 1] = Q[j, i]

        I[i - 1] = -Q[0, i]

    X = np.linalg.solve(M, I)

    populations = np.insert(X, 0, 1.0)
    populations /= populations.sum()

    return populations


# --------------------------------------------------
# Convenience grid wrapper
# --------------------------------------------------

def compute_population_ratios(
    atom,
    nH_grid,
    T_gas,
    z=0.0,
    density_scaling=None,
    use_detailed_balance=True,
    include_radiation=True,
    include_fluorescence=True,
    uv_field="none",
    uv_scale=1.0,
    ks19_interp=None,
    uv_Emin_eV=None,
    uv_Emax_eV=None,
):
    nH_grid = np.asarray(nH_grid, dtype=float)
    T_rad = 2.728 * (1.0 + z)

    if density_scaling is None:
        density_scaling = {
            "electron": 1e-4,
            "proton": 0.0,
            "H0": 1.0,
            "helium": 0.1,
            "p-H2": 0.25,
            "o-H2": 0.75,
        }

    populations = []

    for nH in nH_grid:
        densities = {
            name: factor * nH
            for name, factor in density_scaling.items()
        }

        Q = build_Q_matrix(
            atom,
            densities,
            temperature=T_gas,
            use_detailed_balance=use_detailed_balance,
            include_radiation=include_radiation,
            T_rad=T_rad,
            include_fluorescence=include_fluorescence,
            uv_field=uv_field,
            uv_scale=uv_scale,
            ks19_interp=ks19_interp,
            uv_Emin_eV=uv_Emin_eV,
            uv_Emax_eV=uv_Emax_eV,
        )

        pop = solve_populations(Q)

        if abs(np.sum(pop) - 1.0) > 1e-10:
            print(f"Warning: population sum = {np.sum(pop)} at nH = {nH}")

        populations.append(pop)

    populations = np.asarray(populations)

    ratios_to_ground = populations / populations[:, [0]]

    return {
        "nH": nH_grid,
        "populations": populations,
        "ratios_to_ground": ratios_to_ground,
    }


if __name__ == "__main__":
    print("solver module loaded successfully")
