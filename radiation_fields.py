import numpy as np

from astropy import constants as const
from astropy import units as u
from scipy.interpolate import interp1d


# --------------------------------------------------
# Physical constants
# --------------------------------------------------

c = const.c.to(u.cm / u.s).value
h = const.h.to(u.erg * u.s).value
k_B_erg = const.k_B.to(u.erg / u.K).value


# --------------------------------------------------
# CMB / blackbody radiation
# --------------------------------------------------

def planck_u_nu(nu, T):
    """
    Blackbody spectral energy density u_nu.

    Parameters
    ----------
    nu : float
        Frequency in Hz.
    T : float
        Radiation temperature in K.

    Returns
    -------
    float
        u_nu in erg cm^-3 Hz^-1.
    """

    if T <= 0:
        return 0.0

    x = h * nu / (k_B_erg * T)

    if x > 700:
        return 0.0

    return (8.0 * np.pi * h * nu**3 / c**3) / (np.exp(x) - 1.0)


# --------------------------------------------------
# Draine 1978 standard UV field
# --------------------------------------------------

def draine_uv_u_nu(nu, scale=1.0):
    """
    Draine (1978) standard interstellar UV radiation field.

    Valid for photon energies 5--13.6 eV.

    Parameters
    ----------
    nu : float
        Frequency in Hz.
    scale : float
        Multiplicative scale factor.

    Returns
    -------
    float
        u_nu in erg cm^-3 Hz^-1.
    """

    h_eV = const.h.to(u.eV * u.s).value
    eV_to_erg = (1.0 * u.eV).to(u.erg).value

    E_eV = h_eV * nu

    if E_eV < 5.0 or E_eV > 13.6:
        return 0.0

    # Draine 1978 standard UV background:
    # F(E) in photons cm^-2 s^-1 sr^-1 eV^-1
    F_E = (
        1.658e6 * E_eV
        - 2.152e5 * E_eV**2
        + 6.919e3 * E_eV**3
    )

    if F_E <= 0:
        return 0.0

    # Convert photon intensity per eV to energy density per Hz:
    # J_nu = F_E * E_erg * dE/dnu = F_E * E_erg * h_eV
    # u_nu = 4 pi J_nu / c
    J_nu = F_E * (E_eV * eV_to_erg) * h_eV
    u_nu = 4.0 * np.pi * J_nu / c

    return scale * u_nu


# --------------------------------------------------
# PopRatio / Gondhalekar Galactic UV field
# --------------------------------------------------

def popratio_galactic_uv_u_nu(lambda_A, scale=1.0):
    """
    Original PopRatio Galactic UV field, based on Gondhalekar et al.

    Parameters
    ----------
    lambda_A : float
        Wavelength in Angstrom.
    scale : float
        Multiplicative scale factor.

    Returns
    -------
    float
        u_nu in erg cm^-3 Hz^-1.
    """

    lambdas = np.array([
        930., 975., 1025., 1075., 1125., 1175., 1220., 1270.,
        1325., 1380., 1400., 1420., 1440., 1460., 1480., 1500.,
        1520., 1540., 1560., 1580., 1600., 1620., 1640., 1660.,
        1680., 1700., 1720., 1780., 1800., 1820., 1840., 1860.,
        1880., 1900., 1920., 1940., 1960., 1980., 2000., 2020.,
        2040., 2060., 2080., 2100., 2120., 2180., 2200., 2220.,
        2240., 2260., 2280., 2300., 2320., 2340., 2360., 2380.,
        2400., 2420., 2440., 2460., 2480., 2500., 2520., 2740.
    ])

    field = np.array([
        83.1, 148.2, 145.7, 159.3, 181.6, 166.7, 98.1, 175.4,
        177.8, 156.9, 149.4, 149.4, 154.4, 156.9, 153.1, 150.7,
        143.3, 134.6, 130.9, 134.6, 128.4, 124.7, 128.4, 133.4,
        135.9, 133.4, 126.0, 115.8, 118.2, 114.1, 110.4, 105.1,
        105.5, 101.0, 97.6, 97.2, 97.2, 96.6, 93.1, 88.8,
        86.9, 84.1, 82.3, 81.4, 78.9, 75.1, 73.6, 73.6,
        71.4, 72.7, 72.7, 71.4, 69.2, 67.2, 65.6, 63.6,
        63.7, 64.3, 65.6, 65.2, 64.8, 65.0, 62.2, 53.8
    ])

    if lambda_A < lambdas.min() or lambda_A > lambdas.max():
        return 0.0

    f_lambda = np.interp(lambda_A, lambdas, field)

    # PopRatio conversion:
    # u_nu = lambda^2 / c^2 * 1e-16 * field(lambda)
    return scale * (lambda_A**2 / c**2) * 1e-16 * f_lambda


# --------------------------------------------------
# Khaire & Srianand UVB table
# --------------------------------------------------

def load_ks19_file(filename):
    """
    Load one Khaire & Srianand EBL/UVB file.

    Expected columns
    ----------------
    column 1 : wavelength in Angstrom
    column 2 : J_nu in erg s^-1 cm^-2 Hz^-1 sr^-1

    Returns
    -------
    nu : ndarray
        Frequency in Hz.
    Jnu : ndarray
        Specific intensity in erg s^-1 cm^-2 Hz^-1 sr^-1.
    """

    data = np.loadtxt(filename)

    lambda_A = data[:, 0]
    Jnu = data[:, 1]

    nu = const.c.to(u.AA / u.s).value / lambda_A

    # Sort by increasing frequency for interpolation
    order = np.argsort(nu)

    return nu[order], Jnu[order]


def build_ks19_interpolator(filename):
    """
    Build log-log interpolator for one KS19 redshift file.

    Parameters
    ----------
    filename : str
        Path to KS19 file, e.g.
        Fiducial_Q18/EBL_KS18_Q18_z_1.000.txt

    Returns
    -------
    interp1d
        Interpolator in log10(nu) -> log10(Jnu).
    """

    nu, Jnu = load_ks19_file(filename)

    good = (nu > 0) & (Jnu > 0)

    return interp1d(
        np.log10(nu[good]),
        np.log10(Jnu[good]),
        bounds_error=False,
        fill_value=-300.0,
    )


def ks19_uv_u_nu(
    nu,
    ks19_interp,
    scale=1.0,
    Emin_eV=None,
    Emax_eV=None,
):
    h_eV = const.h.to(u.eV * u.s).value
    E_eV = h_eV * nu

    if Emin_eV is not None and E_eV < Emin_eV:
        return 0.0

    if Emax_eV is not None and E_eV > Emax_eV:
        return 0.0

    logJ = float(ks19_interp(np.log10(nu)))

    if logJ < -250:
        return 0.0

    Jnu = 10.0**logJ

    return scale * (4.0 * np.pi / c) * Jnu

# --------------------------------------------------
# Unified radiation interface
# --------------------------------------------------

def radiation_u_nu(
    nu,
    T_rad,
    uv_field="none",
    uv_scale=1.0,
    ks19_interp=None,
    uv_Emin_eV=None,
    uv_Emax_eV=None,
):
    """
    Total radiation energy density used by the solver.

    Always includes CMB / blackbody component from T_rad.
    Optionally adds one UV field.

    Parameters
    ----------
    nu : float
        Frequency in Hz.
    T_rad : float
        Radiation temperature in K.
    uv_field : str
        One of:
        "none"      : CMB only
        "draine"    : Draine 1978 local ISM UV field
        "popratio"  : original PopRatio/Gondhalekar Galactic UV field
        "ks19"      : Khaire & Srianand extragalactic UVB
    uv_scale : float
        Multiplicative UV-field scale.
    ks19_interp : interp1d or None
        Required only for uv_field="ks19".

    Returns
    -------
    float
        u_nu in erg cm^-3 Hz^-1.
    """

    u_val = planck_u_nu(nu, T_rad)

    if uv_field is None:
        uv_field = "none"

    uv_field = uv_field.lower()

    if uv_field == "none":
        return u_val

    elif uv_field == "draine":
        u_val += draine_uv_u_nu(nu, scale=uv_scale)

    elif uv_field == "popratio":
        lambda_A = const.c.to(u.AA / u.s).value / nu
        u_val += popratio_galactic_uv_u_nu(lambda_A, scale=uv_scale)

    elif uv_field == "ks19":
        u_val += ks19_uv_u_nu(
            nu,
            ks19_interp=ks19_interp,
            scale=uv_scale,
            Emin_eV=uv_Emin_eV,
            Emax_eV=uv_Emax_eV,
        )

    else:
        raise ValueError(
            "Unknown uv_field. Use one of: "
            "'none', 'draine', 'popratio', 'ks19'."
        )

    return u_val


if __name__ == "__main__":
    print("radiation_fields module loaded successfully")
