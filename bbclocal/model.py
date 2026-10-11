"""Per-event quantities of SALT2mu (compute_more_TABLEVAR, fcn_muerrsq, get_gammadm_host)."""
import numpy as np

F32 = np.float32
LN10 = np.log(10.0)
CLIGHT = 299792.458
M0_DEFAULT = -30.0          # M0_DEFAULT_SALT2
EIGMIN_COV = 1e-4


def derive(t):
    """Add mB, the 3x3 (mB, x1, c) fit covariance (float32, packed as [00, 01, 02, 11, 12, 22]), mBERR, BADCOV, zmuerr and
    NCOVFIX to a table of float32 FITRES columns. Idempotent."""
    if "__derived__" in t:
        return t
    x0 = t["x0"].astype(F32)
    with np.errstate(divide="ignore", invalid="ignore"):
        t["mB"] = (F32(-2.5) * np.log10(x0)).astype(F32)                       # single-precision log10f, as in C
        sf = -2.5 / (x0.astype(float) * LN10)
    c00 = (sf * t["x0ERR"].astype(float)) ** 2
    c01 = sf * t["COV_x1_x0"].astype(float)
    c02 = sf * t["COV_c_x0"].astype(float)
    c11 = t["x1ERR"].astype(float) ** 2
    c22 = t["cERR"].astype(float) ** 2
    c12 = t["COV_x1_c"].astype(float)
    C = np.stack([np.stack([c00, c01, c02], -1), np.stack([c01, c11, c12], -1), np.stack([c02, c12, c22], -1)], -2)
    finite = np.isfinite(C).all(axis=(1, 2))
    Cf = np.where(finite[:, None, None], C, 0.0)
    ev, V = np.linalg.eigh(Cf)
    # update_covMatrix: only if an eigenvalue is <= 0, every eigenvalue below EIGMIN_COV is raised to EIGMIN_COV
    bad = (ev <= 0.0).any(axis=1) & finite
    if bad.any():
        evb = np.maximum(ev[bad], EIGMIN_COV)
        C[bad] = np.einsum("nik,nk,njk->nij", V[bad], evb, V[bad])
    t["NCOVFIX"] = int(bad.sum())
    t["cov"] = np.stack([C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 1, 1], C[:, 1, 2], C[:, 2, 2]]).astype(F32)
    t["mBERR"] = np.sqrt(C[:, 0, 0])
    with np.errstate(invalid="ignore"):
        t["BADCOV"] = (np.abs(t["cov"]).max(axis=0) > 8) | (t["mBERR"] > 1) | ~finite
    t["BADERR"] = ~((t["x0ERR"] > 0) & (t["x1ERR"] > 0) & (t["cERR"] > 0))
    t["zmuerr"] = t["VPECERR"].astype(float) / CLIGHT
    t["__derived__"] = True
    return t


def quadform(cov, a, b, sig2=0.0, as_float=False):
    """v^T C v with v = (1, a, -b) and C[0,0] += sig2 (COVINT). as_float: C[0,0] rounded to float32 first, as covmat_tot is
    stored. a, b may be scalars or per-event arrays."""
    c00 = cov[0].astype(float) + sig2
    if as_float:
        c00 = c00.astype(F32).astype(float)
    return c00 + a * a * cov[3] + b * b * cov[5] + 2 * a * cov[1] - 2 * b * cov[2] - 2 * a * b * cov[4]


def muerr_z(z, zerr):
    """fcn_muerrz (OPT 1): distance-modulus error from the peculiar-velocity redshift error."""
    return 5.0 / LN10 * (zerr / z) * (1 + z) / (1 + z / 2.0)


def muerrsq_full(cov, a, b, z, zerr, lensing_zpar, sig2=0.0, as_float=False):
    return quadform(cov, a, b, sig2, as_float) + muerr_z(z, zerr) ** 2 + (lensing_zpar * z) ** 2


def gamma_step(g0, host, cen, tau):
    """get_gammadm_host: g0 * (0.5 - 1 / (1 + exp(-(host - cen) / tau)))  (GAMMADM_OFFSET = 0)."""
    with np.errstate(over="ignore"):
        return g0 * (0.5 - 1.0 / (1.0 + np.exp(-(host - cen) / tau)))


def window(v, lo, hi):
    """Closed window on the float32 value, compared in double, as in reject_CUTWIN."""
    v = np.asarray(v, dtype=F32).astype(float)
    return (v >= lo) & (v <= hi)


def host_values(t, varname):
    """The host variable of the step (varname_gamma), with values < -990 replaced by -9 as SALT2mu does."""
    h = np.asarray(t[varname], dtype=F32).astype(float)
    return np.where(h < -990.0, -9.0, h)
