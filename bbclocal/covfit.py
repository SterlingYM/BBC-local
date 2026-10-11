"""Distance covariance and a wfit-like w fit, for quick design iteration on BBC outputs.

Covariance (unbinned Hubble diagram, as SNANA's create_covariance builds it from systematic BBC variants without EXTRA_COVS):

    C = diag(MUERR^2) + sum_S W_S^2 dmu_S dmu_S^T + OFFAMP^2 sum_k 1_k 1_k^T

where dmu_S = MU_S - MU_nominal for systematic S (same SNe, same order), W_S its weight, and 1_k the indicator of group k (an
optional free magnitude offset per group, e.g. per population). EXTRA_COVS (full external matrices) are not supported. The
low-rank part is handled with the Woodbury identity, so a full grid costs O(N_grid x N_SN x rank).

w fit (as wfit with its grid settings): flat wCDM, D_L = (1 + zHD) D_C(zHD) (wfit's convention), Omega_r = 0.9e-4, absolute
magnitude marginalized analytically with a flat prior, Gaussian prior on Omega_m; w and its uncertainty are the mean and
standard deviation of the marginalized posterior of w on the grid.
"""
import numpy as np

CLIGHT = 299792.458


def _dc_grid(z, om_list, w_list, H0=70.0, omr=0.9e-4, nz=3000):
    """Comoving distance for every (Om, w) at the SNe' zHD: array [n_om, n_w, n_sn] (trapezoid on a uniform grid)."""
    zmax = float(np.max(z)) * 1.0001
    zg = np.linspace(0.0, zmax, nz)
    x = np.asarray(z, float) / (zg[1] - zg[0])
    i0 = np.clip(x.astype(int), 0, nz - 2)
    fr = x - i0
    w = np.asarray(w_list)[:, None]
    out = np.empty((len(om_list), len(w_list), len(z)))
    for i, om in enumerate(om_list):
        E = np.sqrt(om * (1 + zg) ** 3 + omr * (1 + zg) ** 4 + (1.0 - om - omr) * (1 + zg) ** (3 * (1 + w)))
        f = 1.0 / E
        cum = np.concatenate([np.zeros((len(w_list), 1)), np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * np.diff(zg), axis=-1)], -1)
        out[i] = cum[:, i0] * (1 - fr) + cum[:, i0 + 1] * fr
    return out * CLIGHT / H0


class LowRankCov:
    """C = diag(d) + U U^T, solved with the Woodbury identity."""

    def __init__(self, d, U=None):
        self.d = np.asarray(d, float)
        self.Di = 1.0 / self.d
        self.U = np.zeros((len(self.d), 0)) if U is None or len(U) == 0 else np.column_stack(U)
        self.DiU = self.Di[:, None] * self.U
        self.M = np.eye(self.U.shape[1]) + self.U.T @ self.DiU
        self.Minv = np.linalg.inv(self.M) if self.U.shape[1] else None

    def quad(self, R, Y=None):
        """R^T C^-1 Y for R, Y of shape [..., N] (Y defaults to R)."""
        Y = R if Y is None else Y
        a = np.einsum("...n,n,...n->...", R, self.Di, Y)
        if self.U.shape[1] == 0:
            return a
        return a - np.einsum("...k,...k->...", (R @ self.DiU) @ self.Minv, Y @ self.DiU)

    def dense(self):
        return np.diag(self.d) + self.U @ self.U.T


def build_covariance(muerr, sysvecs=(), weights=None, groups=None, offamp=1.0):
    """Covariance of an unbinned Hubble diagram: diag(muerr^2) + sum_S (W_S dmu_S)(W_S dmu_S)^T (+ free offsets per group)."""
    U = []
    weights = np.ones(len(sysvecs)) if weights is None else np.asarray(weights, float)
    for wS, v in zip(weights, sysvecs):
        U.append(wS * np.asarray(v, float))
    if groups is not None:
        g = np.asarray(groups)
        for k in np.unique(g):
            U.append(offamp * (g == k).astype(float))
    return LowRankCov(np.asarray(muerr, float) ** 2, U)


_MODEL_CACHE = {}


def wfit_grid(zhd, mu, muerr, sysvecs=(), groups=None, offamp=1.0, weights=None, ompri=0.315, dompri=0.007,
              om_range=(0.2, 0.45, 126), w_range=(-1.6, -0.4, 481), H0=70.0, omr=0.9e-4):
    """w fit on a grid. sysvecs: list of dmu vectors (systematic minus nominal, same SN order); groups: optional group label per
    SN for free offsets. Returns w (posterior mean), wsig_marg (posterior std), om (posterior mean) and chi2_min."""
    zhd = np.asarray(zhd, float)
    mu = np.asarray(mu, float)
    C = build_covariance(muerr, sysvecs, weights, groups, offamp)
    om = np.linspace(*om_range[:2], int(om_range[2]))
    w = np.linspace(*w_range[:2], int(w_range[2]))
    key = (zhd.tobytes(), tuple(om_range), tuple(w_range), H0, omr)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = 5 * np.log10((1 + zhd) * _dc_grid(zhd, om, w, H0, omr)) + 25
    mod = _MODEL_CACHE[key]
    R = mu[None, None, :] - mod
    one = np.ones_like(mu)
    a = C.quad(one)
    bR = C.quad(R, np.broadcast_to(one, R.shape))
    chi2 = C.quad(R) - bR ** 2 / a
    lnP = -0.5 * chi2 - 0.5 * ((om[:, None] - ompri) / dompri) ** 2
    P = np.exp(lnP - lnP.max())
    Pw = P.sum(axis=0)
    Pw /= Pw.sum()
    wm = np.sum(Pw * w)
    ws = np.sqrt(np.sum(Pw * (w - wm) ** 2))
    Po = P.sum(axis=1)
    Po /= Po.sum()
    return {"w": float(wm), "wsig_marg": float(ws), "om": float(np.sum(Po * om)), "chi2_min": float(chi2.min())}
