"""Binning rules of SALT2mu (setup_BININFO_biasCor, IBINFUN, setup_BININFO_powz), reproduced exactly.

Bin edges are built by cumulative floating-point addition (not vmin + k * vbin), a value exactly on an edge belongs to the
LOWER bin, and the upper edge of the last bin is padded by 1e-10.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Bins:
    lo: np.ndarray
    hi: np.ndarray
    bin: float | None

    @property
    def n(self):
        return len(self.lo)


def bininfo(vmin, vmax, vbin):
    """setup_BININFO_biasCor: for (lo = vmin; lo <= vmax - 0.99 * vbin; lo += vbin)."""
    lo, los = vmin, []
    lomax = vmax - 0.99 * vbin
    while lo <= lomax:
        los.append(lo)
        lo += vbin
    lo_ = np.array(los, dtype=float)
    return Bins(lo_, lo_ + vbin, vbin)


def linear_bins(vmin, vmax, nbin):
    """Edges vmin + k * (vmax - vmin) / nbin (as in malloc_MUCOV for the coarse color bins)."""
    b = (vmax - vmin) / float(nbin)
    lo = np.array([vmin + b * k for k in range(nbin)], dtype=float)
    return Bins(lo, lo + b, b)


def ibin(val, B, opt=0):
    """IBINFUN: closed intervals scanned from bin 0, so an edge value goes to the lower bin; last hi padded by 1e-10.
    opt=0: -9 outside the range; opt=2: clamp to the first/last bin."""
    val = np.asarray(val, dtype=float)
    hi = B.hi.copy()
    hi[-1] += 1e-10
    k = np.searchsorted(hi, val, side="left")
    if opt == 2:
        return np.clip(k, 0, B.n - 1)
    ok = (val >= B.lo[0]) & (k < B.n)
    return np.where(ok, k, -9)


def powz_bins(zmin, zmax, nzbin, powzbin):
    """setup_BININFO_powz: nzbin bins with equal shares of int dz / (1+z)^powzbin, edges built recursively."""
    if powzbin == 1.0:
        raise ValueError("powzbin=1 is not supported")
    p1 = 1.0 - powzbin
    z1 = (((1 + zmax) ** p1 - (1 + zmin) ** p1) / p1) / nzbin
    lo, los, his = zmin, [], []
    for _ in range(nzbin):
        hi = (p1 * z1 + (1 + lo) ** p1) ** (1 / p1) - 1
        los.append(lo)
        his.append(hi)
        lo = hi
    return Bins(np.array(los), np.array(his), None)


def biascor_zgrid(zdata, zbin):
    """set_BINSIZE_SAMPLE_biasCor + setup_BININFO_biasCor for one sample: the redshift grid of the bias-correction map depends
    on the smallest and largest (float32) zHD of the sample's DATA rows, before any cut."""
    zz = np.asarray(zdata, dtype=np.float32).astype(float)
    r0 = 0.01 * int(100.0 * zz.min())
    nbz = int((zz.max() - r0) / zbin) + 1
    r1 = r0 + zbin * nbz
    return bininfo(r0, r1 + zbin, zbin)
