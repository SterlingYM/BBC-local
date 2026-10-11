"""Bias-correction maps of one bias-correction sample (IDSAMPLE): the 5D map of the bias on MU in cells of z x host x x1 x c
(makeMap_binavg_biasCor, makeMap_fitPar_biasCor), its interpolation at an event (get_fitParBias), and the MUCOVSCALE and
MUCOVADD maps (makeMap_sigmu_biasCor, get_muCOVcorr). Thresholds and quirks follow SALT2mu.c.

Population weights (SIM_WGT_POP, "WGT_POP" below) enter exactly where SALT2mu uses them:
- the cell averages AVG_z/m/x1/c and the bias VAL of the 5D map are weighted by WGT_POP / muerr^2;
- the RMS and ERR of the 5D map use WGT_POP sums (WperCell); NperCell stays an unweighted count, and every count threshold
  (min_per_cell_biascor, MINPERCELL_RMS, BIASCOR_MINSUM, MINPERCELL_MUCOVSCALE) compares counts;
- MUCOVSCALE is (1.48 x WGT_POP-weighted median |pull|)^2 (arrayStat_MEDIAN rule) and the MUCOV cell averages are weighted
  by WGT_POP, while MUCOVADD comes from an UNWEIGHTED MAD scan (sigint_muresid_list is called with WGT_IGNORE);
- the sigint of the map weights (SNR >= 60 rows) uses WGT_POP-weighted sums with SALT2mu's STD_from_SUMS(N, ...) quirk.
"""
import numpy as np

from .binning import bininfo, ibin, linear_bins
from .model import F32, muerrsq_full

BIN_X1 = (-4.0, 4.0, 0.5)          # BIASCOR_MINVAL/MAXVAL/BINSIZE_LCFIT for x1 (SALT2)
BIN_C = (-0.3, 0.5, 0.05)          # ... and for c
MINPERCELL_RMS = 10
MINPERCELL_MUCOVSCALE = 5
BIASCOR_MINSUM = 10
NCELL_INTERP_MIN = 3


def std_from_sums(n, s, sq):
    """STD_from_SUMS of sntools.c: sqrt(sq/n - (s/n)^2), 0 if the argument is not positive or n == 0. In the weighted
    arrayStat_STD, s and sq are weighted sums but n is the unweighted COUNT (a SALT2mu quirk reproduced here)."""
    if n == 0:
        return 0.0
    avg = s / n
    arg = sq / n - avg * avg
    return float(np.sqrt(arg)) if arg > 0.0 else 0.0


def gsl_sort_index(data):
    """GSL's gsl_sort_index (in-place heapsort of an index permutation; not stable), as used by sortDouble in sntools.c.
    Pure Python; used only when ties make the result of the weighted median depend on the order of tied elements."""
    data = [float(x) for x in data]
    n = len(data)
    p = list(range(n))
    if n == 0:
        return np.array(p, dtype=np.int64)

    def downheap(N, k):
        pki = p[k]
        while k <= N // 2:
            j = 2 * k
            if j < N and data[p[j]] < data[p[j + 1]]:
                j += 1
            if not (data[pki] < data[p[j]]):
                break
            p[k] = p[j]
            k = j
        p[k] = pki

    N = n - 1
    k = N // 2 + 1
    while True:
        k -= 1
        downheap(N, k)
        if k <= 0:
            break
    while N > 0:
        p[0], p[N] = p[N], p[0]
        N -= 1
        downheap(N, 0)
    return np.array(p, dtype=np.int64)


def weighted_median(vals, wgt=None):
    """arrayStat_MEDIAN of sntools.c. Without weights: the ordinary median. With weights (all >= 0): sort the values, add the
    weights in sorted order until the running sum reaches half the total (sum >= total/2), and call that element imed0 and the
    next one in sorted order imed1; the median is 0.5*(v[imed0] + v[imed1]) for an EVEN number of elements and v[imed0] for an
    odd number (the parity is that of the count N, not of the weights). With unit weights this is the ordinary median.

    The total is summed in input order and the running sum in sorted order, as in the C loops. The sort in SALT2mu is GSL's
    (unstable) heapsort; when tied values carry different weights at the crossing, the order of the tied elements matters,
    and GSL's heapsort is then reproduced exactly. Cases in which SALT2mu aborts or reads past the end of its index array
    (total weight 0; the crossing at the last element with even N) raise RuntimeError."""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n == 0:
        raise RuntimeError("median of an empty list")
    if wgt is None:
        return float(np.median(v))
    w = np.asarray(wgt, dtype=float)
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("population weights must be finite and >= 0")
    total = float(np.cumsum(w)[-1])                       # sequential sum, input order
    half = total / 2.0
    if not half > 0.0:
        raise RuntimeError("weighted median with total weight 0: SALT2mu's arrayStat_MEDIAN aborts here (all SIM_WGT_POP = 0 "
                           "in a cell that passes the count threshold); use with_weights(..., min_weight=0) to drop those rows")

    def crossing(order):
        cs = np.cumsum(w[order])                          # sequential, sorted order
        i = int(np.searchsorted(cs, half, side="left"))   # first i with running sum >= half (cs is non-decreasing)
        if i >= n:
            raise RuntimeError("weighted median: running sum never reaches half the total (rounding); undefined in SALT2mu")
        return i

    order = np.argsort(v, kind="stable")
    i = crossing(order)
    vs = v[order]
    lo = int(np.searchsorted(vs, vs[i], side="left"))
    hi = int(np.searchsorted(vs, vs[i], side="right"))
    if hi - lo > 1 and np.unique(w[order[lo:hi]]).size > 1:
        order = gsl_sort_index(v)                         # tie at the crossing with unequal weights: order matters
        i = crossing(order)
    imed0 = order[i]
    if n % 2 == 1:
        imed1 = imed0
    else:
        if i + 1 >= n:
            raise RuntimeError("weighted median: the crossing is at the last element with an even count; SALT2mu's "
                               "arrayStat_MEDIAN then reads past the end of its index array (undefined result)")
        imed1 = order[i + 1]
    return 0.5 * (v[imed0] + v[imed1])


def _std_pull(dev, mucov, sig, use_mad, wgt=None):
    """STD (or 1.48 MAD) of the pull dev / sqrt(mucov + sig*|sig|), with the 0.02^2 floor for negative sig*|sig|.
    wgt: population weights (arrayStat_STD / arrayStat_MEDIAN with weights)."""
    covt = sig * abs(sig)
    tot = mucov + covt
    if covt < 0:
        tot = np.where(tot < 0.02 ** 2, 0.02 ** 2, tot)
    pull = dev / np.sqrt(tot)
    if wgt is not None:
        if use_mad:
            return 1.48 * weighted_median(np.abs(pull), wgt)
        return std_from_sums(len(pull), np.sum(wgt * pull), np.sum(wgt * pull ** 2))
    if use_mad:
        return 1.48 * np.median(np.abs(pull))
    return np.sqrt(max(np.mean(pull ** 2) - np.mean(pull) ** 2, 0.0))


def sigint_scan(mures, mucov, use_mad, abort=True, bisect=True, wgt=None):
    """sigint_muresid_list: scan sigint downward in steps of 0.01 from an approximate start until the STD (or 1.48 MAD) of the
    pull crosses 1; linear interpolation between the bracketing steps. Negative sigint means sigint*|sigint| < 0 is added.

    The trial values, the one-time upward adjustment (when the first evaluation exceeds 1) and the stopping rule are those of
    SALT2mu. For the MAD (use_mad=True, the MUCOVADD cells) with every mucov >= 0.02^2, each |pull| and hence 1.48 MAD(pull)
    cannot increase with sigint, so the trial value at which the scan stops (the first one above 1) is found by bisection over
    the same sequence of trial values; the result depends only on the bracketing pair and is identical to that of the
    step-by-step scan. STD(pull) has no such guarantee (the mean pull also changes), so use_mad=False always scans step by
    step, as does bisect=False.

    wgt: population weights (OPTMASK & 2 of sigint_muresid_list). The averages of mucov and mures are then weighted means;
    STD uses weighted sums divided by the COUNT N (STD_from_SUMS quirk); the MAD uses the weighted median. A weighted scan
    is always done step by step."""
    mures = np.asarray(mures, float)
    mucov = np.asarray(mucov, float)
    if wgt is not None:
        wgt = np.asarray(wgt, float)
        sw = np.sum(wgt)
        avg_cov = np.sum(wgt * mucov) / sw
        avg_res = np.sum(wgt * mures) / sw
        if use_mad:
            std_orig = 1.48 * weighted_median(np.abs(mures), wgt)
        else:
            std_orig = std_from_sums(len(mures), np.sum(wgt * mures), np.sum(wgt * mures ** 2))
        bisect = False
    else:
        avg_cov = np.mean(mucov)
        avg_res = np.mean(mures)
        if use_mad:
            std_orig = 1.48 * np.median(np.abs(mures))
        else:
            std_orig = np.sqrt(max(np.mean(mures ** 2) - np.mean(mures) ** 2, 0.0))
    dev = mures - avg_res
    tmp = std_orig ** 2 - avg_cov
    approx = -np.sqrt(abs(tmp)) if tmp < 0 else np.sqrt(tmp)
    lo = approx - 50 * 0.01 - 1e-7
    sig = approx + 60 * 0.01
    sp = _std_pull(dev, mucov, sig, use_mad, wgt)
    if sp > 1.0:
        sig += (sp - 1.0) + 0.1            # one-time upward adjustment; this first evaluation is not stored
    seq = []                               # trial values, by repeated subtraction as in SALT2mu
    while sig >= lo:
        seq.append(sig)
        sig -= 0.01
    # monotonic only with the MAD and no 0.02^2 floor jump at sigint = 0 (floor applies only for negative sigint)
    if bisect and use_mad and mucov.min() >= 0.02 ** 2:
        if _std_pull(dev, mucov, seq[-1], use_mad, wgt) <= 1.0:
            stop = None
        else:
            j0, j1 = -1, len(seq) - 1      # MAD > 1 at seq[j1]; <= 1 at seq[j0] (or j0 = -1)
            while j1 - j0 > 1:
                mid = (j0 + j1) // 2
                if _std_pull(dev, mucov, seq[mid], use_mad, wgt) > 1.0:
                    j1 = mid
                else:
                    j0 = mid
            stop = j1
    else:
        stop = None
        for j, s in enumerate(seq):
            if _std_pull(dev, mucov, s, use_mad, wgt) > 1.0:
                stop = j
                break
    if stop is None:
        if abort:
            raise RuntimeError("sigint scan did not bracket STD(pull) = 1")
        return lo
    sig_store = seq[max(stop - 1, 0):stop + 1]
    pull_store = [_std_pull(dev, mucov, s, use_mad, wgt) for s in sig_store]
    if len(sig_store) == 1 and abs(pull_store[0] - 1.0) < 0.001:
        return sig_store[0]
    return float(np.interp(1.0, np.array(pull_store), np.array(sig_store)))   # pull increases as sig decreases


class SampleMaps:
    """Maps of one sample. sims: dict of derived bias-correction arrays (all rows); rows: indices of this sample's rows that
    pass every cut. a_sim, b_sim: alpha, beta used for the cell weights and for mu of the simulations. wpop: population
    weights of all rows (float64 of the float32 SIM_WGT_POP), or None for unit weights."""

    def __init__(self, zgrid, mgrid, sims, rows, sig2int, a_sim, b_sim, cfg, zmax_fit, wpop=None):
        self.Bz, self.Bm = zgrid, mgrid
        self.Bs, self.Bc = bininfo(*BIN_X1), bininfo(*BIN_C)
        self.a_sim, self.b_sim, self.lens = a_sim, b_sim, cfg.lensing_zpar
        self.min_per_cell = cfg.min_per_cell_biascor
        self.zmax_fit = zmax_fit
        z = sims["zHD"][rows].astype(float)
        m = sims["__HOST__"][rows]
        s = sims["x1"][rows].astype(float)
        c = sims["c"][rows].astype(float)
        iz, im, is_, ic = ibin(z, self.Bz), ibin(m, self.Bm), ibin(s, self.Bs), ibin(c, self.Bc)
        good = (iz >= 0) & (im >= 0) & (is_ >= 0) & (ic >= 0)                     # outside_biasCor_grid
        rows = rows[good]
        z, m, s, c = z[good], m[good], s[good], c[good]
        iz, im, is_, ic = iz[good], im[good], is_[good], ic[good]
        self.shape = (self.Bz.n, self.Bm.n, self.Bs.n, self.Bc.n)
        J = np.ravel_multi_index((iz, im, is_, ic), self.shape)
        NC = int(np.prod(self.shape))
        w = 1.0 / muerrsq_full(sims["cov"][:, rows], a_sim, b_sim, z, sims["zmuerr"][rows], self.lens, sig2int)
        self.wpop = wpop
        if wpop is not None:
            wp = wpop[rows]
            w = wp * w                                    # WGT_biasCor: WGT_POP * (1/muerrsq)
            if len(rows) and not (w > 0).any():
                raise RuntimeError("all bias-correction weights are 0 in a sample (SALT2mu's makeMap_binavg_biasCor aborts)")
        sw = np.bincount(J, w, NC)
        self.N = np.bincount(J, minlength=NC)            # NperCell: an unweighted count in SALT2mu, also with weights
        if wpop is not None:
            dead = (self.N >= self.min_per_cell) & (sw == 0)
            if dead.any():
                raise RuntimeError(
                    f"{int(dead.sum())} bias-map cell(s) pass min_per_cell_biascor by count but have total weight 0; SALT2mu "
                    "would interpolate a 0/0 (NaN) bias there. Use with_weights(..., min_weight=0) to drop zero-weight rows")
        with np.errstate(invalid="ignore", divide="ignore"):
            self.avg = {k: np.where(sw >= 1e-9, np.bincount(J, w * v, NC) / sw, 9999.0)
                        for k, v in (("z", z), ("m", m), ("s", s), ("c", c))}
            bias = sims["fitmu"][rows].astype(float) - sims["SIM_MUz"][rows].astype(float)
            self.val = np.where(self.N >= 1, np.bincount(J, w * bias, NC) / sw, 999.0)
        self.rows, self.J = rows, J
        self.nsim = len(rows)
        self._sims = sims

    def bias_rms_err(self):
        """RMS and ERR of the bias in each 5D cell, as makeMap_fitPar_biasCor computes them (diagnostics only: SALT2mu sets
        the interpolated muBiasErr to 0 before the fit, so they do not affect the results). Sums are weighted by WGT_POP
        only (not 1/muerr^2); for a cell with fewer than MINPERCELL_RMS events (count), the sums run over the 3^4 neighbour
        cells in (z, host, x1, c). RMS = sqrt|sum w b^2 / W - (sum w b / W)^2| with W the summed WGT_POP over those cells;
        ERR = RMS / sqrt(WperCell of the cell itself), 1e-12 if 0. Cells with no event: 999."""
        S, rows, J = self._sims, self.rows, self.J
        NC = int(np.prod(self.shape))
        wp = np.ones(len(rows)) if self.wpop is None else self.wpop[rows]
        b = S["fitmu"][rows].astype(float) - S["SIM_MUz"][rows].astype(float)
        s1 = np.bincount(J, wp * b, NC).reshape(self.shape)
        s2 = np.bincount(J, wp * b * b, NC).reshape(self.shape)
        W = np.bincount(J, wp, NC).reshape(self.shape)
        N = self.N.reshape(self.shape)

        def box(a):
            p = np.pad(a, 1)
            out = np.zeros_like(a)
            for off in np.ndindex(3, 3, 3, 3):
                out += p[off[0]:off[0] + a.shape[0], off[1]:off[1] + a.shape[1], off[2]:off[2] + a.shape[2],
                         off[3]:off[3] + a.shape[3]]
            return out
        few = N < MINPERCELL_RMS
        S1 = np.where(few, box(s1), s1)
        S2 = np.where(few, box(s2), s2)
        WL = np.where(few, box(W), W)
        with np.errstate(invalid="ignore", divide="ignore"):
            t1 = S1 / WL
            rms = np.sqrt(np.abs(S2 / WL - t1 * t1))
            err = rms / np.sqrt(W)
        err = np.where(err == 0.0, 1e-12, err)
        rms = np.where(N >= 1, rms, 999.0).ravel()
        err = np.where(N >= 1, err, 999.0).ravel()
        return rms, err

    def interp(self, z, m, s, c):
        """get_fitParBias for many events; returns (bias, status, summed N). status > 0 means success."""
        n = len(z)
        Bz, Bs, Bc = self.Bz, self.Bs, self.Bc
        IZ, IM, IS, IC = ibin(z, Bz), ibin(m, self.Bm), ibin(s, Bs), ibin(c, Bc)
        status = np.ones(n, int)
        status[IC < 0] = -4
        status[IS < 0] = -3
        status[IM < 0] = -2
        status[IZ < 0] = -1
        okb = status > 0
        IZc, IMc, ISc, ICc = [np.where(okb, x, 0) for x in (IZ, IM, IS, IC)]
        Jc = np.ravel_multi_index((IZc, IMc, ISc, ICc), self.shape)
        A = {k: v[Jc] for k, v in self.avg.items()}

        def nbr(I, val, avgc, nb):
            hi = np.where(val >= avgc, I + 1, I)
            lo = np.where(val >= avgc, I, I - 1)
            return np.clip(lo, 0, nb - 1), np.clip(hi, 0, nb - 1)

        zlo, zhi = nbr(IZc, z, A["z"], Bz.n)
        slo, shi = nbr(ISc, s, A["s"], Bs.n)
        clo, chi = nbr(ICc, c, A["c"], Bc.n)
        bz = np.full(n, Bz.bin)
        bs = np.full(n, Bs.bin)
        bc = np.full(n, Bc.bin)
        cells = []
        for zz in (0, 1):
            for ss in (0, 1):
                for cc in (0, 1):
                    iz = zlo if zz == 0 else zhi
                    is_ = slo if ss == 0 else shi
                    ic = clo if cc == 0 else chi
                    dup = ((zz == 1) & (zhi == zlo)) | ((ss == 1) & (shi == slo)) | ((cc == 1) & (chi == clo))
                    j = np.ravel_multi_index((iz, IMc, is_, ic), self.shape)
                    use = (~dup) & (self.N[j] >= self.min_per_cell) & okb
                    center = (iz == IZc) & (is_ == ISc) & (ic == ICc)
                    a = {k: v[j] for k, v in self.avg.items()}
                    valid_avg = (a["z"] <= 9000) & (a["m"] <= 9000) & (a["s"] <= 9000) & (a["c"] <= 9000)
                    nb = use & ~center & valid_avg
                    # effective bin size = max(nominal, |AVG_centre - AVG_neighbour|) over usable neighbours
                    bz = np.where(nb & (iz != IZc), np.maximum(bz, np.abs(A["z"] - a["z"])), bz)
                    bs = np.where(nb & (is_ != ISc), np.maximum(bs, np.abs(A["s"] - a["s"])), bs)
                    bc = np.where(nb & (ic != ICc), np.maximum(bc, np.abs(A["c"] - a["c"])), bc)
                    cells.append((iz, j, use, center, a))
        bad = (bz == 0) | (bz > 9000) | (bs == 0) | (bs > 9000) | (bc == 0) | (bc > 9000)
        status[okb & bad] = -8
        SW = np.zeros(n)
        SV = np.zeros(n)
        NS = np.zeros(n, int)
        NU = np.zeros(n, int)
        usec = np.zeros(n, bool)
        izmax = np.maximum(zlo, zhi)
        usez_max = np.zeros(n, int)
        for iz, j, use, center, a in cells:
            with np.errstate(invalid="ignore"):
                W = (1 - np.abs((z - a["z"]) / bz)) * (1 - np.abs((s - a["s"]) / bs)) * (1 - np.abs((c - a["c"]) / bc))
            W = np.where(W < 0, 0.0, W)                  # only the product is clipped, not each factor
            W = np.where(use, W, 0.0)
            SW += W
            SV += W * np.where(use, self.val[j], 0.0)
            NS += np.where(use, self.N[j], 0)
            NU += use
            usec |= use & center
            usez_max += use & (iz == izmax)               # only the IZMAX bin is effectively tested (USEZ quirk)
        st = status > 0
        status[st & (NU < NCELL_INTERP_MIN)] = -9
        st = status > 0
        status[st & ~usec] = -10
        st = status > 0
        status[st & (usez_max == 0)] = -11
        st = status > 0
        okN = (NS >= BIASCOR_MINSUM) | ((NS >= 2) & (z > self.zmax_fit))
        status[st & ~okN] = -12
        st = status > 0
        status[st & (SW <= 1e-9)] = -13
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(status > 0, SV / SW, np.nan)
        return val, status, NS

    def make_mucov(self, sims, cfg):
        """makeMap_sigmu_biasCor: per coarse (z, host, c) cell with >= 5 simulated events, MUCOVSCALE = (1.48 MAD(pull))^2 and
        MUCOVADD = s|s| with s from the MAD sigint scan, both stored as float32."""
        self.Cz = self.Bz
        if cfg.nbin_logmass == 1:
            self.Cm = linear_bins(-15.0, 15.0, 1)                       # legacy single host bin
        else:
            self.Cm = linear_bins(cfg.logmass_min, cfg.logmass_max, 2)
        self.Cc = linear_bins(cfg.cmin, cfg.cmax, cfg.nbinc_mucovscale)
        rows = self.rows[self.N[self.J] >= self.min_per_cell]
        z = sims["zHD"][rows].astype(float)
        m = sims["__HOST__"][rows]
        s = sims["x1"][rows].astype(float)
        c = sims["c"][rows].astype(float)
        mub, st, _ = self.interp(z, m, s, c)
        ok = st > 0
        rows, z, m, c, mub = rows[ok], z[ok], m[ok], c[ok], mub[ok]
        wp = None if self.wpop is None else self.wpop[rows]
        mudif = sims["muresid"][rows] - mub
        merr2 = muerrsq_full(sims["cov"][:, rows], self.a_sim, self.b_sim, z, sims["zmuerr"][rows], self.lens)
        pull = mudif / np.sqrt(merr2)
        iz = ibin(z, self.Cz, 0)
        im = ibin(m, self.Cm, 2)
        ic = ibin(c, self.Cc, 2)
        inz = iz >= 0
        iz, im, ic, mudif, merr2, pull, z, m, c = (x[inz] for x in (iz, im, ic, mudif, merr2, pull, z, m, c))
        if wp is not None:
            wp = wp[inz]
        shp = (self.Cz.n, self.Cm.n, self.Cc.n)
        NC = int(np.prod(shp))
        K = np.ravel_multi_index((iz, im, ic), shp)
        self.cN = np.bincount(K, minlength=NC)
        self.cuse = self.cN >= MINPERCELL_MUCOVSCALE
        with np.errstate(invalid="ignore", divide="ignore"):
            if wp is None:
                self.cavg = {k: np.where(self.cuse, np.bincount(K, v, NC) / self.cN, 9999.0)
                             for k, v in (("z", z), ("m", m), ("c", c))}
            else:                                          # WGT_POP-weighted averages, divided by WperCell
                self.cW = np.bincount(K, wp, NC)
                self.cavg = {k: np.where(self.cuse, np.bincount(K, v * wp, NC) / self.cW, 9999.0)
                             for k, v in (("z", z), ("m", m), ("c", c))}
        self.cscale = np.ones(NC, F32)
        self.cadd = np.full(NC, 1e-12, F32)
        order = np.argsort(K, kind="stable")
        starts = np.searchsorted(K[order], np.arange(NC + 1))
        for k in np.nonzero(self.cuse)[0]:
            idx = order[starts[k]:starts[k + 1]]
            if wp is None:
                mad = 1.48 * float(np.median(np.abs(pull[idx])))       # unit weights: ordinary median
            else:
                mad = 1.48 * float(weighted_median(np.abs(pull[idx]), wp[idx]))
            self.cscale[k] = F32(mad * mad)
            # MUCOVADD: SALT2mu passes WGT_IGNORE here, so this scan is unweighted even with population weights
            si = sigint_scan(mudif[idx], merr2[idx], use_mad=True, abort=False)
            if si == 0.0:
                si = 1e-12
            self.cadd[k] = F32(si * abs(si))
        self.cshape = shp

    def mucov(self, z, m, c):
        """get_muCOVcorr: interpolation over the 3x3 (z, c) neighbours, host bin frozen; returns (muCOVadd, muCOVscale, ok)."""
        n = len(z)
        IZ = ibin(z, self.Cz, 0)
        IM = ibin(m, self.Cm, 2)
        IC = ibin(c, self.Cc, 2)
        SW = np.zeros(n)
        SA = np.zeros(n)
        SS = np.zeros(n)
        cen = np.zeros(n, bool)
        for dz in (-1, 0, 1):
            for dc in (-1, 0, 1):
                iz = IZ + dz
                ic = IC + dc
                inr = (IZ >= 0) & (iz >= 0) & (iz < self.Cz.n) & (ic >= 0) & (ic < self.Cc.n)
                k = np.ravel_multi_index((np.clip(iz, 0, self.Cz.n - 1), IM, np.clip(ic, 0, self.Cc.n - 1)), self.cshape)
                use = inr & self.cuse[k]
                Dz = np.abs((z - self.cavg["z"][k]) / self.Cz.bin)
                Dc = np.abs((c - self.cavg["c"][k]) / self.Cc.bin)
                use &= (Dz <= 0.99) & (Dc <= 0.99)
                W = np.where(use, (1 - Dz) * (1 - Dc), 0.0)
                SW += W
                SA += W * self.cadd[k].astype(float)
                SS += W * self.cscale[k].astype(float)
                cen |= use & (dz == 0) & (dc == 0)
        ok = (SW > 0.01) & cen
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ok, SA / SW, 0.0), np.where(ok, SS / SW, 1.0), ok
