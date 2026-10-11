"""Bias-correction maps of one bias-correction sample (IDSAMPLE): the 5D map of the bias on MU in cells of z x host x x1 x c
(makeMap_binavg_biasCor, makeMap_fitPar_biasCor), its interpolation at an event (get_fitParBias), and the MUCOVSCALE and
MUCOVADD maps (makeMap_sigmu_biasCor, get_muCOVcorr). Thresholds and quirks follow SALT2mu.c."""
import numpy as np

from .binning import bininfo, ibin, linear_bins
from .model import F32, muerrsq_full

BIN_X1 = (-4.0, 4.0, 0.5)          # BIASCOR_MINVAL/MAXVAL/BINSIZE_LCFIT for x1 (SALT2)
BIN_C = (-0.3, 0.5, 0.05)          # ... and for c
MINPERCELL_RMS = 10
MINPERCELL_MUCOVSCALE = 5
BIASCOR_MINSUM = 10
NCELL_INTERP_MIN = 3


def _std_pull(dev, mucov, sig, use_mad):
    """STD (or 1.48 MAD) of the pull dev / sqrt(mucov + sig*|sig|), with the 0.02^2 floor for negative sig*|sig|."""
    covt = sig * abs(sig)
    tot = mucov + covt
    if covt < 0:
        tot = np.where(tot < 0.02 ** 2, 0.02 ** 2, tot)
    pull = dev / np.sqrt(tot)
    if use_mad:
        return 1.48 * np.median(np.abs(pull))
    return np.sqrt(max(np.mean(pull ** 2) - np.mean(pull) ** 2, 0.0))


def sigint_scan(mures, mucov, use_mad, abort=True, bisect=True):
    """sigint_muresid_list: scan sigint downward in steps of 0.01 from an approximate start until the STD (or 1.48 MAD) of the
    pull crosses 1; linear interpolation between the bracketing steps. Negative sigint means sigint*|sigint| < 0 is added.

    The trial values, the one-time upward adjustment (when the first evaluation exceeds 1) and the stopping rule are those of
    SALT2mu. For the MAD (use_mad=True, the MUCOVADD cells) with every mucov >= 0.02^2, each |pull| and hence 1.48 MAD(pull)
    cannot increase with sigint, so the trial value at which the scan stops (the first one above 1) is found by bisection over
    the same sequence of trial values; the result depends only on the bracketing pair and is identical to that of the
    step-by-step scan. STD(pull) has no such guarantee (the mean pull also changes), so use_mad=False always scans step by
    step, as does bisect=False."""
    mures = np.asarray(mures, float)
    mucov = np.asarray(mucov, float)
    avg_cov = np.mean(mucov)
    avg_res = np.mean(mures)
    dev = mures - avg_res
    if use_mad:
        std_orig = 1.48 * np.median(np.abs(mures))
    else:
        std_orig = np.sqrt(max(np.mean(mures ** 2) - np.mean(mures) ** 2, 0.0))
    tmp = std_orig ** 2 - avg_cov
    approx = -np.sqrt(abs(tmp)) if tmp < 0 else np.sqrt(tmp)
    lo = approx - 50 * 0.01 - 1e-7
    sig = approx + 60 * 0.01
    sp = _std_pull(dev, mucov, sig, use_mad)
    if sp > 1.0:
        sig += (sp - 1.0) + 0.1            # one-time upward adjustment; this first evaluation is not stored
    seq = []                               # trial values, by repeated subtraction as in SALT2mu
    while sig >= lo:
        seq.append(sig)
        sig -= 0.01
    # monotonic only with the MAD and no 0.02^2 floor jump at sigint = 0 (floor applies only for negative sigint)
    if bisect and use_mad and mucov.min() >= 0.02 ** 2:
        if _std_pull(dev, mucov, seq[-1], use_mad) <= 1.0:
            stop = None
        else:
            j0, j1 = -1, len(seq) - 1      # MAD > 1 at seq[j1]; <= 1 at seq[j0] (or j0 = -1)
            while j1 - j0 > 1:
                mid = (j0 + j1) // 2
                if _std_pull(dev, mucov, seq[mid], use_mad) > 1.0:
                    j1 = mid
                else:
                    j0 = mid
            stop = j1
    else:
        stop = None
        for j, s in enumerate(seq):
            if _std_pull(dev, mucov, s, use_mad) > 1.0:
                stop = j
                break
    if stop is None:
        if abort:
            raise RuntimeError("sigint scan did not bracket STD(pull) = 1")
        return lo
    sig_store = seq[max(stop - 1, 0):stop + 1]
    pull_store = [_std_pull(dev, mucov, s, use_mad) for s in sig_store]
    if len(sig_store) == 1 and abs(pull_store[0] - 1.0) < 0.001:
        return sig_store[0]
    return float(np.interp(1.0, np.array(pull_store), np.array(sig_store)))   # pull increases as sig decreases


class SampleMaps:
    """Maps of one sample. sims: dict of derived bias-correction arrays (all rows); rows: indices of this sample's rows that
    pass every cut. a_sim, b_sim: alpha, beta used for the cell weights and for mu of the simulations."""

    def __init__(self, zgrid, mgrid, sims, rows, sig2int, a_sim, b_sim, cfg, zmax_fit):
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
        sw = np.bincount(J, w, NC)
        self.N = np.bincount(J, minlength=NC)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.avg = {k: np.where(sw >= 1e-9, np.bincount(J, w * v, NC) / sw, 9999.0)
                        for k, v in (("z", z), ("m", m), ("s", s), ("c", c))}
            bias = sims["fitmu"][rows].astype(float) - sims["SIM_MUz"][rows].astype(float)
            self.val = np.where(self.N >= 1, np.bincount(J, w * bias, NC) / sw, 999.0)
        self.rows, self.J = rows, J
        self.nsim = len(rows)

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
        mudif = sims["muresid"][rows] - mub
        merr2 = muerrsq_full(sims["cov"][:, rows], self.a_sim, self.b_sim, z, sims["zmuerr"][rows], self.lens)
        pull = mudif / np.sqrt(merr2)
        iz = ibin(z, self.Cz, 0)
        im = ibin(m, self.Cm, 2)
        ic = ibin(c, self.Cc, 2)
        inz = iz >= 0
        iz, im, ic, mudif, merr2, pull, z, m, c = (x[inz] for x in (iz, im, ic, mudif, merr2, pull, z, m, c))
        shp = (self.Cz.n, self.Cm.n, self.Cc.n)
        NC = int(np.prod(shp))
        K = np.ravel_multi_index((iz, im, ic), shp)
        self.cN = np.bincount(K, minlength=NC)
        self.cuse = self.cN >= MINPERCELL_MUCOVSCALE
        with np.errstate(invalid="ignore", divide="ignore"):
            self.cavg = {k: np.where(self.cuse, np.bincount(K, v, NC) / self.cN, 9999.0) for k, v in (("z", z), ("m", m), ("c", c))}
        self.cscale = np.ones(NC, F32)
        self.cadd = np.full(NC, 1e-12, F32)
        order = np.argsort(K, kind="stable")
        starts = np.searchsorted(K[order], np.arange(NC + 1))
        for k in np.nonzero(self.cuse)[0]:
            idx = order[starts[k]:starts[k + 1]]
            mad = 1.48 * float(np.median(np.abs(pull[idx])))           # unit weights: ordinary median
            self.cscale[k] = F32(mad * mad)
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
