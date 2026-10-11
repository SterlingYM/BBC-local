"""The BBC fit (SALT2mu) for the supported configuration: bias-correction samples and cuts, 5D bias maps and MUCOV maps, the
per-event bias and error corrections, the optional chi2max outlier cut, and the fit of alpha, beta, gamma, one M0 per redshift
bin and sigint.

Differences from SALT2mu that remain by design (see README): the minimizer (M0 profiled analytically, alpha/beta/gamma
minimized to a tight tolerance instead of MINUIT/MIGRAD with EDM < 1e-4), and the distance integral (accurate instead of
Romberg at 1e-6).
"""
import warnings

import numpy as np
from scipy import optimize

from . import __version__
from .binning import biascor_zgrid, bininfo, ibin, powz_bins
from .config import UnsupportedOptionError
from .cosmo import Cosmology
from .io import BIASCOR_COLS, DATA_COLS, read_biascor, read_cid_list
from .maps import SampleMaps, sigint_scan
from .model import F32, M0_DEFAULT, derive, gamma_step, host_values, muerr_z, muerrsq_full, quadform, window
from .samples import SampleSet

PAR_BOUNDS = {"alpha": (0.02, 0.30), "beta": (1.0, 6.0), "gamma": (-0.5, 0.5)}     # MINUIT bounds of SALT2mu (prep_fitpar)
MAXFITITER = 20
SNR_SIGINT = 60.0
MIN_SIGINT_EVENTS = 50
SIM_CACHE_MAX = 8      # cached sample assignments / pre-cut row sets per BiasCorSims object

# bbclocal's own cut flags (not SALT2mu's CUTMASK bit values)
CUT_SELECTION = 1      # CUTWIN, legacy windows (z, x1, c, host), BADERR, BADCOV
CUT_CID_REJECT = 2     # cid_reject list
CUT_BIASCOR = 4        # no valid bias correction or MUCOV correction (SALT2mu's "valid BIASCOR")
CUT_CHI2MAX = 8        # chi2max outlier cut


def cosmology_from_config(cfg):
    return Cosmology(OL=cfg.p[9], w0=cfg.p[11], H0=cfg.h0, Ok=cfg.p[10], wa=cfg.p[12])


def sim_key(cfg):
    """The configuration items that the per-row derived quantities of the simulations depend on."""
    return (cfg.varname_gamma, cfg.p[1], cfg.p[2], cfg.p[5], cfg.p[7], cfg.p[8], cfg.use_gamma0, cfg.p[9], cfg.p[11], cfg.h0)


def _sim_windows(cfg):
    """var -> (lo, hi) applied to the simulations (intersection of CUTWINs and legacy windows); zHD upper edge excluded."""
    w = {}

    def add(v, lo, hi):
        a, b = w.get(v, (-np.inf, np.inf))
        w[v] = (max(a, lo), min(b, hi))
    for cw in cfg.cutwins:
        if not cw.dataonly:
            add(cw.var, cw.lo, cw.hi)
    add("x1", cfg.x1min, cfg.x1max)
    add("c", cfg.cmin, cfg.cmax)
    add("zHD", cfg.zmin, np.inf)
    return w


class BiasCorSims:
    """Bias-correction simulations of one model (all files of simfile_biascor), with the per-row quantities SALT2mu derives
    from them. These do not depend on the data or on the population cuts, so one object can serve many fits.

    dust_flag: True/False to override SALT2mu's dust detection (needed for npz files without bbclocal metadata).
    trust_legacy: accept npz files without bbclocal metadata."""

    def __init__(self, paths, cfg, dust_flag=None, trust_legacy=False):
        if isinstance(paths, (str, bytes)):
            paths = [paths]
        self.paths = [str(p) for p in paths]
        need = set(BIASCOR_COLS) | {cfg.varname_gamma} | {c.var for c in cfg.cutwins if not c.dataonly}
        parts, metas = [], []
        for p in self.paths:
            t, meta = read_biascor(p, trust_legacy=trust_legacy, extra_cols=need)
            miss = [c for c in need if c not in t and c not in ("SIM_AV", "FIELD")]
            if miss:
                raise KeyError(f"{p}: bias-correction file lacks columns {miss}")
            parts.append(t)
            metas.append(meta)
        self.metas = metas
        self._check_precuts(cfg)
        # concatenate in file order (SALT2mu reads the files in the listed order); remap the FIELD codes
        names = sorted(set().union(*[set(t["FIELD_NAMES"].tolist()) for t in parts]))
        lut = {n: i for i, n in enumerate(names)}
        cols = [c for c in parts[0] if c not in ("FIELD_NAMES", "FIELD_CODE") and all(c in t for t in parts)]
        S = {c: np.concatenate([t[c] for t in parts]) for c in cols}
        S["FIELD_CODE"] = np.concatenate([np.array([lut[n] for n in t["FIELD_NAMES"]])[t["FIELD_CODE"]] for t in parts])
        S["FIELD_NAMES"] = np.array(names)
        del parts
        # simulated alpha and beta of the map (set_DUST_FLAG_biasCor, get_BININFO_biasCor_abg)
        m0 = metas[0]
        if dust_flag is None:
            dust_flag = m0.get("dust_flag_first20")
            if dust_flag is None:
                raise UnsupportedOptionError(
                    "cannot tell whether SALT2mu's dust flag is set (needs SIM_AV of the first 20 rows of the first file before "
                    "cuts); pass dust_flag=True/False explicitly")
        self.dust_flag = bool(dust_flag)
        a_sim, b_sim = cfg.p[1], cfg.p[2]
        if not self.dust_flag:
            for col, pv in (("SIM_alpha", cfg.p[1]), ("SIM_beta", cfg.p[2])):
                vals = set()
                for m in metas:
                    if col + "_unique" not in m:
                        raise UnsupportedOptionError(f"no dust flag and no {col} information in the bias-correction files")
                    if m[col + "_nunique"] > 1:
                        raise UnsupportedOptionError(f"{col} takes several values: alpha/beta-dependent bias maps are not "
                                                     "implemented")
                    vals |= set(m[col + "_unique"])
                if len(vals) != 1 or abs(vals.pop() - pv) > 1e-6:
                    raise UnsupportedOptionError(f"{col} of the simulations differs from p{1 if col == 'SIM_alpha' else 2}; "
                                                 "set p1/p2 to the simulated values")
        for m in metas:
            if m.get("SIM_gammaDM_nunique", 1) > 1 or any(abs(v) > 0 for v in m.get("SIM_gammaDM_unique", [0.0])):
                raise UnsupportedOptionError("SIM_gammaDM != 0 in the bias-correction simulations is not implemented")
        if "SIM_WGT_POP" in S and not np.all(S["SIM_WGT_POP"] == 1):
            raise UnsupportedOptionError("SIM_WGT_POP != 1: weighted medians/STD of SALT2mu are not reproduced")
        if any(m.get("legacy") for m in metas):
            warnings.warn("legacy npz without metadata: SIM_gammaDM assumed 0, pre-applied cuts assumed compatible")
        self.a_sim, self.b_sim = float(a_sim), float(b_sim)
        # per-row derived quantities (compute_more_TABLEVAR, muresid_biasCor)
        self.key = sim_key(cfg)
        cosmo = cosmology_from_config(cfg)
        S = derive(S)
        S["__HOST__"] = host_values(S, cfg.varname_gamma)
        zhd = S["zHD"].astype(float)
        zt = S["SIM_ZCMB"].astype(float)
        S["SIM_MUz"] = (S["SIM_DLMAG"].astype(float) + 5 * np.log10(cosmo.dl(zhd, zhd) / cosmo.dl(zt, zt))).astype(F32)
        S["fitmu"] = (S["mB"].astype(float) - M0_DEFAULT + self.a_sim * S["x1"] - self.b_sim * S["c"]).astype(F32)
        dmhost = gamma_step(cfg.p[5], S["__HOST__"], cfg.p[7], cfg.p[8]) if cfg.use_gamma0 else 0.0
        S["muresid"] = (S["mB"].astype(float) + cfg.p[1] * S["x1"] - cfg.p[2] * S["c"] + dmhost - M0_DEFAULT) \
            - S["SIM_MUz"].astype(float)
        self.S = S
        self.nrow = len(S["zHD"])
        self._cut_cache = {}       # see BBC._sim_base_rows

    def _check_precuts(self, cfg):
        """Cuts already applied by bbclocal-extract must be implied by the current configuration."""
        eff = _sim_windows(cfg)
        for m in self.metas:
            pre = m.get("precuts")
            if pre is None:
                continue
            for var, (lo, hi) in pre.items():
                lo = -np.inf if lo is None else lo
                hi = np.inf if hi is None else hi
                elo, ehi = eff.get(var, (-np.inf, np.inf))
                if elo < lo or ehi > hi:
                    raise UnsupportedOptionError(
                        f"{m.get('source')}: was reduced with the cut {lo} <= {var} <= {hi}, which the current configuration "
                        f"does not imply ({var} window {elo}..{ehi}); re-extract with a matching --input")


class BBC:
    """One BBC fit: data + bias-correction simulations + configuration.

    cfg: validated BBCConfig; data: dict from io.read_fitres (it is extended in place with derived columns); sims:
    BiasCorSims built with a compatible configuration; survey_ids: {survey name: IDSURVEY} for surveygroup_biascor names;
    cid_reject: iterable of CIDs (data only); if None and cfg.cid_reject_file is readable, that file is used."""

    def __init__(self, cfg, data, sims, survey_ids=None, cid_reject=None, validate=True, allow_unknown=False):
        if validate:
            cfg.validate(allow_unknown=allow_unknown)
        if sims.key != sim_key(cfg):
            raise ValueError("the simulations were prepared with a different varname_gamma/p1/p2/p5/p7/p8/cosmology")
        sims._check_precuts(cfg)
        self.cfg, self.data, self.sims = cfg, data, sims
        self.survey_ids = survey_ids
        if cid_reject is None and cfg.cid_reject_file:
            try:
                cid_reject = read_cid_list(cfg.cid_reject_file)
            except OSError:
                raise FileNotFoundError(f"cid_reject_file {cfg.cid_reject_file} not found; pass cid_reject explicitly")
        self.cid_reject = set(map(str, cid_reject or ()))
        self.cosmo = cosmology_from_config(cfg)
        self.result = None

    # ------------------------------------------------------------------------------------------------------------ prepare
    def prepare(self):
        cfg, d, S = self.cfg, self.data, self.sims.S
        need = set(DATA_COLS) | {cfg.varname_gamma} | {c.var for c in cfg.cutwins}
        miss = [c for c in need if c not in d]
        if miss:
            raise KeyError(f"data lacks columns {miss}")
        if not any(k in d for k in ("x0", "x1", "c")):
            raise UnsupportedOptionError("only SALT2/SALT3 fits (x0, x1, c) are supported")
        derive(d)
        d["__HOST__"] = host_values(d, cfg.varname_gamma)
        n = len(d["zHD"])
        # samples from all data rows; biasCor z grid per sample from all rows (before cuts)
        self.samples = SampleSet(cfg, self.survey_ids)
        d_smp = self.samples.build_from_data(d["IDSURVEY"], d["FIELD"])
        self.data_sample = d_smp
        self.zgrid = {}
        for k, smp in enumerate(self.samples.samples):
            if smp.zbin is None:
                raise UnsupportedOptionError(f"sample '{smp.name}' has no zbin (add it to surveygroup_biascor)")
            self.zgrid[k] = biascor_zgrid(d["zHD"][d_smp == k], smp.zbin)
        self.mgrid = bininfo(cfg.logmass_min, cfg.logmass_max, (cfg.logmass_max - cfg.logmass_min) / cfg.nbin_logmass)
        # simulations: sample and cuts. The cuts other than the CUTWINs are computed once per sample definition and
        # window set and cached on the simulations object; the CUTWINs are then applied to those rows only.
        s_smp, base = self._sim_base_rows()
        ok = np.ones(len(base), bool)
        for cw in cfg.cutwins:
            if not cw.dataonly:
                ok &= window(S[cw.var][base], cw.lo, cw.hi)
        pas = base[ok]                                   # indices (ascending) of the rows passing all cuts
        # sigint for the map weights: SNR >= 60 rows passing all cuts (all samples), STD of the pull
        hi = pas[S["SNRMAX1"][pas] >= SNR_SIGINT]
        if len(hi) < MIN_SIGINT_EVENTS:
            raise RuntimeError(f"only {len(hi)} bias-correction rows with SNRMAX1 >= {SNR_SIGINT} pass the cuts (need 50)")
        a, b = self.sims.a_sim, self.sims.b_sim
        merr2 = muerrsq_full(S["cov"][:, hi], a, b, S["zHD"][hi].astype(float), S["zmuerr"][hi], cfg.lensing_zpar)
        self.sigint_biascor = sigint_scan(S["muresid"][hi], merr2, use_mad=False)
        self.maps = {}
        for k in range(len(self.samples.samples)):
            rows = pas[s_smp[pas] == k]
            M = SampleMaps(self.zgrid[k], self.mgrid, S, rows, self.sigint_biascor ** 2, a, b, cfg, cfg.zmax)
            M.make_mucov(S, cfg)
            self.maps[k] = M
        # data cuts and per-event bias and error corrections
        cut = np.zeros(n, int)
        cut[~self._selection(d, is_data=True, zmax_row=np.full(n, cfg.zmax))] |= CUT_SELECTION
        cut[np.isin(d["CID"].astype(str), list(self.cid_reject))] |= CUT_CID_REJECT
        self.mubias = np.full(n, np.nan)
        self.covadd = np.zeros(n)
        self.covscale = np.ones(n)
        self.nbias = np.zeros(n, int)
        for k, M in self.maps.items():
            q = np.nonzero((cut == 0) & (d_smp == k))[0]
            z, m = d["zHD"][q].astype(float), d["__HOST__"][q]
            c = d["c"][q].astype(float)
            v, st, ns = M.interp(z, m, d["x1"][q].astype(float), c)
            add, scl, okc = M.mucov(z, m, c)
            good = (st > 0) & okc
            self.mubias[q[good]] = v[good]
            self.covadd[q[good]] = add[good]
            self.covscale[q[good]] = scl[good]
            self.nbias[q[good]] = ns[good]
            cut[q[~good]] |= CUT_BIASCOR
        self.cutflag = cut
        return self

    def _sim_base_rows(self):
        """Sample index of every simulation row (-1: sample not in the data) and the indices of the rows that pass every cut
        except the CUTWINs. Both depend only on the sample definitions, the samples present in the data (with their z bin
        sizes) and the global windows, not on the population cuts, so they are cached on the BiasCorSims object."""
        cfg, S = self.cfg, self.sims.S
        smp = self.samples
        gkey = (tuple(tuple(m) for m in smp.members), tuple((g.name, g.zbin) for g in cfg.surveygroups),
                tuple((f.name, f.zbin) for f in cfg.fieldgroups), tuple((s.key, s.zbin) for s in smp.samples))
        wkey = (cfg.x1min, cfg.x1max, cfg.cmin, cfg.cmax, cfg.zmin, cfg.zmax, cfg.logmass_min, cfg.logmass_max)
        cache = self.sims.__dict__.setdefault("_cut_cache", {})
        if gkey not in cache:
            if len(cache) >= SIM_CACHE_MAX:
                cache.pop(next(iter(cache)))
            s_smp = smp.assign(S["IDSURVEY"], S["FIELD_CODE"], S["FIELD_NAMES"])
            cache[gkey] = {"s_smp": s_smp.astype(np.int32), "base": {}}
        entry = cache[gkey]
        s_smp = entry["s_smp"]
        if wkey not in entry["base"]:
            if len(entry["base"]) >= SIM_CACHE_MAX:
                entry["base"].pop(next(iter(entry["base"])))
            zmax_row = np.full(len(s_smp), -1.0)
            for k, sm in enumerate(smp.samples):
                zmax_row[s_smp == k] = cfg.zmax + sm.zbin
            ok = self._selection(S, is_data=False, zmax_row=zmax_row, cutwins=False) & (s_smp >= 0)
            entry["base"][wkey] = np.nonzero(ok)[0]
        return s_smp, entry["base"][wkey]

    def _selection(self, t, is_data, zmax_row, cutwins=True):
        cfg = self.cfg
        ok = np.ones(len(t["zHD"]), bool)
        for cw in cfg.cutwins if cutwins else ():
            if cw.dataonly and not is_data:
                continue
            ok &= window(t[cw.var], cw.lo, cw.hi)
        ok &= window(t["x1"], cfg.x1min, cfg.x1max) & window(t["c"], cfg.cmin, cfg.cmax)
        z = t["zHD"].astype(F32).astype(float)
        ok &= (z >= cfg.zmin) & (z <= zmax_row)
        h = t["__HOST__"]
        ok &= (h >= cfg.logmass_min) & (h <= cfg.logmass_max)
        ok &= ~t["BADERR"] & ~t["BADCOV"]
        return ok

    # ---------------------------------------------------------------------------------------------------------------- fit
    def fit(self, fix_ab=None):
        """Fit alpha, beta, gamma (those floated by u1, u2, u5), one M0 per redshift bin, and sigint (iterated until
        |chi2/dof - 1| < redchi2_tol). fix_ab=(alpha, beta): a bbclocal extension that fixes alpha and beta in the fit only
        (the bias maps keep p1, p2)."""
        cfg, d = self.cfg, self.data
        if not hasattr(self, "cutflag"):
            self.prepare()
        cut = self.cutflag.copy()
        zbins = powz_bins(cfg.zmin, cfg.zmax, cfg.nzbin, cfg.powzbin)
        NZ = cfg.nzbin
        start = np.array([cfg.p[1], cfg.p[2], cfg.p[5] if cfg.use_gamma0 else 0.0])
        free = np.array([cfg.float_alpha, cfg.float_beta, cfg.float_gamma and cfg.use_gamma0])
        if fix_ab is not None:
            start[:2] = fix_ab
            free[:2] = False
        bounds_all = [PAR_BOUNDS["alpha"], PAR_BOUNDS["beta"], PAR_BOUNDS["gamma"]]
        tau, cen, lens = cfg.p[8], cfg.p[7], cfg.lensing_zpar

        def arrays(u):
            z = d["zHD"][u].astype(F32).astype(float)
            zhel = d["zHEL"][u].astype(F32).astype(float)
            A = {"z": z, "mumodel": self.cosmo.mu(zhel, z).astype(F32).astype(float), "mB": d["mB"][u].astype(float),
                 "s": d["x1"][u].astype(float), "c": d["c"][u].astype(float), "host": d["__HOST__"][u], "cov": d["cov"][:, u],
                 "bias": self.mubias[u], "add": self.covadd[u], "scl": self.covscale[u]}
            A["vpec2"] = muerr_z(z, d["zmuerr"][u]) ** 2
            A["IZ"] = ibin(z, zbins)
            return A

        def errsq(A, a, b, sig2):
            e = quadform(A["cov"], a, b, sig2, as_float=True) + A["vpec2"] + (lens * A["z"]) ** 2
            return np.where(A["add"] > 0, e + A["add"], (e - A["vpec2"]) * A["scl"] + A["vpec2"])

        def resid(A, q):
            return A["mB"] + q[0] * A["s"] - q[1] * A["c"] - gamma_step(q[2], A["host"], cen, tau) - A["bias"] - A["mumodel"]

        def full(p):
            q = start.copy()
            q[free] = p
            return q

        # chi2max: M0 re-centring and outlier rejection at the start values (applyCut_chi2max, legacy opt_chi2max=1)
        sig = cfg.sig1
        self.n_chi2max_reject = 0
        u = cut == 0
        if cfg.chi2max_on:
            A = arrays(u)
            e2 = errsq(A, start[0], start[1], sig ** 2)
            mures = resid(A, start) - M0_DEFAULT
            muoff = np.sum(mures / e2) / np.sum(1 / e2)
            chi2 = (mures - muoff) ** 2 / e2
            rej = np.nonzero(u)[0][chi2 > cfg.chi2max]
            cut[rej] |= CUT_CHI2MAX
            self.n_chi2max_reject = len(rej)
            u = cut == 0
        A = arrays(u)
        IZ = A["IZ"]
        if (IZ < 0).any():
            raise RuntimeError("an event outside the fit redshift bins passed the cuts")
        nper = np.bincount(IZ, minlength=NZ)
        nfloat = int(free.sum()) + int((nper >= cfg.min_per_zbin).sum())
        N = int(u.sum())

        def profile(p, sig2, last):
            q = full(p)
            e2 = errsq(A, q[0], q[1], sig2)
            w = 1 / e2
            r = resid(A, q)
            M0 = np.bincount(IZ, w * r, NZ) / np.where(nper > 0, np.bincount(IZ, w, NZ), 1)
            rr = r - M0[IZ]
            chi = np.sum(rr * rr * w)
            return chi + np.sum(np.log(e2 / last)), chi, M0, e2

        p0 = start[free].copy()
        bounds = [b for b, f in zip(bounds_all, free) if f]
        last = errsq(A, start[0], start[1], sig ** 2)             # muerrsq_last at the start values
        hist = []
        for it in range(MAXFITITER):
            if len(p0):
                obj = lambda p: profile(p, sig ** 2, last)[0]      # noqa: E731
                res = optimize.minimize(obj, p0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 20000})
                res = optimize.minimize(obj, res.x, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-15, "gtol": 1e-10})
                p0 = res.x
            tot, chi, M0, e2 = profile(p0, sig ** 2, last)
            red = chi / (N - nfloat + 1e-8)
            hist.append((float(sig), float(red)))
            if abs(red - 1) < cfg.redchi2_tol or (it > 0 and sig <= 0):
                break
            last = e2
            if it == 0:
                sig = sig + 0.01 * np.sign(red - 1)
            else:
                (s1, r1), (s2, r2) = hist[-2], hist[-1]
                sig = s2 - (r2 - 1) / ((r2 - r1) / (s2 - s1))
            if sig > 100:
                raise RuntimeError("sigint iteration diverged")
            sig = max(sig, 0.0)
        tot, chi, M0, e2 = profile(p0, sig ** 2, last)
        q = full(p0)
        errs = self._errors(A, q, free, sig, last, errsq, resid, nper)
        r = resid(A, q)
        fl = nper > 0
        avemag0 = np.sum(nper[fl] * M0[fl]) / np.sum(nper[fl])
        mu = r + A["mumodel"]
        self.MU = np.full(len(u), np.nan)
        self.MUERR = np.full(len(u), np.nan)
        self.MUMODEL = np.full(len(u), np.nan)
        self.IZBIN = np.full(len(u), -9)
        self.MU[u] = mu - avemag0
        self.MUERR[u] = np.sqrt(e2)
        self.MUMODEL[u] = A["mumodel"]
        self.IZBIN[u] = IZ
        self.cutflag_fit = cut
        self.result = {
            "bbclocal_version": __version__,
            "alpha0": float(q[0]), "beta0": float(q[1]), "gamma0": float(q[2]),
            "alpha0_err": errs.get("alpha"), "beta0_err": errs.get("beta"), "gamma0_err": errs.get("gamma"),
            "floated": {"alpha": bool(free[0]), "beta": bool(free[1]), "gamma": bool(free[2])},
            "sigint": float(sig), "NSNFIT": N, "chi2_Ia": float(chi), "ndof": int(N - nfloat), "chi2red": float(red),
            "AVEMAG0": float(avemag0),
            "zbins": [{"iz": i, "zmin": float(zbins.lo[i]), "zmax": float(zbins.hi[i]), "N": int(nper[i]),
                       "M0": float(M0[i]) if fl[i] else None, "M0DIF": float(M0[i] - avemag0) if fl[i] else None,
                       "M0ERR": errs["M0"][i] if fl[i] else None} for i in range(NZ)],
            "sigint_iterations": [{"sigint": s_, "chi2red": r_} for s_, r_ in hist],
            "sigint_biascor_SNR60": float(self.sigint_biascor),
            "n_chi2max_reject": int(self.n_chi2max_reject),
            "NCOVFIX_data": int(d["NCOVFIX"]), "NCOVFIX_biascor": int(self.sims.S["NCOVFIX"]),
            "dust_flag": self.sims.dust_flag,
            "samples": [{"IDSAMPLE": k, "name": smp.name, "zbin": smp.zbin, "N_data_all": int((self.data_sample == k).sum()),
                         "N_fit": int(((self.data_sample == k) & u).sum()), "N_biascor": int(self.maps[k].nsim),
                         "z_grid": [float(self.zgrid[k].lo[0]), float(self.zgrid[k].hi[-1])]}
                        for k, smp in enumerate(self.samples.samples)],
        }
        return self.result

    def _errors(self, A, q, free, sig, last, errsq, resid, nper):
        """Approximate parameter errors from a finite-difference Hessian of the fitted objective in (free alpha, beta, gamma,
        M0 of populated bins), at the final sigint: cov = 2 H^-1. MINUIT/MIGRAD errors in SALT2mu are computed differently
        (its own approximate covariance), so these agree only approximately."""
        NZ = len(nper)
        fl = np.nonzero(nper > 0)[0]
        idx = np.nonzero(free)[0]
        IZ = A["IZ"]

        def obj(x):
            qq = q.copy()
            qq[idx] = x[:len(idx)]
            M0 = np.zeros(NZ)
            M0[fl] = x[len(idx):]
            e2 = errsq(A, qq[0], qq[1], sig ** 2)
            rr = resid(A, qq) - M0[IZ]
            return np.sum(rr * rr / e2) + np.sum(np.log(e2 / last))

        e2 = errsq(A, q[0], q[1], sig ** 2)
        w = 1 / e2
        r = resid(A, q)
        M0 = np.bincount(IZ, w * r, NZ) / np.where(nper > 0, np.bincount(IZ, w, NZ), 1)
        x0 = np.concatenate([q[idx], M0[fl]])
        h = np.concatenate([np.array([1e-4, 1e-3, 1e-4])[idx], np.full(len(fl), 1e-3)])
        n = len(x0)
        H = np.zeros((n, n))
        f0 = obj(x0)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    xp, xm = x0.copy(), x0.copy()
                    xp[i] += h[i]
                    xm[i] -= h[i]
                    H[i, i] = (obj(xp) - 2 * f0 + obj(xm)) / h[i] ** 2
                else:
                    if i >= len(idx) and j >= len(idx):
                        continue                                    # M0 bins do not couple directly
                    vals = []
                    for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        x = x0.copy()
                        x[i] += si * h[i]
                        x[j] += sj * h[j]
                        vals.append(obj(x))
                    H[i, j] = H[j, i] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h[i] * h[j])
        try:
            cov = 2 * np.linalg.inv(H)
            err = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            err = np.full(n, np.nan)
        out = {}
        for k, i in enumerate(idx):
            out[["alpha", "beta", "gamma"][i]] = float(err[k])
        m0e = np.full(NZ, np.nan)
        m0e[fl] = err[len(idx):]
        out["M0"] = [float(v) for v in m0e]
        return out

    # ------------------------------------------------------------------------------------------------------------ outputs
    def table(self):
        """FITRES-like table of all data rows (pandas DataFrame). CUTFLAG is bbclocal's own bit mask (see core.CUT_*);
        rows with CUTFLAG != 0 have MU = MUERR = -9."""
        import pandas as pd
        if self.result is None:
            raise RuntimeError("run fit() first")
        d = self.data
        names = np.array([s.name for s in self.samples.samples], dtype=object)
        used = self.cutflag_fit == 0
        out = pd.DataFrame({
            "CID": d["CID"].astype(str), "IDSURVEY": d["IDSURVEY"], "FIELD": d["FIELD"].astype(str),
            "zHD": d["zHD"].astype(float), "zHEL": d["zHEL"].astype(float),
            "MU": np.where(used, self.MU, -9.0), "MUMODEL": np.where(used, self.MUMODEL, -9.0),
            "MUERR": np.where(used, self.MUERR, -9.0),
            "MURES": np.where(used, self.MU - self.MUMODEL, -9.0),
            "biasCor_mu": np.where(np.isfinite(self.mubias), self.mubias, -9.0),
            "biasCor_muCOVSCALE": np.where(self.covadd > 0, 1.0, self.covscale),
            "biasCor_muCOVADD": self.covadd, "biasCor_nevt": self.nbias,
            "IDSAMPLE": self.data_sample, "SAMPLE": names[self.data_sample], "IZBIN": self.IZBIN,
            "CUTFLAG": self.cutflag_fit})
        return out
