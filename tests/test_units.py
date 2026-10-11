import numpy as np
import pytest

from bbclocal.binning import bininfo, biascor_zgrid, ibin, linear_bins, powz_bins
from bbclocal.config import UnsupportedOptionError, parse_salt2mu_text
from bbclocal.covfit import LowRankCov, build_covariance
from bbclocal.maps import sigint_scan
from bbclocal.model import EIGMIN_COV, derive
from bbclocal.samples import SampleSet

from toy import INPUT, SURVEY_IDS


# ---------------------------------------------------------------- binning
def test_bininfo_cumulative_edges():
    B = bininfo(-0.3, 0.5, 0.05)
    lo, expect = -0.3, []
    while lo <= 0.5 - 0.99 * 0.05:          # the C loop
        expect.append(lo)
        lo += 0.05
    assert B.n == 16
    assert np.array_equal(B.lo, np.array(expect))          # bit-identical to cumulative addition
    assert not np.array_equal(B.lo, -0.3 + 0.05 * np.arange(16))   # ... which differs from vmin + k*vbin


def test_ibin_lower_edge_precedence_and_padding():
    B = bininfo(0.0, 1.0, 0.25)
    assert ibin([0.25], B)[0] == 0                         # on an edge -> lower bin
    assert ibin([0.2500001], B)[0] == 1
    assert ibin([1.0 + 5e-11], B)[0] == 3                  # last hi padded by 1e-10
    assert ibin([1.0 + 1e-9], B)[0] == -9
    assert ibin([-0.1], B)[0] == -9
    assert ibin([-0.1, 2.0], B, opt=2).tolist() == [0, 3]


def test_powz_bins_equal_shares():
    zb = powz_bins(0.025, 1.2, 20, 3.0)
    share = [(1 + lo) ** -2 - (1 + hi) ** -2 for lo, hi in zip(zb.lo, zb.hi)]
    assert np.allclose(share, share[0], rtol=1e-10)
    assert np.isclose(zb.hi[-1], 1.2)


def test_biascor_zgrid_from_data():
    G = biascor_zgrid(np.array([0.0349, 0.71], np.float32), 0.075)
    assert np.isclose(G.lo[0], 0.03)                      # 0.01 * int(100 * zmin)
    assert G.hi[-1] > 0.71 + 0.075 - 1e-9


def test_linear_bins():
    B = linear_bins(-0.3, 0.3, 3)
    assert np.allclose(B.lo, [-0.3, -0.1, 0.1]) and np.allclose(B.hi, [-0.1, 0.1, 0.3])


# ---------------------------------------------------------------- covariance eigenvalue fix
def _table(cov_x1_c, x1err=0.1, cerr=0.1):
    return {"x0": np.array([1e-3], np.float32), "x0ERR": np.array([1e-5], np.float32),
            "x1ERR": np.array([x1err], np.float32), "cERR": np.array([cerr], np.float32),
            "COV_x1_c": np.array([cov_x1_c], np.float32), "COV_x1_x0": np.array([0.0], np.float32),
            "COV_c_x0": np.array([0.0], np.float32), "VPECERR": np.array([250.0], np.float32)}


def test_eigenvalue_fix_only_when_not_positive_definite():
    t = derive(_table(0.005))
    assert t["NCOVFIX"] == 0
    assert np.isclose(t["cov"][4, 0], 0.005)
    t = derive(_table(0.02))                               # |cov| > x1err * cerr -> one negative eigenvalue
    assert t["NCOVFIX"] == 1
    c = t["cov"][:, 0].astype(float)
    C = np.array([[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]])
    assert np.linalg.eigvalsh(C).min() >= EIGMIN_COV * (1 - 1e-5)


# ---------------------------------------------------------------- sigint scan
def test_sigint_scan_recovers_scatter():
    rng = np.random.default_rng(3)
    err = rng.uniform(0.05, 0.15, 20000)
    res = rng.normal(0, np.sqrt(err ** 2 + 0.1 ** 2))
    s = sigint_scan(res, err ** 2, use_mad=False)
    assert abs(s - 0.1) < 0.005
    s = sigint_scan(res, err ** 2, use_mad=True)
    assert abs(s - 0.1) < 0.01


def _sigint_scan_stepwise(mures, mucov, use_mad, abort=True):
    """Straightforward transcription of the SALT2mu step-by-step scan (sigint_muresid_list), for comparison."""
    avg_cov, avg_res = np.mean(mucov), np.mean(mures)
    if use_mad:
        std_orig = 1.48 * np.median(np.abs(mures))
    else:
        std_orig = np.sqrt(max(np.mean(mures ** 2) - np.mean(mures) ** 2, 0.0))
    tmp = std_orig ** 2 - avg_cov
    approx = -np.sqrt(abs(tmp)) if tmp < 0 else np.sqrt(tmp)
    lo, sig = approx - 50 * 0.01 - 1e-7, approx + 60 * 0.01
    sig_store, pull_store, adjusted = [], [], False
    while True:
        if sig < lo:
            if abort:
                raise RuntimeError("fail")
            return lo, "fail", adjusted
        covt = sig * abs(sig)
        tot = mucov + covt
        if covt < 0:
            tot = np.where(tot < 0.02 ** 2, 0.02 ** 2, tot)
        pull = (mures - avg_res) / np.sqrt(tot)
        sp = 1.48 * np.median(np.abs(pull)) if use_mad else np.sqrt(max(np.mean(pull ** 2) - np.mean(pull) ** 2, 0.0))
        if not sig_store and not adjusted and sp > 1.0:
            sig += (sp - 1.0) + 0.1
            adjusted = True
            continue
        sig_store.append(sig)
        pull_store.append(sp)
        sig -= 0.01
        if sp > 1.0:
            break
    if len(sig_store) == 1 and abs(pull_store[0] - 1.0) < 0.001:
        return sig_store[0], "single", adjusted
    return float(np.interp(1.0, np.array(pull_store), np.array(sig_store))), "interp", adjusted


def test_sigint_scan_bisection_equals_stepwise():
    rng = np.random.default_rng(11)
    seen = set()
    for trial in range(300):
        n = int(rng.integers(20, 3000))
        kind = trial % 3
        if kind == 0:        # ordinary: scatter + errors of similar size
            err = rng.uniform(0.03, 0.2, n)
            res = rng.normal(0.0, np.sqrt(err ** 2 + rng.uniform(0, 0.2) ** 2))
        elif kind == 1:      # a few very large errors pull the start value far down: upward adjustment
            err = np.full(n, 0.02)
            err[: max(1, n // 50)] = rng.uniform(3, 10)
            res = rng.normal(0.0, rng.uniform(0.1, 0.4), n)
        else:                # residuals far below the errors: the scan never reaches STD(pull) = 1
            err = rng.uniform(0.5, 1.0, n)
            res = rng.normal(0.0, 1e-4, n)
        for use_mad in (False, True):
            ref, how, adjusted = _sigint_scan_stepwise(res, err ** 2, use_mad, abort=False)
            seen.add(how)
            if adjusted:
                seen.add("adjusted")
            assert sigint_scan(res, err ** 2, use_mad, abort=False) == ref
            assert sigint_scan(res, err ** 2, use_mad, abort=False, bisect=False) == ref
            if how == "fail":
                with pytest.raises(RuntimeError):
                    sigint_scan(res, err ** 2, use_mad)
    assert {"interp", "fail", "adjusted"} <= seen


def test_sigint_scan_std_steps_mad_bisects(monkeypatch):
    """STD(pull) is not guaranteed monotonic in sigint, so its scan must visit the trial values one by one; the MAD scan
    (monotonic when every mucov >= 0.02^2) bisects."""
    import bbclocal.maps as maps
    calls = []
    orig = maps._std_pull

    def spy(dev, mucov, sig, use_mad):
        calls.append(sig)
        return orig(dev, mucov, sig, use_mad)
    monkeypatch.setattr(maps, "_std_pull", spy)
    rng = np.random.default_rng(5)
    err = rng.uniform(0.05, 0.15, 5000)
    res = rng.normal(0, np.sqrt(err ** 2 + 0.1 ** 2))
    calls.clear()
    maps.sigint_scan(res, err ** 2, use_mad=False)
    steps = np.diff(calls[1:-2])               # skip the first call and the final two (bracketing pair) calls
    assert len(steps) > 10 and np.allclose(steps, -0.01)
    calls.clear()
    maps.sigint_scan(res, err ** 2, use_mad=True)
    assert len(calls) < 15                     # 1 + ~log2(110) + 2
    calls.clear()
    maps.sigint_scan(res, np.r_[err[:-1], 0.01] ** 2, use_mad=True)    # a mucov below 0.02^2: step by step
    assert np.allclose(np.diff(calls[1:-2]), -0.01) and len(calls) > 15


# ---------------------------------------------------------------- covariance helper
def test_woodbury_matches_dense():
    rng = np.random.default_rng(0)
    n = 50
    C = build_covariance(rng.uniform(0.1, 0.2, n), [rng.normal(0, 0.02, n), rng.normal(0, 0.03, n)],
                         groups=rng.integers(0, 3, n))
    R = rng.normal(size=(4, n))
    dense = np.einsum("in,nm,im->i", R, np.linalg.inv(C.dense()), R)
    assert np.allclose(C.quad(R), dense)
    assert isinstance(C, LowRankCov)


# ---------------------------------------------------------------- configuration
def test_parse_input_and_muopt():
    cfg = parse_salt2mu_text("CONFIG:\n  X: 1\n#END_YAML\n" + INPUT, muopt="CUTWIN c -0.3 0.0 chi2max=10000000")
    cfg.validate()
    assert cfg.opt_biascor == 4336 and cfg.nzbin == 8 and cfg.powzbin == 3.0
    assert [g.name for g in cfg.surveygroups] == ["LOWZ", "DES"] and cfg.surveygroups[1].zbin == 0.075
    assert [g.name for g in cfg.fieldgroups] == ["C3", "X3"]
    assert [c.var for c in cfg.cutwins] == ["x1ERR", "FITPROB", "c"]
    assert cfg.chi2max == 1e7 and cfg.chi2max_on
    assert cfg.p[2] == 3.1 and cfg.u[2] == 3


@pytest.mark.parametrize("extra", [
    "opt_biascor=4592", "interp_biascor_logmass=1", "varname_pIa=PROB_SNNV19", "simfile_ccprior=x.FITRES",
    "prescale_biascor=0,2", "sigma_cell_biascor=5", "u3=1", "p10=0.1", "uM0=2", "blindflag=1", "foo_bar=3",
    "zphot_shift=0.01", "surveygroup_biascor='LOWZ,DES(zbin=0.075)'"])
def test_refuses_unsupported(extra):
    with pytest.raises(UnsupportedOptionError):
        parse_salt2mu_text(INPUT + "\n" + extra + "\n").validate()


@pytest.mark.parametrize("extra", ["CUTWIN(BIASCORONLY) c -0.3 0.3", "chi2max(DES)=10", "CUTWIN NONE"])
def test_refuses_unsupported_at_parse(extra):
    with pytest.raises(UnsupportedOptionError):
        parse_salt2mu_text(INPUT + "\n" + extra + "\n")


def test_dataonly_cutwin_accepted():
    cfg = parse_salt2mu_text(INPUT + "\nCUTWIN(DATAONLY) PKMJDERR 0 2\n").validate()
    assert cfg.cutwins[-1].dataonly


# ---------------------------------------------------------------- samples
def test_field_groups_and_order_rule():
    cfg = parse_salt2mu_text(INPUT.replace("fieldGroup_biascor=C3,X3", "fieldGroup_biascor=X3,C3,X1+X2"))
    S = SampleSet(cfg, SURVEY_IDS)
    ids = np.array([1, 1, 10, 10, 10, 10])
    fld = np.array(["VOID", "VOID", "C3", "X1", "X2+X3", "X3"])
    k = S.build_from_data(ids, fld)
    names = [S.samples[i].name for i in k]
    assert names == ["LOWZ", "LOWZ", "DES(C3)", "DES(X1+X2)", "DES(X3)", "DES(X3)"]   # overlap: first group in input order
    with pytest.raises(UnsupportedOptionError):                                       # unmatched after matched
        SampleSet(cfg, SURVEY_IDS).build_from_data(ids[::-1], fld[::-1])
