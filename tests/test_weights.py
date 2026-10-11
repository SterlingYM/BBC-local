"""Population weights (SIM_WGT_POP): the weighted statistics of SALT2mu (arrayStat_MEDIAN, arrayStat_STD), the weighted
maps, the with_weights view and its min_weight option."""
import gzip

import numpy as np
import pandas as pd
import pytest

from bbclocal import BBC, BiasCorSims, read_fitres, read_salt2mu_input
from bbclocal.maps import SampleMaps, gsl_sort_index, sigint_scan, std_from_sums, weighted_median, _std_pull

import toy


# ---------------------------------------------------------------- weighted median (arrayStat_MEDIAN)
def c_median(vals, wgt):
    """Line-by-line transcription of arrayStat_MEDIAN (sntools.c) with weights, using GSL's heapsort index."""
    N = len(vals)
    idx = gsl_sort_index(vals)
    SUM_WGT = 0.0
    for i in range(N):
        SUM_WGT += wgt[i]
    half = SUM_WGT / 2.0
    s, i, imed0, imed1 = 0.0, 0, -9, -9
    while s < half:
        imed0 = idx[i]
        imed1 = idx[i + 1] if i + 1 < N else -9     # C reads past the end here; the test never gets there
        s += wgt[imed0]
        i += 1
    if N % 2 == 1:
        imed1 = imed0
    assert 0 <= imed0 < N and 0 <= imed1 < N
    return 0.5 * (vals[imed0] + vals[imed1])


def test_gsl_sort_index_sorts():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 7, 50, 501):
        v = rng.integers(0, 5, n).astype(float)        # many ties
        p = gsl_sort_index(v)
        assert sorted(p.tolist()) == list(range(n))
        assert np.all(np.diff(v[p]) >= 0)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 11, 100, 101])
def test_weighted_median_unit_weights_is_ordinary_median(n):
    v = np.random.default_rng(n).normal(size=n)
    assert weighted_median(v, np.ones(n)) == np.median(v)
    assert weighted_median(v) == np.median(v)


@pytest.mark.parametrize("seed", range(40))
def test_weighted_median_matches_c_algorithm(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 60))
    ties = seed % 2 == 0
    v = (rng.integers(0, 6, n).astype(float) if ties else rng.normal(size=n))
    w = rng.choice([0.0, 0.5, 1.0, 2.0, 3.0], n) if seed % 4 < 2 else rng.uniform(0, 2, n)
    w[0] = max(w[0], 0.1)
    # skip draws in which the C code would read past the end (tested separately)
    order = gsl_sort_index(v)
    cs = np.cumsum(w[order])
    if n % 2 == 0 and np.searchsorted(cs, np.cumsum(w)[-1] / 2, side="left") >= n - 1:
        pytest.skip("crossing at the last element")
    assert weighted_median(v, w) == c_median(v, w)


def test_weighted_median_ties_follow_gsl_heapsort():
    """Tied values with different weights at the crossing: the result depends on the order of the tied elements, so the
    stable order of numpy would differ from GSL's heapsort; bbclocal must follow the heapsort."""
    found = 0
    rng = np.random.default_rng(1)
    for _ in range(2000):
        n = int(rng.integers(4, 12)) * 2               # even N
        v = rng.integers(0, 3, n).astype(float)
        w = rng.choice([0.0, 1.0, 5.0], n)
        w[0] = 1.0
        order = gsl_sort_index(v)
        cs = np.cumsum(w[order])
        if np.searchsorted(cs, np.cumsum(w)[-1] / 2, side="left") >= n - 1:
            continue
        st = np.argsort(v, kind="stable")
        cs2 = np.cumsum(w[st])
        if np.searchsorted(cs2, np.cumsum(w)[-1] / 2, side="left") >= n - 1:
            continue
        stable = 0.5 * (v[st[np.searchsorted(cs2, cs2[-1] / 2)]] + v[st[np.searchsorted(cs2, cs2[-1] / 2) + 1]])
        c = c_median(v, w)
        assert weighted_median(v, w) == c
        found += stable != c
    assert found > 0                                    # the case occurs and is handled


def test_weighted_median_parity_by_count_and_errors():
    # even count: average of the crossing element and the next one, even if the next one has weight 0
    assert weighted_median([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.0, 0.0]) == 1.5
    assert weighted_median([1.0, 2.0, 3.0, 4.0], [3.0, 1.0, 0.0, 0.0]) == 1.5
    # odd count: the crossing element alone
    assert weighted_median([1.0, 2.0, 3.0], [3.0, 1.0, 0.0]) == 1.0
    with pytest.raises(RuntimeError, match="total weight 0"):
        weighted_median([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="reads past the end"):
        weighted_median([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        weighted_median([1.0, 2.0], [1.0, -1.0])


# ---------------------------------------------------------------- weighted STD (arrayStat_STD quirk)
def test_std_from_sums_uses_count():
    rng = np.random.default_rng(3)
    x = rng.normal(0.3, 1.0, 1000)
    w = rng.uniform(1.0, 3.0, 1000)                  # mean weight 2: the count and the weight sum differ
    got = std_from_sums(len(x), np.sum(w * x), np.sum(w * x * x))
    avg = np.sum(w * x) / len(x)
    assert got == pytest.approx(np.sqrt(np.sum(w * x * x) / len(x) - avg * avg), rel=1e-14)
    proper = np.sqrt(np.cov(x, aweights=w, bias=True))
    assert abs(got - proper) > 0.1                    # the quirk differs from a weighted STD
    assert std_from_sums(0, 1.0, 1.0) == 0.0
    assert std_from_sums(2, 0.0, -1.0) == 0.0


def test_weighted_std_pull_and_duplication():
    """With the count in the denominator, weight 2 on a row is NOT the same as duplicating it (the count differs); with
    weights normalised to a mean of 1 it would be, which is not what SALT2mu does."""
    rng = np.random.default_rng(4)
    res = rng.normal(0, 0.15, 400)
    cov = rng.uniform(0.05, 0.1, 400) ** 2
    w = np.ones(400)
    w[7] = 2.0
    dup_res, dup_cov = np.r_[res, res[7]], np.r_[cov, cov[7]]
    s_w = _std_pull(res - 0.0, cov, 0.1, False, w)
    s_d = _std_pull(dup_res - 0.0, dup_cov, 0.1, False, np.ones(401))
    assert s_w != pytest.approx(s_d, rel=1e-6)
    # unit weights reproduce the unweighted scan exactly
    assert sigint_scan(res, cov, use_mad=False, wgt=np.ones(400)) == sigint_scan(res, cov, use_mad=False)
    assert sigint_scan(res, cov, use_mad=True, wgt=np.ones(400)) == sigint_scan(res, cov, use_mad=True)
    assert sigint_scan(res, cov, use_mad=False, wgt=rng.uniform(0.5, 1.5, 400)) != sigint_scan(res, cov, use_mad=False)


# ---------------------------------------------------------------- maps and the with_weights view
@pytest.fixture(scope="module")
def toyset(tmp_path_factory):
    d = tmp_path_factory.mktemp("toyw")
    paths = toy.make_toy(str(d), nsim=15000, ndata=(150, 600), seed=3)
    cfg = read_salt2mu_input(paths["input"]).validate()
    sims = BiasCorSims([paths["sim_lowz"], paths["sim_des"]], cfg)
    return paths, cfg, sims


def _bbc(cfg, paths, sims):
    return BBC(cfg, read_fitres(paths["data"]), sims, survey_ids=toy.SURVEY_IDS, cid_reject=[]).prepare()


def test_view_shares_memory_and_has_own_cache(toyset):
    paths, cfg, sims = toyset
    w = np.random.default_rng(0).uniform(0.5, 1.5, sims.nrow)
    v = sims.with_weights(w)
    assert sims.weights is None and v.weights is not None
    assert v.S is not sims.S
    for k in ("x1", "c", "zHD", "cov", "muresid", "fitmu", "SIM_MUz", "__HOST__", "IDSURVEY"):
        assert np.shares_memory(v.S[k], sims.S[k])
    assert not np.shares_memory(v.S["SIM_WGT_POP"], sims.S["SIM_WGT_POP"])
    assert np.all(sims.S["SIM_WGT_POP"] == 1)                       # parent unchanged
    assert v.S["SIM_WGT_POP"].dtype == np.float32
    assert np.array_equal(v.weights, w.astype(np.float32).astype(float))
    _bbc(cfg, paths, sims)
    _bbc(cfg, paths, v)
    assert v._cut_cache is not sims._cut_cache
    assert len(v._cut_cache) == 1 and len(sims._cut_cache) == 1
    v2 = sims.with_weights(w, min_weight=0.9)
    B2 = _bbc(cfg, paths, v2)
    (entry,) = v2._cut_cache.values()
    (base2,) = entry["base"].values()
    (base,) = next(iter(sims._cut_cache.values()))["base"].values()
    assert np.array_equal(base2, base[v2.weights[base] > 0.9])       # min_weight rows removed, only in that view
    assert all(M.nsim > 0 for M in B2.maps.values())
    with pytest.raises(ValueError):
        sims.with_weights(w[:-1])
    with pytest.raises(ValueError):
        sims.with_weights(-w)


def test_unit_weight_view_identical_fit(toyset):
    paths, cfg, sims = toyset
    B0 = _bbc(cfg, paths, sims)
    r0 = B0.fit()
    B1 = _bbc(cfg, paths, sims.with_weights(np.ones(sims.nrow)))
    r1 = B1.fit()
    for k in ("alpha0", "beta0", "gamma0", "sigint", "AVEMAG0"):
        assert r0[k] == r1[k]
    for a in ("mubias", "covadd", "covscale", "MU", "MUERR", "cutflag_fit"):
        assert np.array_equal(getattr(B0, a), getattr(B1, a), equal_nan=True)
    assert B0.sigint_biascor == B1.sigint_biascor


def _maps(B, sims, rows, wpop):
    cfg = B.cfg
    M = SampleMaps(B.zgrid[1], B.mgrid, sims.S, rows, B.sigint_biascor ** 2, sims.a_sim, sims.b_sim, cfg, cfg.zmax, wpop=wpop)
    M.make_mucov(sims.S, cfg)
    return M


def test_integer_weight_equals_duplicate_for_averages(toyset):
    """Weight 2 on a row equals listing the row twice for the weighted averages (VAL, AVG of the 5D map; WGT_POP-weighted
    MUCOV cell averages), but not for the counts (NperCell), which SALT2mu keeps unweighted."""
    paths, cfg, sims = toyset
    B = _bbc(cfg, paths, sims)
    M0 = B.maps[1]
    rows = M0.rows
    rng = np.random.default_rng(2)
    pick = rng.choice(rows, 200, replace=False)
    w = np.ones(sims.nrow)
    w[pick] = 2.0
    Mw = _maps(B, sims, rows, w)
    Md = _maps(B, sims, np.sort(np.r_[rows, pick]), None)
    used = Mw.N >= 1
    assert np.allclose(Mw.val[used], Md.val[used], rtol=0, atol=1e-12)
    for k in Mw.avg:
        assert np.allclose(Mw.avg[k], Md.avg[k], rtol=0, atol=1e-12)
    # counts are not weighted: the duplicate adds to NperCell, the weight does not
    assert np.array_equal(Mw.N, M0.N) and Md.N.sum() == M0.N.sum() + 200
    # RMS/ERR diagnostics use WGT_POP sums: equal to the duplicate where the neighbour rule (a count) is the same
    rw, ew = Mw.bias_rms_err()
    rd, ed = Md.bias_rms_err()
    same = used & ((Mw.N >= 10) == (Md.N >= 10))
    assert np.allclose(rw[same], rd[same], atol=1e-10) and np.allclose(ew[same], ed[same], atol=1e-10)


def test_mucovadd_unweighted_mucovscale_weighted(toyset):
    """MUCOVADD comes from an unweighted scan (WGT_IGNORE in SALT2mu): weights that leave the 5D bias map unchanged leave
    MUCOVADD unchanged. MUCOVSCALE (weighted median) and the MUCOV cell averages change."""
    paths, cfg, sims = toyset
    B = _bbc(cfg, paths, sims)
    M0 = B.maps[1]
    rows = M0.rows
    w = np.random.default_rng(5).uniform(0.2, 1.8, sims.nrow)
    Mw = _maps(B, sims, rows, w)
    # put the unweighted 5D map into the weighted one, so that the bias of each simulated event is the same
    Mw.val, Mw.avg = M0.val, M0.avg
    Mw.make_mucov(sims.S, cfg)
    assert np.array_equal(Mw.cadd, M0.cadd)
    assert np.array_equal(Mw.cN, M0.cN)
    assert not np.array_equal(Mw.cscale[M0.cuse], M0.cscale[M0.cuse])
    assert not np.allclose(Mw.cavg["c"][M0.cuse], M0.cavg["c"][M0.cuse], rtol=0, atol=1e-12)


def test_zero_weights_count_but_min_weight_removes(toyset, tmp_path):
    """Faithful zero weights still count toward NperCell; min_weight=0 instead equals deleting the rows from the files."""
    paths, cfg, sims = toyset
    rng = np.random.default_rng(7)
    w = np.where(rng.uniform(size=sims.nrow) < 0.02, 0.0, 1.0)
    Bz = _bbc(cfg, paths, sims.with_weights(w))
    B1 = _bbc(cfg, paths, sims)
    for k in B1.maps:
        assert np.array_equal(Bz.maps[k].N, B1.maps[k].N)
        assert Bz.maps[k].nsim == B1.maps[k].nsim
    rz = Bz.fit()
    assert np.isfinite(rz["alpha0"]) and rz["population_weights"]
    # min_weight=0: same as simulation files without those rows
    Bm = _bbc(cfg, paths, sims.with_weights(w, min_weight=0.0))
    rm = Bm.fit()
    keep_lowz = w[:15000] > 0
    keep_des = w[15000:] > 0
    newp = []
    for name, keep in (("sim_lowz", keep_lowz), ("sim_des", keep_des)):
        with gzip.open(paths[name], "rt") as f:
            head = [next(f) for _ in range(2)]
        df = pd.read_csv(paths[name], sep=r"\s+", skiprows=2, header=None, names=["KEY"] + head[1].split()[1:], dtype=str)
        out = str(tmp_path / f"{name}.FITRES.gz")
        with gzip.open(out, "wt") as f:
            f.write(head[0] + head[1])
            for row in df[keep].itertuples(index=False):
                f.write(" ".join(row) + "\n")
        newp.append(out)
    sims_del = BiasCorSims(newp, cfg)
    Bd = _bbc(cfg, paths, sims_del)
    rd = Bd.fit()
    for k in ("alpha0", "beta0", "gamma0", "sigint"):
        assert rm[k] == pytest.approx(rd[k], abs=1e-12)
    assert np.allclose(Bm.mubias, Bd.mubias, equal_nan=True, rtol=0, atol=1e-12)
    for k in Bd.maps:
        assert np.array_equal(Bm.maps[k].N, Bd.maps[k].N)


def test_zero_weight_cell_raises(toyset):
    paths, cfg, sims = toyset
    w = np.ones(sims.nrow)
    w[sims.S["c"] > 0.1] = 0.0                     # whole colour cells with weight 0 but enough rows
    with pytest.raises(RuntimeError, match="min_weight"):
        _bbc(cfg, paths, sims.with_weights(w))
    B = _bbc(cfg, paths, sims.with_weights(w, min_weight=0.0))
    assert B.fit()["min_weight"] == 0.0


# ---------------------------------------------------------------- map_weights (research option, not SALT2mu)
def _bbc_mw(cfg, paths, sims, mw):
    return BBC(cfg, read_fitres(paths["data"]), sims, survey_ids=toy.SURVEY_IDS, cid_reject=[], map_weights=mw).prepare()


def _hand_val(B, sims, mw):
    """VAL and AVG_c of each sample's 5D map with weights WGT_POP * mw, computed directly."""
    S = sims.S
    out = {}
    for k, M in B.maps.items():
        r = M.rows
        w = mw[r] * (1.0 if sims.weights is None else sims.weights[r])
        NC = M.N.size
        sw = np.bincount(M.J, w, NC)
        b = S["fitmu"][r].astype(float) - S["SIM_MUz"][r].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = (np.bincount(M.J, w * b, NC) / sw, np.bincount(M.J, w * S["c"][r].astype(float), NC) / sw)
    return out


def test_map_weights_default_reproduces_and_accessors(toyset):
    paths, cfg, sims = toyset
    B0 = _bbc(cfg, paths, sims)
    r0 = B0.fit()
    rows, smp = B0.map_rows()
    assert np.array_equal(rows, np.sort(np.concatenate([M.rows for M in B0.maps.values()])))
    for k, M in B0.maps.items():
        assert np.array_equal(rows[smp == k], M.rows)
    mw = B0.default_map_weights()
    assert (mw[rows] > 0).all() and np.count_nonzero(mw) == len(rows)
    B1 = _bbc_mw(cfg, paths, sims, mw)
    r1 = B1.fit()
    assert r1["map_weights"] and not r0["map_weights"]
    for k in ("alpha0", "beta0", "gamma0", "sigint", "AVEMAG0"):
        assert r0[k] == r1[k]
    for a in ("mubias", "covadd", "covscale", "MU", "MUERR", "cutflag_fit"):
        assert np.array_equal(getattr(B0, a), getattr(B1, a), equal_nan=True)
    # mucov_at reproduces the per-event corrections of the data
    u = np.isfinite(B0.mubias)
    d = B0.data
    add, scl, ok = B0.mucov_at(B0.data_sample[u], d["zHD"][u].astype(float), d["__HOST__"][u], d["c"][u].astype(float))
    assert ok.all() and np.array_equal(add, B0.covadd[u]) and np.array_equal(scl, B0.covscale[u])
    a1, s1, o1 = B0.mucov_at(1, [0.3], [10.5], [0.0])
    assert a1.shape == (1,) and o1.dtype == bool
    with pytest.raises(ValueError):
        B0.mucov_at(99, [0.3], [10.5], [0.0])
    with pytest.raises(RuntimeError):
        BBC(cfg, read_fitres(paths["data"]), sims, survey_ids=toy.SURVEY_IDS, cid_reject=[]).map_rows()


def test_map_weights_replace_inverse_variance(toyset):
    """map_weights replace 1/muerr^2 in VAL and AVG; the counts and the map sigint are unchanged."""
    paths, cfg, sims = toyset
    B0 = _bbc(cfg, paths, sims)
    mw = np.random.default_rng(11).uniform(10.0, 200.0, sims.nrow)
    B = _bbc_mw(cfg, paths, sims, mw)
    assert B.sigint_biascor == B0.sigint_biascor
    hand = _hand_val(B, sims, mw)
    for k, M in B.maps.items():
        assert np.array_equal(M.N, B0.maps[k].N) and np.array_equal(M.rows, B0.maps[k].rows)
        used = M.N >= 1
        assert np.allclose(M.val[used], hand[k][0][used], rtol=0, atol=1e-12)
        assert np.allclose(M.avg["c"][used], hand[k][1][used], rtol=0, atol=1e-12)
        assert not np.allclose(M.val[used], B0.maps[k].val[used], rtol=0, atol=1e-9)
    # scaling all map weights by a constant leaves the maps unchanged (up to rounding)
    B2 = _bbc_mw(cfg, paths, sims, 3.0 * mw)
    for k in B.maps:
        used = B.maps[k].N >= 1
        assert np.allclose(B.maps[k].val[used], B2.maps[k].val[used], rtol=0, atol=1e-12)
    assert np.isfinite(B.fit()["alpha0"])


def test_map_weights_zero_and_validation(toyset):
    paths, cfg, sims = toyset
    B0 = _bbc(cfg, paths, sims)
    mw = B0.default_map_weights()
    rng = np.random.default_rng(13)
    mz = np.where(rng.uniform(size=sims.nrow) < 0.03, 0.0, mw)       # zero map weight: still counted
    Bz = _bbc_mw(cfg, paths, sims, mz)
    for k in B0.maps:
        assert np.array_equal(Bz.maps[k].N, B0.maps[k].N)
    assert np.isfinite(Bz.fit()["alpha0"])
    dead = np.where(sims.S["c"] > 0.1, 0.0, mw)                        # whole colour cells with weight 0
    with pytest.raises(RuntimeError, match="map_weights"):
        _bbc_mw(cfg, paths, sims, dead)
    data = read_fitres(paths["data"])
    for bad in (mw[:-1], -mw, np.where(np.arange(sims.nrow) == 5, np.nan, mw), np.where(np.arange(sims.nrow) == 5, np.inf, mw)):
        with pytest.raises(ValueError, match="map_weights"):
            BBC(cfg, data, sims, survey_ids=toy.SURVEY_IDS, cid_reject=[], map_weights=bad)


def test_map_weights_with_weights_view(toyset):
    """With a population-weight view, the map weight is WGT_POP * map_weights; default weights reproduce the view."""
    paths, cfg, sims = toyset
    w = np.random.default_rng(17).uniform(0.3, 1.7, sims.nrow)
    for v in (sims.with_weights(w), sims.with_weights(np.where(w > 0.5, w, 0.0), min_weight=0.0)):
        Bv = _bbc(cfg, paths, v)
        rv = Bv.fit()
        mw = Bv.default_map_weights()
        B1 = _bbc_mw(cfg, paths, v, mw)
        r1 = B1.fit()
        for k in ("alpha0", "beta0", "gamma0", "sigint"):
            assert rv[k] == r1[k]
        assert np.array_equal(Bv.mubias, B1.mubias, equal_nan=True)
        if v.keep is not None:
            rows, _ = Bv.map_rows()
            assert v.keep[rows].all()
        mr = np.random.default_rng(19).uniform(10.0, 200.0, sims.nrow)
        B2 = _bbc_mw(cfg, paths, v, mr)
        hand = _hand_val(B2, v, mr)
        for k, M in B2.maps.items():
            used = M.N >= 1
            assert np.allclose(M.val[used], hand[k][0][used], rtol=0, atol=1e-12)
            assert np.array_equal(M.N, Bv.maps[k].N)


# ---------------------------------------------------------------- undefined weighted medians (non-SNANA "standard" policy)
def _conventional_median(v, w):
    """value of the first sorted element whose cumulative weight reaches half the total (the largest if none does)"""
    v, w = np.asarray(v, float), np.asarray(w, float)
    o = gsl_sort_index(v)
    i = min(int(np.searchsorted(np.cumsum(w[o]), np.cumsum(w)[-1] / 2, side="left")), len(v) - 1)
    return float(v[o[i]])


def test_undefined_median_default_raises_both_cases(monkeypatch):
    import bbclocal.maps as maps
    with pytest.raises(RuntimeError, match="reads past the end"):                       # (b) last element, even count
        weighted_median([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="reads past the end"):
        weighted_median([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0], undefined_median="raise")
    monkeypatch.setattr(maps, "_crossing_index", lambda cs, half: len(cs))              # (a) running sum never reaches half
    with pytest.raises(RuntimeError, match="never reaches half"):
        weighted_median([3.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    fired = []
    assert weighted_median([3.0, 1.0, 2.0], [1.0, 1.0, 1.0], undefined_median="standard", fired=fired) == 3.0
    assert fired == ["rounding"]


def test_undefined_median_standard_values():
    fired = []
    assert weighted_median([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0], "standard", fired) == 4.0
    assert weighted_median([4.0, 3.0, 1.0, 2.0], [3.0, 1.0, 0.0, 0.0], "standard", fired) == 4.0   # cumsum 0,0,1,4; half 2
    assert weighted_median([0.5, 0.2, 0.9, 0.7], [0.1, 0.2, 0.6, 0.1], "standard", fired) == 0.9
    assert fired == ["last_even"] * 3
    assert weighted_median([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.0, 0.0], "standard", fired) == 1.5   # defined: unchanged
    assert len(fired) == 3
    with pytest.raises(RuntimeError, match="total weight 0"):                           # SALT2mu aborts: still raises
        weighted_median([1.0, 2.0], [0.0, 0.0], undefined_median="standard")
    with pytest.raises(ValueError, match="undefined_median"):
        weighted_median([1.0, 2.0], [1.0, 1.0], undefined_median="mean")


@pytest.mark.parametrize("seed", range(300))
def test_undefined_median_policy_changes_only_undefined_cases(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(1, 12))
    v = rng.integers(0, 4, n).astype(float) if seed % 3 == 0 else rng.normal(size=n)
    w = rng.choice([0.0, 0.3, 1.0, 4.0], n) if seed % 2 else rng.uniform(0, 1, n) ** 4
    if not w.sum() > 0:
        w[0] = 1.0
    fired = []
    try:
        ref = weighted_median(v, w)
    except RuntimeError:
        ref = None
    got = weighted_median(v, w, undefined_median="standard", fired=fired)
    if ref is not None:
        assert got == ref and fired == []
        assert type(got) is type(ref)
    else:
        assert fired == ["last_even"] and n % 2 == 0
        assert got == _conventional_median(v, w) == v.max()


def test_undefined_median_propagates_to_sigint_scan_and_bbc(toyset, monkeypatch):
    """the policy reaches every weighted median (MUCOVSCALE, weighted MAD scans) and the fallbacks are counted"""
    import bbclocal.maps as maps
    rng = np.random.default_rng(23)
    res, cov, w = rng.normal(0, 0.15, 40), np.full(40, 0.01), rng.uniform(0.5, 1.5, 40)
    w2 = np.where(np.abs(res) == np.abs(res).max(), 1e6, w)          # the largest |residual| carries almost all the weight
    with pytest.raises(RuntimeError, match="reads past the end"):
        sigint_scan(res, cov, use_mad=True, abort=False, wgt=w2)
    fired = []
    sigint_scan(res, cov, use_mad=True, abort=False, wgt=w2, undefined_median="standard", fired=fired)
    assert len(fired) > 0 and set(fired) == {"last_even"}
    assert sigint_scan(res, cov, use_mad=True, wgt=w, undefined_median="standard") == sigint_scan(res, cov, use_mad=True,
                                                                                                    wgt=w)
    paths, cfg, sims = toyset
    v = sims.with_weights(np.random.default_rng(29).uniform(0.5, 1.5, sims.nrow))
    with pytest.raises(ValueError, match="undefined_median"):
        BBC(cfg, read_fitres(paths["data"]), v, survey_ids=toy.SURVEY_IDS, cid_reject=[], undefined_median="x")
    Br = _bbc(cfg, paths, v)
    rr = Br.fit()
    assert Br.undefined_median == "raise" and Br.n_undefined_median == 0
    assert rr["undefined_median"] == "raise" and rr["n_undefined_median"] == 0
    seen = []
    real = maps.weighted_median

    def spy(vals, wgt=None, undefined_median="raise", fired=None):
        seen.append(undefined_median)
        out = real(vals, wgt, undefined_median, fired)
        if fired is not None and len(seen) % 2 == 0:
            fired.append("last_even")                       # pretend every second median used the fallback
        return out
    monkeypatch.setattr(maps, "weighted_median", spy)
    Bs = BBC(cfg, read_fitres(paths["data"]), v, survey_ids=toy.SURVEY_IDS, cid_reject=[],
             undefined_median="standard").prepare()
    rs = Bs.fit()
    assert seen and set(seen) == {"standard"}
    assert Bs.n_undefined_median == len(seen) // 2 == rs["n_undefined_median"] > 0
    assert rs["undefined_median"] == "standard"
    for k in ("alpha0", "beta0", "gamma0", "sigint"):                # no undefined case in this toy: identical results
        assert rs[k] == rr[k]
    assert np.array_equal(Bs.mubias, Br.mubias, equal_nan=True)
