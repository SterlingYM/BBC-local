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
