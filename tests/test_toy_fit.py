"""End-to-end test on a small synthetic data set: the fit must recover the input alpha, beta, gamma and sigint, the FITRES
and extracted-npz routes must give identical results, and the CLI must run."""
import numpy as np
import pytest

from bbclocal import BBC, BiasCorSims, read_fitres, read_salt2mu_input
from bbclocal.cli import main as cli_main
from bbclocal.config import UnsupportedOptionError
from bbclocal.covfit import wfit_grid
from bbclocal.extract import extract

import toy


@pytest.fixture(scope="module")
def toyfiles(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    return toy.make_toy(str(d))


@pytest.fixture(scope="module")
def fitted(toyfiles):
    cfg = read_salt2mu_input(toyfiles["input"]).validate()
    sims = BiasCorSims([toyfiles["sim_lowz"], toyfiles["sim_des"]], cfg)
    B = BBC(cfg, read_fitres(toyfiles["data"]), sims, survey_ids=toy.SURVEY_IDS, cid_reject=[]).prepare()
    return cfg, sims, B, B.fit()


def test_recovers_input_parameters(fitted):
    cfg, sims, B, r = fitted
    assert sims.dust_flag
    assert abs(r["alpha0"] - toy.ALPHA) < 3 * r["alpha0_err"]
    assert abs(r["beta0"] - toy.BETA) < 3 * r["beta0_err"]
    assert abs(r["gamma0"] - toy.GAMMA) < 3 * r["gamma0_err"]
    # with MUCOVADD the intrinsic scatter of the simulations enters each event's error through the MUCOVADD map, so the
    # fitted sigint is only the excess over it (near 0 here, since data and simulations share the same scatter)
    used = B.cutflag_fit == 0
    assert 0.06 < np.sqrt(np.median(B.covadd[used])) < 0.14
    assert r["sigint"] < 0.06
    assert abs(r["chi2red"] - 1) < cfg.redchi2_tol
    assert sorted(s["name"] for s in r["samples"]) == ["DES(C3)", "DES(X3)", "LOWZ"]
    assert r["NSNFIT"] > 1400


def test_bias_correction_removes_regression_bias(fitted):
    cfg, sims, B, r = fitted
    t = B.table()
    u = t.CUTFLAG == 0
    # with the bias correction, Hubble residuals show no trend with observed color
    res = (t.MU - t.MUMODEL)[u]
    slope = np.polyfit(B.data["c"][u.values].astype(float), res - res.mean(), 1)[0]
    assert abs(slope) < 0.5
    assert np.all(np.isfinite(t.loc[u, "biasCor_mu"]))


def test_fitres_and_npz_routes_identical(fitted, toyfiles, tmp_path):
    cfg, sims, B, r = fitted
    npz = [extract(toyfiles[k], str(tmp_path), cfg=cfg, verbose=False) for k in ("sim_lowz", "sim_des")]
    sims2 = BiasCorSims(npz, cfg)
    B2 = BBC(cfg, read_fitres(toyfiles["data"]), sims2, survey_ids=toy.SURVEY_IDS, cid_reject=[]).prepare()
    r2 = B2.fit()
    for k in ("alpha0", "beta0", "gamma0", "sigint", "NSNFIT"):
        assert r2[k] == r[k], k
    assert np.array_equal(np.nan_to_num(B2.MU), np.nan_to_num(B.MU))
    # a configuration that does not imply the pre-applied cuts is refused
    looser = cfg.copy()
    looser.cutwins = [c for c in cfg.cutwins if c.var != "x1ERR"]
    with pytest.raises(UnsupportedOptionError):
        BBC(looser, read_fitres(toyfiles["data"]), sims2, survey_ids=toy.SURVEY_IDS, cid_reject=[])


def test_population_cut_and_fixed_ab(fitted, toyfiles):
    cfg, sims, B, r = fitted
    c2 = cfg.apply_muopt("CUTWIN c -0.3 0.0 chi2max=10000000").validate()
    B2 = BBC(c2, read_fitres(toyfiles["data"]), sims, survey_ids=toy.SURVEY_IDS, cid_reject=[]).prepare()
    r2 = B2.fit()
    assert r2["NSNFIT"] < r["NSNFIT"]
    assert B2.data["c"][B2.cutflag_fit == 0].max() <= 0.0
    r3 = B2.fit(fix_ab=(0.15, 3.1))
    assert r3["alpha0"] == 0.15 and r3["beta0"] == 3.1 and not r3["floated"]["alpha"]


def test_chi2max_rejects_outliers(toyfiles, fitted):
    cfg, sims, B, r = fitted
    c2 = cfg.apply_muopt("chi2max=4")
    B2 = BBC(c2, read_fitres(toyfiles["data"]), sims, survey_ids=toy.SURVEY_IDS, cid_reject=[]).prepare()
    r2 = B2.fit()
    assert r2["n_chi2max_reject"] > 0 and r2["NSNFIT"] == r["NSNFIT"] - r2["n_chi2max_reject"]


def test_cid_reject(toyfiles, fitted):
    cfg, sims, B, r = fitted
    data = read_fitres(toyfiles["data"])
    B2 = BBC(cfg, data, sims, survey_ids=toy.SURVEY_IDS, cid_reject=list(data["CID"][:10])).prepare()
    assert B2.fit()["NSNFIT"] <= r["NSNFIT"] - 1


def test_refuses_alpha_beta_dependent_sims(toyfiles):
    cfg = read_salt2mu_input(toyfiles["input"]).validate()
    with pytest.raises(UnsupportedOptionError):
        BiasCorSims([toyfiles["sim_lowz"]], cfg.copy(p={**cfg.p, 2: 2.9}), dust_flag=False)


def test_cli(toyfiles, tmp_path):
    out = str(tmp_path / "toyfit")
    rc = cli_main(["fit", "--input", toyfiles["input"], "--data", toyfiles["data"], "--biascor", toyfiles["sim_lowz"],
                   toyfiles["sim_des"], "--survey-id", "LOWZ=1,DES=10", "--out", out])
    assert rc == 0
    t = read_fitres(out + ".FITRES.gz")
    assert "MU" in t and "CUTFLAG" in t


def test_wfit_on_fitted_distances(fitted):
    cfg, sims, B, r = fitted
    t = B.table()
    t = t[t.CUTFLAG == 0]
    res = wfit_grid(t.zHD.values, t.MU.values, t.MUERR.values, ompri=0.315, dompri=0.007)
    assert abs(res["w"] + 1) < 3 * res["wsig_marg"]
