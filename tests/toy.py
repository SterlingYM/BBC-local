"""Small synthetic bias-correction simulations and data in SNANA FITRES text format, for tests.

Two surveys: LOWZ (IDSURVEY 1, FIELD VOID, 0.02 < z < 0.1) and DES (IDSURVEY 10, fields C3 and X3, 0.1 < z < 0.8).
Model: mB = -30 + mu(z) - alpha x1 + beta c + gamma step(logmass) + intrinsic scatter, then Gaussian measurement noise on
mB, x1, c. The host step is in the data only (a step in the simulations would be absorbed by the host bins of the bias map).
No selection cut, so the bias correction only has to undo the regression effect of the measurement noise.
"""
import gzip

import numpy as np
import pandas as pd

from bbclocal.cosmo import Cosmology

ALPHA, BETA, GAMMA, SIGINT = 0.15, 3.1, 0.05, 0.10
ERR_MB, ERR_X1, ERR_C = 0.06, 0.25, 0.035

INPUT = """# toy SALT2mu input
opt_biascor=4336
surveygroup_biascor='LOWZ(zbin=0.02),DES(zbin=0.075)'
fieldGroup_biascor=C3,X3
zmin=0.02
zmax=0.85
nzbin=8
powzbin=3
min_per_zbin=1
lensing_zpar=0.055
interp_biascor_logmass=0
logmass_min=-15
logmass_max=35
nbin_logmass=2
x1min=-3.0
x1max=3.0
cmin=-0.3
cmax=0.3
CUTWIN x1ERR 0 1
CUTWIN FITPROB 0.001 1.1
sig1=0.04
redchi2_tol=0.01
fitflag_sigmb=2
blindflag=0
p1=0.15
p2=3.1
p5=0.00001
p7=10
p8=0.001
p9=0.685
p11=-1.0
u1=1
u2=3
u5=1
h0=70
"""
SURVEY_IDS = {"LOWZ": 1, "DES": 10}


def _rows(rng, n, zlo, zhi, idsurvey, fields, start_cid, gamma=0.0):
    cosmo = Cosmology()
    z = rng.uniform(zlo, zhi, n)
    mu = cosmo.mu(z, z)
    x1 = rng.normal(0.0, 1.0, n)
    c = rng.normal(0.0, 0.08, n)
    m = rng.uniform(8.0, 12.0, n)
    step = gamma * np.where(m > 10.0, -0.5, 0.5)
    mB = -30.0 + mu - ALPHA * x1 + BETA * c + step + rng.normal(0, SIGINT, n)
    mB_obs = mB + rng.normal(0, ERR_MB, n)
    x1_obs = x1 + rng.normal(0, ERR_X1, n)
    c_obs = c + rng.normal(0, ERR_C, n)
    x0 = 10 ** (-0.4 * mB_obs)
    df = pd.DataFrame({
        "CID": [str(start_cid + i) for i in range(n)], "IDSURVEY": idsurvey,
        "FIELD": rng.choice(fields, n), "zHD": z, "zHEL": z, "VPECERR": 250.0,
        "SNRMAX1": np.where(rng.uniform(size=n) < 0.3, 100.0, 30.0), "x0": x0, "x0ERR": ERR_MB * x0 * np.log(10) / 2.5,
        "x1": x1_obs, "x1ERR": ERR_X1, "c": c_obs, "cERR": ERR_C, "COV_x1_c": 0.0, "COV_x1_x0": 0.0, "COV_c_x0": 0.0,
        "HOST_LOGMASS": m, "PKMJDERR": 0.5, "FITPROB": 0.5, "SIM_WGT_POP": 1.0, "SIM_ZCMB": z, "SIM_DLMAG": mu,
        "SIM_AV": 0.1, "SIM_alpha": ALPHA, "SIM_beta": BETA, "SIM_gammaDM": 0.0, "SIM_TEMPLATE_INDEX": 0, "c_true": c})
    return df


def write_fitres(df, path):
    with gzip.open(path, "wt") as f:
        f.write("# toy FITRES\n")
        f.write("VARNAMES: " + " ".join(df.columns) + "\n")
        body = df.to_csv(sep=" ", header=False, index=False, float_format="%.7g")
        f.write("".join("SN: " + line + "\n" for line in body.splitlines()))


def make_toy(dirpath, nsim=40000, ndata=(300, 1200), seed=1):
    rng = np.random.default_rng(seed)
    sim_l = _rows(rng, nsim, 0.015, 0.13, 1, ["VOID"], 1)
    sim_d = _rows(rng, 2 * nsim, 0.08, 0.92, 10, ["C3", "X3"], 10 ** 6)
    dat = pd.concat([_rows(rng, ndata[0], 0.025, 0.1, 1, ["VOID"], 5 * 10 ** 6, GAMMA),
                     _rows(rng, ndata[1], 0.1, 0.8, 10, ["C3", "X3"], 6 * 10 ** 6, GAMMA)], ignore_index=True)
    paths = {"sim_lowz": f"{dirpath}/TOY_LOWZ.FITRES.gz", "sim_des": f"{dirpath}/TOY_DES.FITRES.gz",
             "data": f"{dirpath}/TOY_DATA.FITRES.gz", "input": f"{dirpath}/toy.input"}
    write_fitres(sim_l, paths["sim_lowz"])
    write_fitres(sim_d, paths["sim_des"])
    write_fitres(dat.drop(columns=["SIM_TEMPLATE_INDEX"]), paths["data"])
    with open(paths["input"], "w") as f:
        f.write(INPUT)
    return paths
