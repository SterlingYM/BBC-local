"""Compare one bbclocal fit with the output of SNANA's SALT2mu for the same inputs.

Example:
    python examples/validate_against_snana.py \
        --input SALT2mu.input --data INPUT_0001.FITRES.gz \
        --biascor LOWZ_BIASCOR.npz FOUND_BIASCOR.npz DES_BIASCOR.npz \
        --snana-fitres OUTPUT_BBCFIT-0001/FITOPT000_MUOPT001.FITRES.gz \
        --muopt "chi2max=10000000" --cid-reject exclude_SN.txt

The SALT2mu result is read from the header of its output FITRES file (alpha0, beta0, gamma0, sigint, NSNFIT) and the
per-SN distances from its rows with CUTMASK == 0. Use the same --muopt tokens as the SNANA MUOPT that produced the file.
"""
import argparse
import gzip
import re
import time

import numpy as np
import pandas as pd

from bbclocal import BBC, BiasCorSims, read_cid_list, read_fitres, read_salt2mu_input, read_survey_def


def snana_header(path):
    head = {}
    pat = re.compile(r"^#\s+(alpha0|beta0|gamma0)\s+=\s+(\S+)\s+\+-\s+(\S+)")
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        for line in f:
            if line.startswith("SN:"):
                break
            m = pat.match(line)
            if m:
                head[m.group(1)] = float(m.group(2))
                head[m.group(1) + "_err"] = float(m.group(3))
            m = re.match(r"^#\s+sigint\s+=\s+(\S+)", line)
            if m:
                head["sigint"] = float(m.group(1))
            m = re.match(r"^#\s+NSNFIT\s+=\s+(\d+)", line)
            if m:
                head["NSNFIT"] = int(m.group(1))
            m = re.match(r"^#\s+M0avg\s+=\s+(\S+)", line)
            if m:
                head["M0avg"] = float(m.group(1))
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--biascor", nargs="+", required=True)
    ap.add_argument("--snana-fitres", required=True)
    ap.add_argument("--muopt", default=None)
    ap.add_argument("--cid-reject", default=None)
    ap.add_argument("--survey-def", default=None)
    ap.add_argument("--survey-id", action="append")
    ap.add_argument("--dust-flag", default=None, choices=["true", "false"])
    ap.add_argument("--trust-legacy-npz", action="store_true")
    a = ap.parse_args()

    ids = read_survey_def(a.survey_def) if a.survey_def else {}
    for item in a.survey_id or []:
        for tok in item.split(","):
            k, _, v = tok.partition("=")
            ids[k] = int(v)
    cfg = read_salt2mu_input(a.input, muopt=a.muopt).validate()
    t0 = time.time()
    dust = None if a.dust_flag is None else a.dust_flag == "true"
    sims = BiasCorSims(a.biascor, cfg, dust_flag=dust, trust_legacy=a.trust_legacy_npz)
    t1 = time.time()
    data = read_fitres(a.data)
    rej = read_cid_list(a.cid_reject) if a.cid_reject else []
    B = BBC(cfg, data, sims, survey_ids=ids or None, cid_reject=rej).prepare()
    r = B.fit()
    t2 = time.time()

    h = snana_header(a.snana_fitres)
    print(f"read+derive sims {t1 - t0:.1f} s, prepare+fit {t2 - t1:.1f} s")
    print(f"{'':10s} {'SNANA':>12s} {'bbclocal':>12s}")
    for k, fmt in (("alpha0", "%.5f"), ("beta0", "%.5f"), ("gamma0", "%.5f"), ("sigint", "%.4f")):
        print(f"{k:10s} {fmt % h.get(k, np.nan):>12s} {fmt % r[k]:>12s}")
    for k in ("alpha0", "beta0", "gamma0"):
        if k + "_err" in h and r.get(k + "_err") is not None:
            print(f"{k + '_err':10s} {h[k + '_err']:12.5f} {r[k + '_err']:12.5f}   (bbclocal: finite-difference Hessian)")
    print(f"{'NSNFIT':10s} {h.get('NSNFIT', -1):12d} {r['NSNFIT']:12d}")
    if "M0avg" in h:
        print(f"{'M0avg':10s} {h['M0avg']:12.5f} {r['AVEMAG0']:12.5f}")

    sn = pd.read_csv(a.snana_fitres, sep=r"\s+", comment="#", dtype={"CID": str})
    sn = sn[sn.CUTMASK == 0] if "CUTMASK" in sn else sn
    loc = B.table()
    loc = loc[loc.CUTFLAG == 0]
    key_s = sn.CID.astype(str) + "_" + sn.IDSURVEY.astype(str)
    key_l = loc.CID.astype(str) + "_" + loc.IDSURVEY.astype(str)
    only_s, only_l = set(key_s) - set(key_l), set(key_l) - set(key_s)
    m = sn.assign(key=key_s.values).merge(loc.assign(key=key_l.values), on="key", suffixes=("", "_loc"))
    d = (m.MU_loc - m.MU) * 1e3
    de = (m.MUERR_loc - m.MUERR) * 1e3
    print(f"SNe: SNANA {len(sn)}, bbclocal {len(loc)}, common {len(m)}, only SNANA {len(only_s)}, only bbclocal {len(only_l)}")
    print(f"MU - MU_SNANA [mmag]: mean {d.mean():.4f}, rms {d.std():.4f}, max |.| {d.abs().max():.4f}")
    print(f"MUERR - MUERR_SNANA [mmag]: rms {de.std():.4f}, max |.| {de.abs().max():.4f}")
    if "biasCor_mu" in m:
        db = ((m.biasCor_mu_loc - m.biasCor_mu_loc.mean()) - (m.biasCor_mu - m.biasCor_mu.mean())) * 1e3
        print(f"biasCor_mu (mean removed) - SNANA [mmag]: rms {db.std():.4f}, max |.| {db.abs().max():.4f} "
              "(SNANA prints 4 decimals)")


if __name__ == "__main__":
    main()
