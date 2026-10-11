"""bbclocal-extract: one-time reduction of SNANA bias-correction FITRES files to compact npz files.

Only the columns bbclocal needs are kept (stored as float32, as SALT2mu stores them), and only cuts that cannot change the
result are applied:
  * always: SIM_TEMPLATE_INDEX == 0 (SALT2mu rejects other rows) and BADERR (x0ERR, x1ERR or cERR <= 0);
  * with --input: the CUTWINs of that SALT2mu input that apply to the simulations (not DATAONLY, not on varname_gamma), the
    x1 and c windows, and zHD >= zmin.
The cuts applied are recorded in the file, and bbclocal refuses to use the file with a configuration that does not imply
them. Information that SALT2mu derives from the rows before cuts (the dust flag from the first 20 rows, the distinct values of
SIM_alpha, SIM_beta, SIM_gammaDM and SIM_WGT_POP) is recorded too.
"""
import argparse
import json
import os
import sys

import numpy as np

from .config import read_salt2mu_input
from .io import BIASCOR_COLS, SIM_PAR_COLS, fitres_header, read_fitres, _summarize_sim_pars

ALWAYS_KEEP = ["HOST_LOGMASS"]


def extract(path, outdir, cfg=None, extra_cols=(), verbose=True):
    _, names = fitres_header(path)
    keep = list(dict.fromkeys(BIASCOR_COLS + ALWAYS_KEEP + list(extra_cols)
                              + ([cfg.varname_gamma] + [c.var for c in cfg.cutwins if not c.dataonly] if cfg else [])))
    missing = [c for c in keep if c not in names and c != "SIM_AV"]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")
    read = [c for c in keep if c in names] + [c for c in SIM_PAR_COLS + ["SIM_TEMPLATE_INDEX"] if c in names]
    t = read_fitres(path, columns=read)
    meta = _summarize_sim_pars(t)
    meta["source"] = os.path.basename(str(path))
    n0 = len(t["zHD"])
    ok = (t["x0ERR"] > 0) & (t["x1ERR"] > 0) & (t["cERR"] > 0)
    if "SIM_TEMPLATE_INDEX" in t:
        ok &= t["SIM_TEMPLATE_INDEX"] == 0
    precuts = {}

    def win(var, lo, hi):
        v = np.asarray(t[var], np.float32).astype(float)
        precuts[var] = [None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi)]
        return (v >= lo) & (v <= hi)
    if cfg is not None:
        for cw in cfg.cutwins:
            if not cw.dataonly and cw.var != cfg.varname_gamma:
                ok &= win(cw.var, cw.lo, cw.hi)
        ok &= win("x1", cfg.x1min, cfg.x1max) & win("c", cfg.cmin, cfg.cmax) & win("zHD", cfg.zmin, np.inf)
    meta["precuts"] = precuts
    meta["nrow_kept"] = int(ok.sum())
    out = {c: t[c][ok] for c in keep if c in t and c not in ("FIELD", "IDSURVEY")}
    out["IDSURVEY"] = t["IDSURVEY"][ok].astype(np.int32)
    fields, code = np.unique(t["FIELD"][ok].astype(str), return_inverse=True)
    out["FIELD_CODE"] = code.ravel().astype(np.int16)
    out["FIELD_NAMES"] = fields
    out["__meta__"] = np.array(json.dumps(meta))
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(str(path))
    for suf in (".gz", ".FITRES", ".fitres", ".TEXT", ".txt"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    tmp = os.path.join(outdir, name + ".part.npz")
    np.savez_compressed(tmp, **out)
    final = os.path.join(outdir, name + ".npz")
    os.replace(tmp, final)
    if verbose:
        print(f"{name}: read {n0} rows, kept {int(ok.sum())}; dust flag {meta['dust_flag_first20']}; fields {list(fields)}",
              flush=True)
    return final


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bbclocal-extract", description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="bias-correction FITRES files (.gz ok)")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--input", help="SALT2mu input file whose simulation cuts may be pre-applied")
    ap.add_argument("--muopt", default=None, help="extra SALT2mu tokens applied on top of --input")
    ap.add_argument("--extra-cols", default="", help="comma-separated extra columns to keep (e.g. for later CUTWINs)")
    a = ap.parse_args(argv)
    cfg = read_salt2mu_input(a.input, muopt=a.muopt) if a.input else None
    extra = [c for c in a.extra_cols.split(",") if c]
    for f in a.files:
        extract(f, a.outdir, cfg=cfg, extra_cols=extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
