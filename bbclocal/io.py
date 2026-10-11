"""Reading SNANA FITRES tables (text, optionally gzipped), bias-correction simulations, SURVEY.DEF and CID lists.

All numeric columns are stored as float32, as SALT2mu stores them (values are parsed as float64 and then cast; SALT2mu parses
to long double and casts, which can differ only in rare double-rounding cases).
"""
import gzip
import json
import os

import numpy as np
import pandas as pd

F32 = np.float32
STRING_COLS = ("CID", "FIELD")
INT_COLS = ("IDSURVEY",)

# columns SALT2mu reads from every bias-correction file in the supported configuration
BIASCOR_COLS = ["IDSURVEY", "FIELD", "zHD", "zHEL", "VPECERR", "SNRMAX1", "x0", "x0ERR", "x1", "x1ERR", "c", "cERR",
                "COV_x1_c", "COV_x1_x0", "COV_c_x0", "SIM_WGT_POP", "SIM_ZCMB", "SIM_DLMAG", "SIM_AV"]
SIM_PAR_COLS = ["SIM_alpha", "SIM_beta", "SIM_gammaDM"]
DATA_COLS = ["CID", "IDSURVEY", "FIELD", "zHD", "zHEL", "VPECERR", "x0", "x0ERR", "x1", "x1ERR", "c", "cERR",
             "COV_x1_c", "COV_x1_x0", "COV_c_x0"]


def _open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def fitres_header(path):
    """Return (number of lines before the VARNAMES line, list of variable names)."""
    with _open_text(path) as f:
        for n, line in enumerate(f):
            s = line.strip()
            if s.startswith("VARNAMES:"):
                return n, s.split()[1:]
    raise ValueError(f"{path}: no VARNAMES line")


def read_fitres(path, columns=None, required=()):
    """Read a SNANA FITRES text table into a dict of numpy arrays: CID/FIELD as str, IDSURVEY as int, other numeric columns
    as float32, non-numeric columns as str. columns: subset to read (missing optional columns are skipped)."""
    nskip, names = fitres_header(path)
    missing = [c for c in required if c not in names]
    if missing:
        raise KeyError(f"{path}: missing required columns {missing}")
    use = names if columns is None else [c for c in names if c in set(columns)]
    df = pd.read_csv(path, sep=r"\s+", comment="#", skiprows=nskip + 1, header=None, names=["__KEY__"] + names,
                     usecols=use, dtype={c: str for c in STRING_COLS if c in names}, engine="c")
    out = {}
    for c in use:
        v = df[c].to_numpy()
        if c in STRING_COLS:
            out[c] = v.astype(str)
        elif c in INT_COLS:
            out[c] = v.astype(np.int64)
        elif np.issubdtype(v.dtype, np.number):
            out[c] = v.astype(np.float64).astype(F32)
        else:
            out[c] = v.astype(str)
    return out


def read_cid_list(path):
    """cid_reject_file: one CID per line (first token); lines starting with '#' are comments."""
    out = []
    with _open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.replace(",", " ").split()
            if tok[0].upper() in ("CID", "SNID", "VARNAMES:"):
                continue
            out.append(tok[1] if tok[0].endswith(":") and len(tok) > 1 else tok[0])
    return out


def read_survey_def(path):
    """Parse a SNANA SURVEY.DEF file ('SURVEY: <NAME> <IDSURVEY> ...') into {NAME: IDSURVEY}."""
    out = {}
    with open(path) as f:
        for line in f:
            tok = line.split("#")[0].split()
            if len(tok) >= 3 and tok[0] == "SURVEY:":
                try:
                    out[tok[1]] = int(tok[2])
                except ValueError:
                    pass
    return out


# ---------------------------------------------------------------- bias-correction simulations
def _summarize_sim_pars(t, n_first=20):
    """Information SALT2mu derives from ALL rows before cuts: the dust flag (set_DUST_FLAG_biasCor: any of the first 20 rows
    with SIM_AV > 0) and the distinct SIM_alpha / SIM_beta / SIM_gammaDM / SIM_WGT_POP values."""
    meta = {"nrow_read": int(len(t["zHD"]))}
    av = t.get("SIM_AV")
    meta["dust_flag_first20"] = None if av is None else bool((np.asarray(av[:n_first], float) > 0).any())
    for c in SIM_PAR_COLS + ["SIM_WGT_POP"]:
        if c in t:
            u = np.unique(np.asarray(t[c], float))
            meta[c + "_unique"] = [float(x) for x in u[:10]]
            meta[c + "_nunique"] = int(len(u))
    return meta


def read_biascor_fitres(path, extra_cols=()):
    """Read an original SNANA bias-correction FITRES file; returns (table, meta). Rows with SIM_TEMPLATE_INDEX != 0 are
    dropped here, exactly as SALT2mu rejects them."""
    nskip, names = fitres_header(path)
    want = set(BIASCOR_COLS + SIM_PAR_COLS + ["SIM_TEMPLATE_INDEX"] + list(extra_cols))
    t = read_fitres(path, columns=want, required=[c for c in BIASCOR_COLS if c != "SIM_AV"])
    meta = _summarize_sim_pars(t)
    meta["source"] = os.path.basename(str(path))
    if "SIM_TEMPLATE_INDEX" in t:
        keep = t.pop("SIM_TEMPLATE_INDEX") == 0
        t = {k: v[keep] for k, v in t.items()}
    meta["precuts"] = {}
    names, code = np.unique(t.pop("FIELD"), return_inverse=True)
    t["FIELD_NAMES"], t["FIELD_CODE"] = names.astype(str), code.ravel().astype(np.int64)
    return t, meta


def read_biascor_npz(path, trust_legacy=False):
    """Read a reduced bias-correction file written by bbclocal-extract; returns (table, meta).
    trust_legacy: accept an npz without bbclocal metadata (meta then says so, and the caller must state the dust flag)."""
    z = np.load(path, allow_pickle=False)
    t = {k: z[k] for k in z.files if k not in ("FIELD_NAMES", "FIELD_CODE", "__meta__")}
    t["FIELD_NAMES"] = np.asarray(z["FIELD_NAMES"]).astype(str)
    t["FIELD_CODE"] = np.asarray(z["FIELD_CODE"]).astype(np.int64)
    if "IDSURVEY" in t:
        t["IDSURVEY"] = t["IDSURVEY"].astype(np.int64)
    if "__meta__" in z.files:
        meta = json.loads(str(z["__meta__"]))
    elif trust_legacy:
        meta = {"legacy": True, "source": os.path.basename(str(path)), "dust_flag_first20": None, "precuts": None}
    else:
        raise ValueError(f"{path}: no bbclocal metadata (not written by bbclocal-extract); pass trust_legacy=True only if "
                         "you know how it was made")
    return t, meta


def read_biascor(path, trust_legacy=False, extra_cols=()):
    """Read one bias-correction file (original FITRES text or bbclocal-extract npz). The table has float32 columns,
    IDSURVEY (int), FIELD_CODE (int) and FIELD_NAMES (the names the codes point to)."""
    p = str(path)
    if p.endswith(".npz"):
        return read_biascor_npz(p, trust_legacy=trust_legacy)
    return read_biascor_fitres(p, extra_cols=extra_cols)
