"""Writing fit outputs: a FITRES-like text table and a YAML summary."""
import gzip

import numpy as np
import yaml

_FMT = {"MU": "%.5f", "MUMODEL": "%.5f", "MUERR": "%.5f", "MURES": "%.5f", "biasCor_mu": "%.5f",
        "biasCor_muCOVSCALE": "%.4f", "biasCor_muCOVADD": "%.6f", "zHD": "%.6f", "zHEL": "%.6f"}


def write_fitres(df, path, header_lines=()):
    """Write a pandas DataFrame as a SNANA-style FITRES text file (VARNAMES line, one 'SN:' row per event)."""
    op = gzip.open if str(path).endswith(".gz") else open
    cols = list(df.columns)
    with op(path, "wt") as f:
        for h in header_lines:
            f.write(f"# {h}\n")
        f.write("VARNAMES: " + " ".join(cols) + "\n")
        fmt = []
        for c in cols:
            if c in _FMT:
                fmt.append(_FMT[c])
            elif np.issubdtype(df[c].dtype, np.integer):
                fmt.append("%d")
            elif np.issubdtype(df[c].dtype, np.floating):
                fmt.append("%.6g")
            else:
                fmt.append("%s")
        vals = [df[c].to_numpy() for c in cols]
        for i in range(len(df)):
            f.write("SN: " + " ".join(fm % v[i] for fm, v in zip(fmt, vals)) + "\n")


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_yaml(result, path):
    with open(path, "w") as f:
        yaml.safe_dump(_plain(result), f, sort_keys=False)
