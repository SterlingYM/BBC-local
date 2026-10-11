"""Configuration of a BBC fit, read from a SNANA SALT2mu (BBC) input file.

Only one family of SALT2mu configurations is implemented (see README, "Supported configuration"). Every option that would make
SALT2mu behave differently from this implementation raises UnsupportedOptionError instead of being ignored.
"""
import os
import re
import shlex
import warnings
from dataclasses import dataclass, field, replace

SUPPORTED_OPT_BIASCOR = 4336      # 16 (5D map) + 32 (MUCOVSCALE) + 64 (per sample) + 128 (bias on MU) + 4096 (MUCOVADD)
OPT_BIASCOR_MAD = 2048            # added internally by SALT2mu when MUCOVADD is set
CHI2MAX_OFF = 1.0e8               # parse_chi2max: chi2max > 1e8 disables the cut and its M0 pre-step


class UnsupportedOptionError(ValueError):
    """A SALT2mu option (or a property of the inputs) that bbclocal does not reproduce."""


@dataclass
class CutWin:
    var: str
    lo: float
    hi: float
    dataonly: bool = False


@dataclass
class SurveyGroup:
    name: str                 # e.g. "CFA1+CFA2+CSP"
    members: list             # survey names or IDSURVEY integers as given
    zbin: float | None = None


@dataclass
class FieldGroup:
    name: str                 # e.g. "X1+X2"
    zbin: float | None = None


# keys whose value only affects printing/output files of SALT2mu, or that are irrelevant without a CC prior
_IGNORED_KEYS = {"prefix", "datafile", "errmask_write", "cutmask_write", "ndump_nobiascor", "dumpflag_nobiascor",
                 "surveygroup_biascor_abortflag", "idsurvey_list_probcc0", "nthread", "write_yaml", "write_csv",
                 "min_per_zbin", "fitflag_sigmb", "u13", "p13", "p14", "u14"}


@dataclass
class BBCConfig:
    """SALT2mu settings used by bbclocal. Defaults are SALT2mu's own defaults (SALT2mu.c set_defaults) where SALT2mu has one;
    the keys that the supported configuration requires are checked in validate()."""
    opt_biascor: int = 0
    simfile_biascor: list = field(default_factory=list)
    surveygroups: list = field(default_factory=list)      # [SurveyGroup]
    fieldgroups: list = field(default_factory=list)       # [FieldGroup]
    zmin: float = -0.01
    zmax: float = 9.99
    nzbin: int = 0
    powzbin: float = 0.0
    min_per_zbin: int = 1
    lensing_zpar: float = 0.0
    sig1: float = 0.0                 # starting sigint
    redchi2_tol: float = 0.02
    chi2max: float = 1.0e9
    x1min: float = -3.0
    x1max: float = 3.0
    cmin: float = -0.3
    cmax: float = 0.3
    logmass_min: float = -40.0
    logmass_max: float = 40.0
    nbin_logmass: int = 1
    interp_biascor_logmass: int = 1
    sigma_cell_biascor: float = 50.0
    min_per_cell_biascor: int = 3
    nbinc_mucovscale: int = 3
    varname_gamma: str = "HOST_LOGMASS"
    cutwins: list = field(default_factory=list)          # [CutWin]
    cid_reject_file: str | None = None
    h0: float = 70.0
    p: dict = field(default_factory=lambda: {i: 0.0 for i in range(1, 13)} | {7: 10.0})
    u: dict = field(default_factory=lambda: {i: 0 for i in range(1, 13)})
    blindflag: int | None = None
    other: dict = field(default_factory=dict)            # every other key=value seen (for the record and for refusals)

    # convenience ------------------------------------------------------------------------------------------------------
    @property
    def float_alpha(self):
        return self.u[1] != 0

    @property
    def float_beta(self):
        return self.u[2] != 0

    @property
    def float_gamma(self):
        return self.u[5] != 0

    @property
    def use_gamma0(self):
        """SALT2mu USE_GAMMA0: gamma floated, or its start value p5 non-zero."""
        return self.u[5] != 0 or abs(self.p[5]) > 1e-8

    @property
    def chi2max_on(self):
        return self.chi2max <= CHI2MAX_OFF

    def copy(self, **kw):
        c = replace(self, simfile_biascor=list(self.simfile_biascor), surveygroups=list(self.surveygroups),
                    fieldgroups=list(self.fieldgroups), cutwins=list(self.cutwins), p=dict(self.p), u=dict(self.u),
                    other=dict(self.other))
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def apply_muopt(self, muopt):
        """Return a copy with the tokens of a MUOPT string (same syntax as the input file) applied on top."""
        c = self.copy()
        _apply_tokens(c, _tokenize(muopt), source="muopt")
        return c

    def validate(self, allow_unknown=False):
        """Raise UnsupportedOptionError for every setting that bbclocal does not reproduce."""
        errs = []
        if (self.opt_biascor & ~OPT_BIASCOR_MAD) != SUPPORTED_OPT_BIASCOR:
            errs.append(f"opt_biascor={self.opt_biascor}: only {SUPPORTED_OPT_BIASCOR} (5D bias on MU per sample with "
                        "MUCOVSCALE + MUCOVADD) is implemented")
        if self.interp_biascor_logmass != 0:
            errs.append("interp_biascor_logmass must be 0 (SALT2mu's default is 1; interpolation in logmass is not implemented)")
        if self.sigma_cell_biascor < 10:
            errs.append("sigma_cell_biascor < 10 (distance-weighted cells) is not implemented")
        if self.nzbin <= 0 or self.powzbin <= 0 or self.powzbin == 1.0:
            errs.append("nzbin > 0 and powzbin > 0 (powzbin != 1) are required; linear fit z bins are not implemented")
        if self.min_per_zbin != 1:
            errs.append("min_per_zbin != 1 is not implemented")
        for i in (3, 4, 6, 7, 8, 9, 10, 11, 12):
            if self.u[i] != 0:
                errs.append(f"u{i}={self.u[i]}: only alpha (u1), beta (u2) and gamma (u5) may float")
        for i in (3, 4, 6, 10, 12):
            if self.p[i] != 0:
                errs.append(f"p{i}={self.p[i]}: must be 0 (z-dependent alpha/beta/gamma, Omega_k and wa are not implemented)")
        if self.u[1] not in (0, 1) or self.u[2] not in (0, 1, 3) or self.u[5] not in (0, 1):
            errs.append(f"u1={self.u[1]}, u2={self.u[2]}, u5={self.u[5]}: supported values are u1 0/1, u2 0/1/3, u5 0/1")
        if self.p[8] <= 0:
            errs.append("p8 (tau of the host step) must be > 0")
        if not self.surveygroups:
            errs.append("surveygroup_biascor is required (with zbin=... for every group)")
        for g in self.surveygroups:
            if g.zbin is None:
                errs.append(f"survey group '{g.name}' has no zbin option; SALT2mu's default z binning is not implemented")
        if self.blindflag not in (None, 0):
            errs.append("blindflag != 0: blinding is not implemented")
        if self.blindflag is None:
            warnings.warn("blindflag not set in the input; bbclocal never blinds")
        for k, v in self.other.items():
            msg = _refusal(k, v)
            if msg is None:
                continue
            if msg:
                errs.append(msg)
            elif not allow_unknown:
                errs.append(f"unknown or unsupported key '{k}={v}' (pass allow_unknown=True to ignore it at your own risk)")
        if errs:
            raise UnsupportedOptionError("unsupported SALT2mu configuration:\n  - " + "\n  - ".join(errs))
        return self


_REFUSE = {
    "varname_pia": "CC prior / BEAMS (varname_pIa) is not implemented",
    "simfile_ccprior": "CC prior (simfile_ccprior) is not implemented",
    "prescale_biascor": "prescale_biascor is not implemented",
    "cid_select_file": "cid_select_file is not implemented",
    "opt_chi2max": "opt_chi2max other than 1 is not implemented",
    "nsplitran": "NSPLITRAN is not implemented",
    "restore_sigz": "restore_sigz != 0 is not implemented",
    "zpecerr": "zpecerr != 0 is not implemented",
    "uave": "uave != 1 is not implemented",
    "um0": "uM0 != 1 (one free M0 per z bin) is not implemented",
    "mag0": "mag0 other than -30 is not implemented",
    "debug_flag": "debug_flag is not supported",
    "fieldlist_biascor": "fieldlist_biascor is not implemented (use fieldGroup_biascor)",
    "sigint_fix": "sigint_fix is not implemented",
    "iflag_duplicate": "iflag_duplicate != 0 is not implemented",
    "lcfit_model": "only SALT2/SALT3 light-curve fits are supported (no BayeSN)",
}
_ALLOWED_VALUES = {"opt_chi2max": {"1"}, "iflag_duplicate": {"0"}, "restore_sigz": {"0"}, "zpecerr": {"0", "0.", "0.0"}, "uave": {"1"},
                   "um0": {"1"}, "mag0": {"-30", "-30.0", "-30.00"}, "debug_flag": {"0"},
                   "simfile_ccprior": {"none", "null", ""}, "varname_pia": {"none", "null", ""}, "sigint_fix": {""}}


def _refusal(key, value):
    k = key.lower()
    if k in _ALLOWED_VALUES and str(value).strip().lower() in _ALLOWED_VALUES[k]:
        return None
    if k in _REFUSE:
        return f"{key}={value}: {_REFUSE[k]}"
    if "zphot" in k or "photoz" in k:
        return f"{key}={value}: photo-z options are not implemented"
    if "bayesn" in k:
        return f"{key}={value}: BayeSN is not implemented"
    if "ccprior" in k or k.startswith("pia") or "beams" in k:
        return f"{key}={value}: CC prior / BEAMS options are not implemented"
    return ""      # unknown


# ---------------------------------------------------------------- parsing
def _tokenize(text):
    lex = shlex.shlex(text, posix=True)
    lex.whitespace_split = True
    lex.commenters = "#"
    return list(lex)


def _parse_groups(value, cls):
    out = []
    for item in [s for s in value.split(",") if s.strip()]:
        item = item.strip()
        m = re.fullmatch(r"([^()]+)(?:\((.*)\))?", item)
        if not m:
            raise UnsupportedOptionError(f"cannot parse group '{item}'")
        name, opts = m.group(1).strip(), m.group(2)
        if "|" in name:
            raise UnsupportedOptionError(f"exact-match field groups ('|') are not implemented: '{item}'")
        zbin = None
        if opts:
            for o in opts.split(":"):
                k, _, v = o.partition("=")
                if k.strip() == "zbin":
                    zbin = float(v)
                else:
                    raise UnsupportedOptionError(f"group option '{o}' in '{item}' is not implemented (only zbin=)")
        if cls is SurveyGroup:
            out.append(SurveyGroup(name=name, members=[s.strip() for s in name.split("+")], zbin=zbin))
        else:
            out.append(FieldGroup(name=name, zbin=zbin))
    return out


_FLOAT_KEYS = {"zmin", "zmax", "powzbin", "lensing_zpar", "sig1", "redchi2_tol", "x1min", "x1max", "cmin", "cmax",
               "logmass_min", "logmass_max", "sigma_cell_biascor", "h0"}
_INT_KEYS = {"nzbin", "nbin_logmass", "interp_biascor_logmass", "min_per_cell_biascor", "nbinc_mucovscale", "blindflag",
             "min_per_zbin"}


def _apply_tokens(cfg, tokens, source="input"):
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.upper().startswith("CUTWIN"):
            opt = re.fullmatch(r"CUTWIN(?:\((.*)\))?", tok, flags=re.I)
            if not opt:
                raise UnsupportedOptionError(f"cannot parse '{tok}'")
            o = (opt.group(1) or "").upper()
            if i + 1 < len(tokens) and tokens[i + 1].upper() == "NONE":
                raise UnsupportedOptionError("CUTWIN NONE is not implemented")
            if o not in ("", "DATAONLY"):
                raise UnsupportedOptionError(f"CUTWIN({o}) is not implemented (only plain CUTWIN and CUTWIN(DATAONLY))")
            if i + 3 >= len(tokens):
                raise UnsupportedOptionError(f"incomplete CUTWIN near '{' '.join(tokens[i:])}'")
            var, lo, hi = tokens[i + 1], float(tokens[i + 2]), float(tokens[i + 3])
            if any(c.var == var for c in cfg.cutwins):
                raise UnsupportedOptionError(f"two CUTWINs on '{var}': SALT2mu's handling of repeated windows is not "
                                             "reproduced; give one window per variable")
            cfg.cutwins.append(CutWin(var, lo, hi, dataonly=(o == "DATAONLY")))
            i += 4
            continue
        if "=" not in tok:
            raise UnsupportedOptionError(f"cannot parse token '{tok}' in {source}")
        key, _, val = tok.partition("=")
        if val == "" and i + 1 < len(tokens) and "=" not in tokens[i + 1]:
            val = tokens[i + 1]
            i += 1
        i += 1
        _set_key(cfg, key.strip(), val.strip())


def _set_key(cfg, key, val):
    k = key.lower()
    m = re.fullmatch(r"([pu])(\d+)", k)
    if m:
        n = int(m.group(2))
        if 1 <= n <= 12:
            if m.group(1) == "p":
                cfg.p[n] = float(val)
            else:
                cfg.u[n] = int(float(val))
            return
    if k == "opt_biascor":
        cfg.opt_biascor = int(val)
    elif k == "simfile_biascor":
        cfg.simfile_biascor = [os.path.expandvars(s) for s in val.split(",") if s]
    elif k == "surveygroup_biascor":
        cfg.surveygroups = _parse_groups(val, SurveyGroup)
    elif k == "fieldgroup_biascor":
        cfg.fieldgroups = _parse_groups(val, FieldGroup)
    elif k == "varname_gamma":
        cfg.varname_gamma = val
    elif k == "chi2max":
        cfg.chi2max = float(val)
    elif k.startswith("chi2max("):
        raise UnsupportedOptionError(f"{key}: per-survey chi2max is not implemented")
    elif k == "cid_reject_file":
        cfg.cid_reject_file = None if val.lower() in ("none", "null", "") else os.path.expandvars(val)
    elif k in _FLOAT_KEYS:
        setattr(cfg, k, float(val))
    elif k in _INT_KEYS:
        setattr(cfg, k, int(float(val)))
    elif k in _IGNORED_KEYS:
        if k == "min_per_zbin":
            cfg.min_per_zbin = int(float(val))
    else:
        cfg.other[key] = val


def parse_salt2mu_text(text, muopt=None):
    """Parse the text of a SALT2mu/BBC input file. If the file has a Pippin YAML header ending with '#END_YAML', only the
    lines after it are read. muopt: optional string of extra tokens applied last (like a MUOPT)."""
    lines = text.splitlines()
    for j, line in enumerate(lines):
        if line.strip().startswith("#END_YAML"):
            lines = lines[j + 1:]
            break
    cfg = BBCConfig()
    for line in lines:
        toks = _tokenize(line)
        if toks:
            _apply_tokens(cfg, toks)
    if muopt:
        _apply_tokens(cfg, _tokenize(muopt), source="muopt")
    return cfg


def read_salt2mu_input(path, muopt=None):
    with open(path) as f:
        return parse_salt2mu_text(f.read(), muopt=muopt)
