"""Command-line interface: `bbclocal fit ...`."""
import argparse
import sys
import time

from .config import read_salt2mu_input
from .core import BBC, BiasCorSims
from .io import read_cid_list, read_fitres, read_survey_def
from .output import write_fitres, write_yaml


def _survey_ids(args):
    ids = {}
    if args.survey_def:
        ids.update(read_survey_def(args.survey_def))
    for item in args.survey_id or []:
        for tok in item.split(","):
            if tok:
                name, _, val = tok.partition("=")
                ids[name.strip()] = int(val)
    return ids or None


def cmd_fit(args):
    t0 = time.time()
    cfg = read_salt2mu_input(args.input, muopt=args.muopt)
    cfg.validate(allow_unknown=args.allow_unknown)
    biascor = args.biascor or cfg.simfile_biascor
    if not biascor:
        raise SystemExit("no bias-correction files: give --biascor or simfile_biascor in the input")
    cid_reject = read_cid_list(args.cid_reject) if args.cid_reject else None
    if args.no_cid_reject:
        cid_reject = []
    data = read_fitres(args.data)
    sims = BiasCorSims(biascor, cfg, dust_flag=args.dust_flag, trust_legacy=args.trust_legacy_npz)
    B = BBC(cfg, data, sims, survey_ids=_survey_ids(args), cid_reject=cid_reject, validate=False).prepare()
    r = B.fit()
    r["cpu_seconds"] = round(time.time() - t0, 2)
    write_yaml(r, args.out + ".YAML")
    hdr = [f"bbclocal fit: alpha0={r['alpha0']:.5f} beta0={r['beta0']:.5f} gamma0={r['gamma0']:.5f} sigint={r['sigint']:.4f} "
           f"NSNFIT={r['NSNFIT']}", "CUTFLAG bits: 1 selection, 2 cid_reject, 4 no valid bias correction, 8 chi2max"]
    write_fitres(B.table(), args.out + ".FITRES.gz", header_lines=hdr)
    print(f"alpha0={r['alpha0']:.5f} beta0={r['beta0']:.5f} gamma0={r['gamma0']:.5f} sigint={r['sigint']:.4f} "
          f"NSNFIT={r['NSNFIT']} chi2red={r['chi2red']:.4f}  ({r['cpu_seconds']} s)")
    print(f"wrote {args.out}.YAML and {args.out}.FITRES.gz")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bbclocal", description="Local re-implementation of SNANA's BBC fit (SALT2mu) for one "
                                 "common configuration.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="run one BBC fit")
    f.add_argument("--input", required=True, help="SALT2mu/BBC input file (key=value lines; Pippin YAML header allowed)")
    f.add_argument("--data", required=True, help="data FITRES file (.gz ok)")
    f.add_argument("--biascor", nargs="+", help="bias-correction files (FITRES or bbclocal-extract npz); default: "
                   "simfile_biascor of the input")
    f.add_argument("--muopt", default=None, help="extra SALT2mu tokens applied on top of the input, e.g. "
                   "\"CUTWIN c -0.3 0 chi2max=1e7\"")
    f.add_argument("--out", required=True, help="output prefix (writes <prefix>.YAML and <prefix>.FITRES.gz)")
    f.add_argument("--cid-reject", help="CID reject list (overrides cid_reject_file of the input)")
    f.add_argument("--no-cid-reject", action="store_true", help="ignore cid_reject_file")
    f.add_argument("--survey-def", help="SNANA SURVEY.DEF file, to translate survey names in surveygroup_biascor")
    f.add_argument("--survey-id", action="append", help="NAME=IDSURVEY[,NAME=IDSURVEY...] (repeatable)")
    f.add_argument("--dust-flag", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                   help="override SALT2mu's dust-flag detection (needed for npz files without metadata)")
    f.add_argument("--trust-legacy-npz", action="store_true", help="accept npz files without bbclocal metadata")
    f.add_argument("--allow-unknown", action="store_true", help="ignore unknown input keys instead of refusing")
    f.set_defaults(func=cmd_fit)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
