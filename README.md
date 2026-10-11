# bbclocal

`bbclocal` is a fast local re-implementation of the BBC ("BEAMS with Bias Corrections") fit of SNANA's `SALT2mu.exe`, for
one specific and commonly used configuration. It is meant for design iteration, diagnostics, teaching, and validation sweeps
that would otherwise need hundreds of SNANA jobs. **SNANA remains the reference implementation**: numbers that go into a
publication should come from SNANA itself.

Given a data FITRES file, the bias-correction simulations, and a SALT2mu input file, `bbclocal` reproduces SALT2mu's event
selection, its 5D bias-correction maps and error-scaling maps, the per-event bias and error corrections, and the fit of
alpha, beta, the host-galaxy step gamma, one M0 per redshift bin, and the intrinsic scatter sigint. It writes a FITRES-like
table of distances and a YAML summary of the fit. A small helper module builds the distance covariance from systematic
variants and runs a wfit-like w fit.

## Why it is fast

A SALT2mu job spends most of its time reading the bias-correction simulations (millions of rows of gzipped text, parsed
twice) and deriving per-row quantities; the fit itself is a few percent of the run time. `bbclocal`:

1. reads and derives the simulations once per process (or once ever, with the `bbclocal-extract` npz files) and reuses them
   across fits (`bbclocal.cache.get_sims`);
2. builds the bias maps, the MUCOVSCALE/MUCOVADD maps and the interpolation with vectorized numpy operations;
3. uses the fact that, in the supported configuration, the bias correction does not depend on the fitted alpha and beta (the
   simulated alpha/beta grid has a single cell), so the bias is fixed during the fit and the M0 of each redshift bin can be
   profiled analytically; only alpha, beta and gamma are minimized numerically;
4. computes the sample assignment of each simulated event and the cuts that do not depend on the population (redshift, x1,
   c and host windows, error cuts) once per sample definition and caches them with the simulations, so a fit with other
   `CUTWIN`s applies only its own windows to the pre-selected rows;
5. in the per-cell MUCOVADD scans (`sigint_muresid_list` with the MAD), finds the step at which 1.48 MAD(pull) first
   exceeds 1 by bisection over the same sequence of trial values. Each |pull| cannot increase with sigint, so neither can
   their median, and the result is identical to that of the step-by-step scan. The STD(pull) scan has no such guarantee
   and is done step by step;
6. in the covariance helper, treats the systematic part of the covariance as low rank and solves with the Woodbury identity.

Typical cost on a laptop: about 9 s to read and derive ~4 million simulated rows from npz files, then 3-8 s per BBC fit
(preparation of the maps plus the fit), less for a fit restricted to a sub-population (about 0.5 s for one of eight color
bins with ~0.5 million simulated rows), and about 1 GB of memory per process.

## Supported configuration

`bbclocal` reads the plain `key=value` lines and `CUTWIN var lo hi` lines of a SALT2mu input file (a Pippin YAML header
ending in `#END_YAML` is skipped). Extra tokens in the same syntax can be applied on top, like a Pippin `MUOPT`.

Supported:

- `opt_biascor=4336`: 5D map of the bias on MU in cells of z x host variable x x1 x c, one map per bias-correction sample
  (IDSAMPLE), with the MUCOVSCALE and MUCOVADD error maps (bit 2048, MAD, is implied, as in SALT2mu).
- Bias-correction simulations whose map has a single (alpha, beta, gammaDM) cell: either SALT2mu's dust flag is set (any of
  the first 20 simulated rows has `SIM_AV > 0`, so `SIM_alpha`, `SIM_beta` are overwritten by `p1`, `p2`), or the simulations
  have a single `SIM_alpha` and `SIM_beta` equal to `p1`, `p2`. `SIM_gammaDM` must be 0.
- Per-event population weights of the bias-correction simulations (`SIM_WGT_POP`, finite and >= 0), read from the files or
  set after loading with `BiasCorSims.with_weights` (see "Population weights" below).
- `surveygroup_biascor` with `zbin=` for every group (survey names need a `SURVEY.DEF` file or a name-to-IDSURVEY map, or
  can be written as IDSURVEY integers) and `fieldGroup_biascor` (substring matching on `+`-separated fields, first group wins,
  as in SALT2mu).
- Fit redshift bins: `zmin`, `zmax`, `nzbin`, `powzbin` (> 0), `min_per_zbin=1`; one free M0 per bin (`uM0=1`).
- `u1` (alpha), `u2` (beta; 0, 1 or 3), `u5` (gamma) floated or fixed at `p1`, `p2`, `p5`; host step with `p7` (center),
  `p8` (width) on `varname_gamma` (e.g. `HOST_LOGMASS` or `HOST_COLOR`); `logmass_min/max`, `nbin_logmass`.
- `sig1` (starting sigint), `redchi2_tol`, `lensing_zpar`, `x1min/x1max/cmin/cmax`, `CUTWIN` and `CUTWIN(DATAONLY)`,
  `nbinc_mucovscale`, `min_per_cell_biascor`, `sigma_cell_biascor >= 10`, `interp_biascor_logmass=0`.
- `chi2max=X` (global; the M0 re-centering and one-time outlier rejection at the start values; `X > 1e8` disables it).
- `cid_reject_file` (applied to data only).
- Reference cosmology flat with constant w: `p9` (Omega_Lambda), `p11` (w), `h0`.
- A local extension: `BBC.fit(fix_ab=(alpha, beta))` fixes alpha and beta in the fit only, while the bias maps keep `p1`, `p2`.

Details that are reproduced because they change results: float32 storage of every input column and of the derived mB
(single-precision `log10f`), fit covariance, MUCOVSCALE and MUCOVADD; bin edges built by cumulative addition; an event on a
bin edge goes to the lower bin; the redshift range of each sample's bias map is set by the data (all rows, before cuts); the
interpolation rules of `get_fitParBias` (neighbor choice relative to cell averages, effective bin sizes, weight clipping on
the product only, minimum counts, the edge-bin test); the covariance eigenvalue fix (`EIGMIN_COV = 1e-4`, applied only to
non-positive-definite matrices); the sigint for the map weights from the simulated events with `SNRMAX1 >= 60`; the
`sigint_muresid_list` scan; the log term `log(sigma^2 / sigma^2_last)` in the chi2 with its reference from the previous
iteration; the secant update of sigint.

## Unsupported options (refused with `UnsupportedOptionError`)

`bbclocal` raises an error rather than silently differ from SALT2mu. Refused, among others:

- any `opt_biascor` other than 4336 (plus 2048);
- bias-correction simulations that need alpha/beta-dependent maps (several `SIM_alpha` or `SIM_beta` values without the
  dust flag), `SIM_gammaDM != 0`, negative or non-finite `SIM_WGT_POP`;
- CC prior / BEAMS (`varname_pIa`, `simfile_ccprior`), photo-z options, BayeSN, `prescale_biascor`;
- `interp_biascor_logmass=1` (SALT2mu's default, so it must be set to 0 explicitly), `sigma_cell_biascor < 10`;
- survey groups without `zbin`, per-group `x1bin`/`cbin`, exact-match field groups (`|`), `fieldlist_biascor`;
- `chi2max(SURVEYLIST)=...`, `opt_chi2max` other than 1, `CUTWIN` options other than `DATAONLY`, repeated `CUTWIN`s on one
  variable, `CUTWIN NONE`;
- `uM0 != 1`, `uave != 1`, floated `p3`, `p4`, `p6`, `p7`, `p8` or cosmology parameters, non-zero `p3`, `p4`, `p6`, `p10`, `p12`,
  `blindflag != 0`, `restore_sigz`, `zpecerr`, `iflag_duplicate != 0`, `sigint_fix`, `cid_select_file`;
- an event whose FIELD matches no field group placed after an event that matches one (SALT2mu would then reuse the previous
  event's field group);
- any key it does not recognize (override with `allow_unknown=True` / `--allow-unknown` at your own risk).

The covariance helper reproduces unbinned `create_covariance` output of the form `diag(MUERR^2) + sum_S W_S^2 dmu_S dmu_S^T`
(plus optional free offsets per group); external full matrices (`EXTRA_COVS`) are not supported.

## Validation against SNANA

Validated against SNANA v12_02d `SALT2mu.exe` on DES-Dovekie-like validation samples (simulated LOWZ + Foundation + DES
data, DES field groups, `opt_biascor=4336`, 20 `powz` redshift bins, host-mass step, seven dust/scatter models for the
bias-correction simulations):

- the same SNe are selected in every fit;
- alpha, beta, gamma and sigint agree to SNANA's printed precision (5 decimals, 4 for sigint) in 13 fits covering global,
  color-split and color x mass-split populations and the 7 models (in two fits beta differs by 2e-5);
- per-SN distances: MU differs by 0.003 mmag rms, at most 0.016 mmag, over ~30,000 distances (SNANA prints MU to 0.01 mmag);
- distance covariance + w fit (`bbclocal.covfit` vs `create_covariance` + `wfit` on SNANA's own distances): sigma_w within
  0.1%, w within 1e-4;
- speed: about 70 times faster per BBC fit than a SALT2mu job, and about 300 times faster for covariance + w fit.

These comparisons were made with the code from which this package was generalized. The packaged code was checked again on
two of the SNANA fits: the global fit with the nominal simulations (`chi2max=1e7`) and the base fit with `chi2max=16`, in
which SALT2mu rejects 9 outliers. In both, the selected SNe (1455 and 1446), alpha, beta, gamma, sigint and the average M0
are identical to SNANA's printed values, and MU differs by 0.003 mmag rms (at most 0.006 mmag).

The parameter uncertainties reported by `bbclocal` (finite-difference Hessian) agree with SALT2mu's MIGRAD uncertainties to
1-5%; they are not used anywhere in the fit.

`examples/validate_against_snana.py` repeats this comparison for any SALT2mu output you have (paths are arguments; no data
are included in this repository).

### Known differences

- **Minimizer.** SALT2mu uses MINUIT (MIGRAD, stopping at EDM < 1e-4, so its minimum is defined to about 1e-2 sigma);
  `bbclocal` profiles the M0 exactly and minimizes alpha, beta, gamma to a much tighter tolerance. Parameter differences up to
  ~1e-2 sigma are therefore possible, although none was seen at printed precision in the validation.
- **Distance integral.** SALT2mu uses Romberg integration at relative tolerance 1e-6; `bbclocal` uses an accurate tabulated
  integral. Differences are ~1e-6 mag.
- `IDSAMPLE` numbers follow the order of first appearance in the data, not SALT2mu's sorted order; sample names follow
  SALT2mu's convention (e.g. `DES(X3)`).
- The output column `CUTFLAG` is `bbclocal`'s own bit mask (1 selection, 2 cid_reject, 4 no valid bias correction,
  8 chi2max), not SALT2mu's `CUTMASK`.

## Population weights

SALT2mu reads an optional column `SIM_WGT_POP` from the bias-correction files and uses it as a per-event weight when it
builds the bias-correction maps. A typical use is to reweight one set of simulations to a different population (for
example, a different color or stretch distribution) instead of generating new simulations. `bbclocal` reproduces how
SALT2mu uses these weights (checked line by line against `SALT2mu.c` and `sntools.c`):

- **Bias map.** The weight of a simulated event in the averages of each 5D cell (the cell's mean z, host, x1 and c, and
  the bias value) is `SIM_WGT_POP / sigma_mu^2` instead of `1 / sigma_mu^2`.
- **Counts are not weighted.** SALT2mu counts every simulated event that passes the cuts toward the number of events in a
  cell, whatever its weight. All count thresholds (`min_per_cell_biascor`, the minimum summed count of 10 over the
  interpolation cells, the 5 events needed for an error-map cell, the 50 events with `SNRMAX1 >= 60` for the map sigint)
  therefore use counts.
- **Error maps.** MUCOVSCALE is (1.48 x weighted median of |pull|)^2 with the population weights; the cell averages used to
  interpolate the error maps are weighted by `SIM_WGT_POP`. MUCOVADD, however, comes from an **unweighted** scan: SALT2mu
  calls `sigint_muresid_list` there without weights.
- **Weighted median** (`arrayStat_MEDIAN`): sort the values, add their weights in sorted order until the running sum
  reaches half the total, and take that element (odd number of events) or the mean of that element and the next one in
  sorted order (even number). The parity is that of the count, not of the weights. When tied values with different weights
  meet at the crossing, the result depends on how the tied elements are ordered; `bbclocal` then reproduces the GSL
  heapsort that SALT2mu uses.
- **Weighted STD** (`arrayStat_STD`, used for the sigint of the map weights from the events with `SNRMAX1 >= 60`): the
  mean pull and the mean variance are weighted means, but the STD is `sqrt(sum(w x^2)/N - (sum(w x)/N)^2)` with N the
  **count**, not the sum of the weights. This is a quirk of SALT2mu and is reproduced as is.
- The RMS and statistical error of the bias in each cell (`SampleMaps.bias_rms_err()`, diagnostics only, because SALT2mu
  sets the bias error to 0 before the fit) use sums weighted by `SIM_WGT_POP` alone.
- The weights are stored as float32, as SALT2mu reads them.

Consequences worth knowing:

- A weight of 2 on an event is equivalent to listing the event twice for the weighted averages (bias, cell averages), but
  not for the counts, the weighted STD (whose denominator is the count) or MUCOVADD (unweighted).
- A weight of 0 is **not** the same as removing the event: the event still counts toward every threshold and still enters
  the MUCOVADD scan. If every event of a cell that passes a count threshold has weight 0, SALT2mu divides 0 by 0 (bias map)
  or aborts (weighted median); `bbclocal` raises an error in both cases.
- Cases in which SALT2mu's behavior is undefined raise an error: the weighted median when the crossing falls on the last
  element with an even count (SALT2mu reads past the end of its index array).
- The weighted mean redshift per fit bin (`calc_zM0_biasCor`, used only for the redshift column of SALT2mu's M0DIF output)
  is not computed by `bbclocal`.

From Python, weights can be set after loading without copying the simulations:

```python
sims = BiasCorSims(biascor_files, cfg)
w = my_population_weights(sims.S)          # one value per simulated row, in file order
view = sims.with_weights(w)                # shares all large arrays with sims; SIM_WGT_POP replaced by w
B = BBC(cfg, data, view).prepare()
```

The view has its own cache of the population-independent row selection, so views with different weights do not interfere
with each other or with the original object.

`with_weights(w, min_weight=x)` is an option that SALT2mu does not have: rows with `w <= x` are removed from the
simulations before any map is made, so they no longer count toward the thresholds or enter MUCOVADD. The result equals
SALT2mu run on bias-correction files from which those rows were deleted, and in general differs from SALT2mu run with
`SIM_WGT_POP = w`. Use it, for example, with `min_weight=0` when the weights select a sub-population and the remaining
rows should play no role at all. The default (`min_weight=None`) is faithful to SALT2mu.

With all weights equal to 1, the weighted code path gives results bit-identical to the unweighted one (checked on real data
for the global fit and two color windows on two data samples).

## Installation

```
git clone https://github.com/SterlingYM/BBC-local.git
cd BBC-local
pip install -e .            # dependencies: numpy, scipy, pandas, pyyaml
pip install pytest && pytest -q tests     # ~30 s, synthetic data only
```

## Quick start

1. Reduce the bias-correction simulations once (optional but recommended; keeps only the needed columns, records what was
   cut, and loads in seconds):

   ```
   bbclocal-extract LOWZ_BIASCOR.FITRES.gz FOUND_BIASCOR.FITRES.gz DES_BIASCOR.FITRES.gz \
       -o biascor_npz --input SALT2mu.input --extra-cols c_true
   ```

   With `--input`, the cuts of that input that apply to the simulations (CUTWINs, x1/c windows, zHD >= zmin) are applied
   and recorded; `bbclocal` later refuses a configuration that does not imply them. Without `--input` only the cuts SALT2mu
   always applies are made.

2. Run a fit:

   ```
   bbclocal fit --input SALT2mu.input --data DATA.FITRES.gz \
       --biascor biascor_npz/LOWZ_BIASCOR.npz biascor_npz/FOUND_BIASCOR.npz biascor_npz/DES_BIASCOR.npz \
       --survey-def $SNDATA_ROOT/SURVEY.DEF --muopt "CUTWIN c -0.3 0.0 chi2max=10000000" --out fit_blue
   ```

   This writes `fit_blue.YAML` (alpha0, beta0, gamma0, sigint, NSNFIT, chi2red, M0 per redshift bin, samples, ...) and
   `fit_blue.FITRES.gz` (CID, IDSURVEY, FIELD, zHD, MU, MUERR, MUMODEL, biasCor_mu, biasCor_muCOVADD, IDSAMPLE, IZBIN,
   CUTFLAG, ...). The bias-correction files can also be the original FITRES files.

3. From Python, for many fits against the same simulations:

   ```python
   from bbclocal import BBC, read_fitres, read_salt2mu_input
   from bbclocal.cache import get_sims

   base = read_salt2mu_input("SALT2mu.input")
   data = read_fitres("DATA.FITRES.gz")
   for lo, hi in [(-0.3, 0.0), (0.0, 0.3)]:
       cfg = base.apply_muopt(f"CUTWIN c {lo} {hi} chi2max=10000000").validate()
       sims = get_sims(biascor_files, cfg)              # loaded once, reused
       B = BBC(cfg, data, sims, survey_ids={"DES": 10, "FOUNDATION": 150}).prepare()
       print(B.fit()["beta0"])
   ```

4. Compare with SNANA:

   ```
   python examples/validate_against_snana.py --input SALT2mu.input --data DATA.FITRES.gz \
       --biascor biascor_npz/*.npz --snana-fitres OUTPUT_BBCFIT/FITOPT000_MUOPT001.FITRES.gz \
       --muopt "chi2max=10000000" --cid-reject exclude.txt --survey-def SURVEY.DEF
   ```

## Package layout

| module | content |
|---|---|
| `config.py` | `BBCConfig`, the SALT2mu input parser, MUOPT overrides, refusal of unsupported options |
| `io.py` | FITRES reader, bias-correction readers (FITRES or npz), `SURVEY.DEF`, CID lists |
| `extract.py` | `bbclocal-extract` |
| `samples.py` | survey and field groups -> bias-correction samples |
| `binning.py` | SALT2mu bin edges and bin lookup, `powz` redshift bins |
| `model.py` | per-event mB, covariance (with the eigenvalue fix), distance errors, host step |
| `maps.py` | 5D bias map, its interpolation, MUCOVSCALE/MUCOVADD maps, sigint scan, weighted median/STD of SALT2mu |
| `core.py` | `BiasCorSims` (simulations + derived quantities) and `BBC` (cuts, maps, fit, outputs) |
| `covfit.py` | low-rank distance covariance (Woodbury) and the wfit-like grid fit |
| `cache.py` | per-process cache of prepared simulations |
| `cli.py`, `output.py` | `bbclocal fit`, FITRES/YAML writers |

## References

If you use `bbclocal`, please cite the methods and software it re-implements:

- SNANA: Kessler et al. 2009, PASP, 121, 1028
- BBC: Kessler & Scolnic 2017, ApJ, 836, 56
- SALT2mu: Marriner et al. 2011, ApJ, 740, 72

## License

No license has been chosen yet.
