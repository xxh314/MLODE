# MLODE — code for the reported configurations

Code behind the *J. R. Soc. Interface* revision of "Toward Scalable and Long-Term Mosquito Abundance Forecasts:
A Machine Learning–Mechanistic Modeling Approach with Environmental Drivers". Two species, *Aedes aegypti* and
*Culex quinquefasciatus*, surveillance traps in Miami-Dade County, 2021 forecast origins.

This folder is deliberately narrow: the **best-performing configuration for each species under both the two-stage
and the three-stage ODE**, plus the three comparison models. Exploratory variants, sensitivity sweeps and
alternative inverse-step estimators are not included.

## Not included

* **The data.** Surveillance counts, weather records and site covariates are read from a separate `ODE-2025`
  data repository: `mlode/mlode_pipeline.py` imports it through `util.load_input_feature()`, and `mlode/inverse_step_2stage.py`
  reads the per-trap label files from `../ODE-2025/label/`. Place that repository as a sibling of `mlode/`, or
  edit the two paths marked `ODE-2025` in those files.
* **MCMC sampler output.** The posterior draws (~9 GB) are not here. `baselines/mcmc/` contains the model and the
  commands that produce them.
* `baselines/mcmc/manifest.csv` is kept because the fitting scripts enumerate runs from it; it lists trap identifiers
  and station codes only, no counts. Remove it if trap identifiers should not be shared.

## Layout

File names say what each file does; the two inverse-step modules differ only in how many
compartments the ODE has.

```
mlode/                           the forecasting pipeline and the non-mechanistic baseline
  inverse_step_2stage.py           the inverse step for the two-stage ODE (juveniles, adults). Also the shared
                                   data layer every other module uses: per-trap collections, station temperature,
                                   the thermal rate functions, the count smoothers, and the A0 admissibility rule
  inverse_step_3stage.py           the inverse step for the three-stage ODE (larvae, pupae, adults). Imports the
                                   data layer above; the extra compartment needs a third derivative of the
                                   smoothed counts, so a quintic spline replaces the pchip interpolant
  mlode_pipeline.py                the pipeline end to end: recover k(t), build features, fit the learner, run the
                                   forward ODE. Also holds the per-trap climatology baseline and the DESIGNS table
                                   of A0 and smoothing-window settings. Every option is a command-line flag
  scoring_metrics.py               per-trap skill, paired median differences with a trap bootstrap, Wilcoxon,
                                   Peirce alert score
  baseline_direct_gbr.py           baseline: gradient boosting straight from covariates to counts, no ODE
baselines/mcmc/                  baseline: per-trap Bayesian calibration of the same two-stage ODE
  mcmc_2stage_ode.stan             the reported variant: observed temperature and observed scaled 7-day rainfall,
                                   four parameters per trap (J0, A0, alpha, q)
  mcmc_prepare_counts.py           step 1 - one Stan data file per trap x origin, from manifest.csv
  mcmc_prepare_weather.py          step 2 - adds the daily thermal-rate and rainfall tables the observed-weather
                                   model needs; --clim gives the climatological-weather scenario
  mcmc_fit_trap.py                 step 3 - CmdStan sampling for one trap
  mcmc_forecast.py                 step 4 - forward simulation from the posterior, 500 draws
  run_mcmc.sh                      the four steps in order
run_best.sh                      every reported configuration, in order
requirements.txt
```

## The reported configurations

All four use recipe v3 — the learner's target is the **station-pooled anomaly of the trap-wise scaled k(t)** —
with an A0 multiplier of 2, a 14-day persistence decay into the learner's k, and CatBoost. *Ae. aegypti* uses
tuned CatBoost hyper-parameters, *Cx. quinquefasciatus* the untuned defaults.

| Model | Species | Origin | Design | Smoothing window | Origin treatment |
|---|---|---|---|---|---|
| two-stage (reported) | *Ae. aegypti* | 2021-06-01 | `W112` | 112 d | `D1` |
| two-stage (reported) | *Cx. quinquefasciatus* | 2021-01-01 | `W112` | 112 d | `D0` |
| three-stage | *Ae. aegypti* | 2021-06-01 | `W84` | 84 d | `D1` |
| three-stage | *Cx. quinquefasciatus* | 2021-01-01 | `W112` | 112 d | `D0` |

`D1` drops the last 14 days of training targets, where the centred smoother becomes one-sided; `D0` keeps them.
Each run writes both, so select the row shown above.

```bash
./run_best.sh                      # all four, then the direct-gradient-boosting baseline
cd baselines/mcmc && ./run_mcmc.sh aedes 2021-06-01
```

## Two things that are easy to get wrong

**1. There is no post-processing. The `fc` column is the forecast.** `mlode_pipeline.py` writes the raw
forward-ODE trajectory and that is what is reported and scored, against comparators that are equally raw. An
earlier version of this work blended the forecast toward climatology with lead time, and for
*Cx. quinquefasciatus* with a damped-persistence member as well. Those steps are **not** part of the reported
model and are not included here: they were applied to MLODE but not to the comparators, which made the comparison
unfair, and they masked how the forward integration itself behaves. Every number in the paper is raw against raw.

**2. The climatology baseline needs no separate run.** The `clim` column in every `mlode_obs_*.csv` is each trap's
own weekly climatology - a 5-week circularly smoothed day-of-year mean over the training years, built by
`mlode_pipeline.climatology`. It is both the third comparison model and the denominator of the skill score
`1 - RMSE/RMSE_climatology` used throughout the paper.

**A note on the smoothing window.** Both species use 112 days. For *Cx. quinquefasciatus* this replaces the
28-day window of the submitted version: at 28 days the raw forward ODE collapses, with A0 reaching 22 times the
mean count and 65% of predictions beyond three months falling to exactly zero, which the earlier post-processing
concealed. At 112 days A0 is 3.7 times the mean count and the extinction is gone. For *Ae. aegypti* the raw
sensitivity grid puts the optimum at 70-84 days with 112 days close behind, inside a plateau; 112 days is retained
for continuity with the submitted configuration.

## Scoring

```bash
python mlode/scoring_metrics.py A.csv fc B.csv fc        # paired comparison of two forecast files
```
`scoring_metrics.per_trap` gives the per-trap window RMSE and skill; `compare` gives the median paired difference
with a 2000-resample trap bootstrap, the Wilcoxon signed-rank p, the share of traps won, and the Peirce alert
score. Horizons are cumulative windows from the origin: 30 / 90 / 153 days for *Ae. aegypti* and 45 / 90 / 180
for *Cx. quinquefasciatus*, the first widened because weekly winter trapping leaves too few January collections
to score a 30-day window.

## Environment

Python 3.10.13 with `requirements.txt`; CmdStan 2.40 for the MCMC baseline. `CB_THREADS` caps CatBoost threads
and `REC_PROCS` the parallel inverse-step recovery, both useful when running several configurations at once.
