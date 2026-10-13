#!/bin/bash
# The mechanistic MCMC baseline, one trap at a time. Sampler output is NOT included in this release; these are the
# commands that produce it. mcmc_2stage_ode.stan is the reported variant: the two-stage ODE driven by observed temperature
# and observed scaled 7-day rainfall, four parameters per trap (J0, A0, alpha, q).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
SP=${1:-aedes}; ORIGIN=${2:-2021-06-01}

$PY mcmc_prepare_counts.py                      # one Stan data file per trap x origin, from manifest.csv
$PY mcmc_prepare_weather.py "$SP" "$ORIGIN"  # adds the daily thermal-rate and rainfall tables the observed-weather model needs
# climatological-weather scenario: same posterior, forecast window driven by day-of-year means
$PY mcmc_prepare_weather.py "$SP" "$ORIGIN" --clim

while IFS=, read -r NAME SPECIES TRAP STATION ORIG REST; do
  [ "$SPECIES" = "$SP" ] && [ "$ORIG" = "$ORIGIN" ] || continue
  STAN_FILE=mcmc_2stage_ode.stan $PY mcmc_fit_trap.py "data/${NAME}_obs.json" "fits/${NAME}_obs" 4 2000 2000 2
  $PY mcmc_forecast.py "$NAME" 500     # forward simulation -> forecasts/${NAME}_obs.csv
done < <(tail -n +2 manifest.csv)
