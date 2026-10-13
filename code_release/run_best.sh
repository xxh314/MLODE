#!/bin/bash
# The reported configurations, end to end. Each MLODE command writes mlode_obs_<...>.csv next to mlode_pipeline.py.
# The `fc` column in those files IS the reported forecast: the raw forward-ODE trajectory, with no post-processing.
# The `clim` column in the same files is the climatology baseline and the denominator of the skill score.
set -e
cd "$(dirname "$0")/mlode"
PY=${PY:-python}
HP='{"depth":8,"learning_rate":0.05,"l2_leaf_reg":1,"min_data_in_leaf":50,"grow_policy":"Depthwise"}'

# ---------------------------------------------------------------- MLODE, two-stage ODE (the reported model)
# Ae. aegypti - A0 multiplier 2, 112-day centred smoothing window, recipe v3, tuned CatBoost. Report D1 rows.
$PY mlode_pipeline.py --species aedes --origin 2021-06-01 --designs W112 --ml anom --hp "$HP" \
    --k_persist 14 --anom_pool station --suffix _best2s
# Cx. quinquefasciatus - A0 multiplier 2, 112-day window, recipe v3, untuned CatBoost. Report D0 rows.
$PY mlode_pipeline.py --species culex --origin 2021-01-01 --designs W112 --ml anom \
    --k_persist 14 --anom_pool station --suffix _best2s

# ---------------------------------------------------------------- MLODE, three-stage ODE (structural robustness)
# Ae. aegypti - A0 multiplier 2, 84-day window.        Cx. quinquefasciatus - A0 multiplier 2, 112-day window.
$PY mlode_pipeline.py --species aedes --origin 2021-06-01 --designs W84 --ml anom --stages 3 --hp "$HP" \
    --k_persist 14 --anom_pool station --suffix _best3s
$PY mlode_pipeline.py --species culex --origin 2021-01-01 --designs W112 --ml anom --stages 3 \
    --k_persist 14 --anom_pool station --suffix _best3s

# ---------------------------------------------------------------- the same four under climatological weather
# the only setting in which every number is a genuine forecast: the window's weather is the day-of-year mean
# of the training years rather than what actually happened.
$PY mlode_pipeline.py --species aedes --origin 2021-06-01 --designs W112 --ml anom --hp "$HP" \
    --k_persist 14 --anom_pool station --climate_scenario clim --suffix _best2s_clim
$PY mlode_pipeline.py --species culex --origin 2021-01-01 --designs W112 --ml anom \
    --k_persist 14 --anom_pool station --climate_scenario clim --suffix _best2s_clim

# ---------------------------------------------------------------- baseline: direct gradient boosting (no ODE)
$PY baseline_direct_gbr.py --species aedes --origin 2021-06-01
$PY baseline_direct_gbr.py --species culex --origin 2021-01-01
$PY baseline_direct_gbr.py --species aedes --origin 2021-06-01 --climate_scenario clim --suffix _clim
$PY baseline_direct_gbr.py --species culex --origin 2021-01-01 --climate_scenario clim --suffix _clim

# ---------------------------------------------------------------- baseline: climatology
# no command: the `clim` column written by every mlode_pipeline.py run above is the climatology forecast.

echo "done. Score with: python scoring_metrics.py A.csv fc B.csv fc"
