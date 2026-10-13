// Weather-driven variant (user request 2026-09-17): the four rates come from the thermal response curves evaluated on the
// station's observed daily temperature (28-day trailing mean, exactly as in MLODE's ODE), and the rainfall index is the observed
// 7-day cumulative precipitation at the trap scaled by its training-period mean. Tables are passed on the daily grid
// (t = 0 on 1 Aug 2016) and interpolated linearly. Parameters, priors and likelihood as in jam_cum14.stan.
functions {
  int idx_of(real t, int n) {
    // largest i in [1, n-1] with (i - 1) <= t, by bisection (to_int is not available on autodiff reals)
    int lo = 1; int hi = n - 1;
    while (hi > lo) { int mid = (lo + hi + 1) / 2; if (t >= mid - 1) lo = mid; else hi = mid - 1; }
    return lo;
  }
  real interp(vector tab, real t) {
    int n = num_elements(tab);
    real tt = fmin(fmax(t, 0), n - 1.000001);
    int i = idx_of(tt, n);
    real f = tt - (i - 1);
    return tab[i] + f * (tab[i + 1] - tab[i]);
  }
  vector jam(real t, vector y, real alpha, real K, vector ovi, vector pea, vector dev, vector mu, vector rain) {
    real ov = interp(ovi, t); real pe = interp(pea, t); real de = interp(dev, t); real m = interp(mu, t); real r = interp(rain, t);
    vector[2] dydt;
    dydt[1] = y[2] * (1 - y[1] / (K + alpha * K * r)) * ov - y[1] * de;
    dydt[2] = 0.5 * y[1] * pe * de - y[2] * m;
    return dydt;
  }
}
data {
  int<lower=1> n_obs;
  array[n_obs] int yy;
  real t0;
  array[n_obs] real ts;
  int<lower=2> T;
  vector[T] ovi; vector[T] pea; vector[T] dev; vector[T] mu; vector[T] rain;
}
transformed data { real K = 1000; }
parameters {
  real<lower=0> J0;
  real<lower=0> A0;
  real<lower=0> alpha;
  real<lower=0, upper=1> q;
}
transformed parameters {
  array[n_obs] vector[2] y_hat;
  vector[2] y_init = [J0 * 1000, A0 * 1000]';
  y_hat = ode_rk45_tol(jam, y_init, t0, ts, 1e-5, 1e-5, 1000000, alpha, K, ovi, pea, dev, mu, rain);
}
model {
  J0 ~ normal(0.8, 1);
  A0 ~ normal(0.6, 1);
  alpha ~ normal(1, 1);
  q ~ normal(0.47, 1);
  for (i in 1:n_obs) yy[i] ~ poisson(q * y_hat[i][2] / 100);
}
generated quantities {
  vector[n_obs] log_lik;
  for (j in 1:n_obs) log_lik[j] = poisson_lpmf(yy[j] | q * y_hat[j][2] / 100);
}
