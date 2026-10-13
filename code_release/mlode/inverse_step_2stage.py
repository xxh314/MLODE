"""
Phase 0 library: parameterised re-implementation of the MLODE inverse step (two-stage model).

Reproduces the original chain (MLODE_official notebooks) when called with defaults:
  nonzero counts -> linear daily interpolation -> 14-day trailing mean -> pchip -> A0 = 2*ceil(-min(x'/mu + x)) -> k(t)
and exposes every choice as a parameter: zero handling, cutoff date, smoothing window/centring, A0 rule, species.
Also provides k_T(t) (the temperature-only inverse, x == 0), the forward ODE, and admissibility statistics.
Day index everywhere = days since 2016-08-01 (DAY 0), as in the original code.
"""
import os, math, datetime
import numpy as np, pandas as pd
from scipy.interpolate import pchip
from scipy.integrate import odeint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "MLODE_official", "MLODE_official")
DAY0 = datetime.datetime(2016, 8, 1)
STATIONS = ["MIA", "FLL", "HST", "TMB", "HWO", "OPF"]

def day(d):
    """datetime or 'YYYY-MM-DD' -> day index"""
    if isinstance(d, str): d = datetime.datetime.strptime(d, "%Y-%m-%d")
    return (d - DAY0).days

def date(i):
    return DAY0 + datetime.timedelta(days=int(i))

# ---------------- thermodynamic functions (verbatim from the generation notebooks) ----------------
def _aedes():
    def ovi(T): c,T0,Tm=8.56e-3,14.58,34.61; return 0.01 if (T<T0 or T>Tm) else max(0.01,c*T*(T-T0)*(Tm-T)**0.5)
    def pEA(T): c,T0,Tm=-5.99e-3,13.56,38.29; return 0.01 if (T<T0 or T>Tm) else max(0.01,c*(T-T0)*(T-Tm))
    def dev(T): c,T0,Tm=7.86e-5,11.36,39.17; return 0.01 if (T<T0 or T>Tm) else max(0.01,c*T*(T-T0)*(Tm-T)**0.5)
    def mu(T): c,T0,Tm=-1.48e-1,9.16,37.73; return 1 if (T<T0 or T>Tm) else 1/(c*(T-T0)*(T-Tm))
    # larval and pupal development, fitted to Rueda 1990 instar durations
    # (MLODE_official/Aedes aegypti/entomological parameters_Aedes aegypti.ipynb)
    def dL(T): c,T0,Tm=1.46890266e-04,1.24883825e+01,3.74296901e+01; return 0.0 if (T<T0 or T>Tm) else max(0.0,c*T*(T-T0)*abs(T-Tm)**0.5)
    def dP(T): c,T0,Tm=4.40424243e-04,1.35748546e+01,40.0; return 0.0 if (T<T0 or T>Tm) else max(0.0,c*T*(T-T0)*abs(T-Tm)**0.5)
    return dict(ovi=ovi, pEA=pEA, dev=dev, mu=mu, dL=dL, dP=dP)
def _culex():
    def dev(T): c,T0,Tm=6.38e-5,8.6,39.66; return 0 if (T<T0 or T>Tm) else max(0,c*T*(T-T0)*abs(T-Tm)**0.5)
    def ovi(T): c,T0,Tm=0.23828148,14.689913,40; return 0 if (T<T0 or T>Tm) else max(0,c*(T-T0)*abs(T-Tm))
    def pEA(T): c,T0,Tm=-3.47067577e-03,1.00038210e+01,3.94412177e+01; return 0 if (T<T0 or T>Tm) else max(0,c*(T-T0)*(T-Tm))
    def mu(T): c,T0,Tm=-1.21163521,9.0001,33.45979268; return 1 if (T<T0 or T>Tm) else min(1,1/(c*(T-Tm)*abs(T-T0)**0.5))
    def dL(T): c,T0,Tm=9.14980475e-05,8.39127742e+00,4.02811247e+01; return 0.0 if (T<T0 or T>Tm) else max(0.0,c*T*(T-T0)*abs(T-Tm)**0.5)
    def dP(T): c,T0,Tm=2.81315473e-04,5.68289812e+00,3.85667034e+01; return 0.0 if (T<T0 or T>Tm) else max(0.0,c*T*(T-T0)*abs(T-Tm)**0.5)
    return dict(ovi=ovi, pEA=pEA, dev=dev, mu=mu, dL=dL, dP=dP)

SPECIES = {
    "aedes": dict(col="Ae_aegypti_fem", temp_window=28, thermo=_aedes(), label="model2spec1"),
    "culex": dict(col="Cx_quinquefasciatus_fem", temp_window=28, thermo=_culex(), label="model2spec3"),
}

# ---------------- data ----------------
_cache = {}
def raw_counts():
    if "raw" not in _cache:
        r = pd.read_csv(os.path.join(ROOT, "MSQTrapCount_November_2021.csv"), encoding="utf-8-sig", low_memory=False)
        r["t"] = (pd.to_datetime(r.Date_Submitted, errors="coerce") - DAY0).dt.days
        _cache["raw"] = r
    return _cache["raw"]

def temp_tables():
    if "temp" not in _cache:
        t1 = pd.read_csv(os.path.join(ROOT, "Traps_tmpf_2016_2019.csv"), low_memory=False)
        t2 = pd.read_csv(os.path.join(ROOT, "Traps_tmpf_2020_2022.csv"), low_memory=False)
        _cache["temp"] = (t1, t2)
    return _cache["temp"]

def trap_station():
    """trap -> station (first listing in the temperature file)"""
    t1, _ = temp_tables()
    return dict(zip(t1.TRAPID, t1.station))

def station_daily_temp(st):
    """daily mean temperature (F), index = day since DAY0, NaN filled with previous value (as in the notebooks)"""
    t1, t2 = temp_tables()
    r1 = t1[t1.station == st].iloc[0]; r2 = t2[t2.station == st].iloc[0]
    v = np.concatenate([r1.values[10:].astype(float), r2.values[10:].astype(float)])
    for i in range(len(v)):
        if np.isnan(v[i]): v[i] = v[i-1]
    return np.array([np.mean(v[(i-1)*8:i*8]) for i in range(1, int(len(v)/8))])

def climatological_temp(st, cutoff):
    """station daily temperature with everything from `cutoff` onward replaced by the day-of-year mean of the years before it"""
    ave = np.array(station_daily_temp(st), dtype=float); c = int(cutoff); n = len(ave)
    doy = (np.arange(n) + DAY0.timetuple().tm_yday - 1) % 365
    tab = pd.Series(ave[:c]).groupby(doy[:c]).mean().reindex(range(365)).interpolate(limit_direction="both")
    tab = pd.Series(np.convolve(np.r_[tab.values[-7:], tab.values, tab.values[:7]], np.ones(15)/15, mode="valid"), index=range(365))
    out = ave.copy()
    if c < n: out[c:] = tab.reindex(doy[c:]).values
    return list(out)

def analogue_temp(st, cutoff, shift):
    """station daily temperature with everything from `cutoff` onward replaced by the observed value `shift` days
    earlier (shift a whole number of 365-day years), i.e. the same calendar period of an earlier year."""
    ave = np.array(station_daily_temp(st), dtype=float); c = int(cutoff); n = len(ave); k = int(shift)
    out = ave.copy()
    if c < n:
        idx = np.arange(c, n) - k
        ok = idx >= 0
        out[c:][ok] = ave[idx[ok]]
    return list(out)

class Rates:
    """pchip splines of b (ovi), d (dev), mu, pEA over day index, from smoothed station temperature.
    clim_from: if set, temperature from that day onward is replaced by its day-of-year climatology (E1 climatology scenario)."""
    def __init__(self, st, species, temp_window=None, clim_from=None, analogue=None):
        sp = SPECIES[species]; w = temp_window or sp["temp_window"]
        ave = analogue_temp(st, *analogue) if analogue is not None else (station_daily_temp(st) if clim_from is None else climatological_temp(st, clim_from))
        lw = list(ave[:w]) + [np.mean(ave[i-w:i]) for i in range(w, len(ave))]   # trailing mean, first w days raw
        Tc = [(t-32)*5/9 for t in lw]
        xs = np.arange(len(lw), dtype=float)
        th = sp["thermo"]
        self.b = pchip(xs, [th["ovi"](T) for T in Tc]); self.d = pchip(xs, [th["dev"](T) for T in Tc])
        self.mu = pchip(xs, [th["mu"](T) for T in Tc]); self.pEA = pchip(xs, [th["pEA"](T) for T in Tc])
        if "dL" in th: self.dL = pchip(xs, [th["dL"](T) for T in Tc]); self.dP = pchip(xs, [th["dP"](T) for T in Tc])
        self.station, self.species, self.tmax = st, species, xs[-1]

_rates_cache = {}
def rates(st, species, temp_window=None, clim_from=None, analogue=None):
    key = (st, species, temp_window, clim_from, analogue)
    if key not in _rates_cache: _rates_cache[key] = Rates(st, species, temp_window, clim_from, analogue)
    return _rates_cache[key]

# ---------------- count chain ----------------
def trap_observations(trap, species, zeros="drop", floor=0.5, cutoff=None, start="2017-01-01"):
    """Collection-date observations for one trap. zeros: 'drop' (original), 'keep', 'floor' (zeros -> floor).
    cutoff: last day index (exclusive) to use, e.g. day('2021-01-01'); None = full record (original)."""
    r = raw_counts(); col = SPECIES[species]["col"]
    d = r[r.TRAPID == trap].assign(c=pd.to_numeric(r.loc[r.TRAPID == trap, col], errors="coerce")).dropna(subset=["c", "t"])
    if zeros == "drop": d = d[d.c != 0]
    elif zeros == "floor": d = d.assign(c=d.c.where(d.c != 0, floor))
    if cutoff is not None: d = d[d.t < cutoff]
    d = d.drop_duplicates("t").sort_values("t")
    return d.t.values.astype(float), d.c.values.astype(float)

def smooth_series(t, y, smooth="trailing", window=14, start="2017-01-01"):
    """linear daily interpolation then smoothing. smooth: 'trailing' (original), 'centred', 'none',
    'spline<n>' (cubic smoothing spline on the raw counts, s = 10^-n * len(t) * var(y); no interpolation step),
    'log' (centred window applied to log(x+1)), 'gp' (Gaussian-process regression on log(count+1) at the
    collection dates, Matern 3/2 + white noise, evaluated daily; no interpolation step). Returns days, x."""
    if len(t) < 2: return None, None
    days = np.arange(t[0], t[-1] + 1)
    if smooth.startswith("pspline"):
        # Poisson Whittaker smoother: log-intensity eta on the daily grid, second-difference penalty lam * sum (D2 eta)^2,
        # fitted by IRLS / Newton with a banded sparse system; observations only at collection days, so the boundary is set by
        # the penalty (linear extrapolation of the trend) rather than by a one-sided window. lam = 10^(number after 'pspline').
        import scipy.sparse as sp, scipy.sparse.linalg as spl
        lam = 10.0 ** float(smooth[7:] or "3"); n = len(days); idx = (t - t[0]).astype(int)
        w = np.zeros(n); yy = np.zeros(n); w[idx] = 1.0; yy[idx] = y
        D = sp.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)); P = lam * (D.T @ D)
        eta = np.full(n, np.log(max(y.mean(), 0.1)))
        for it in range(50):
            mu = np.exp(eta); W = sp.diags(w * mu); z = w * (yy - mu)
            step = spl.spsolve((W + P).tocsc(), z - P @ eta); eta_new = eta + step
            if not np.all(np.isfinite(eta_new)): break
            done = np.max(np.abs(eta_new - eta)) < 1e-6; eta = np.clip(eta_new, -20, 20)
            if done: break
        x = np.exp(eta); m = days >= day(start); return days[m], x[m]
    if smooth.startswith("spline"):
        # cubic smoothing spline fitted directly to the collection-day counts, so the smoothing and the
        # differentiation are the same object rather than a rolling mean followed by a separate interpolant.
        # "spline<n>" sets the smoothing factor s = 10^-n * len(t) * var(y); larger n means a stiffer fit.
        from scipy.interpolate import UnivariateSpline
        rel = 10.0 ** (-float(smooth[6:] or "2"))
        s_ = rel * len(t) * max(np.var(y), 1e-12)
        sp_ = UnivariateSpline(t.astype(float), y.astype(float), k=3, s=s_)
        x = np.clip(sp_(days.astype(float)), 0.0, None)
        m = days >= day(start); return days[m], x[m]
    if smooth in ("centredP","centredZ","centredN"):
        # the same centred rolling mean as "centred", but with the daily grid filled a different way, so that the
        # interpolation step can be varied on its own with the smoothing held fixed:
        #   centredP  shape-preserving cubic (pchip) interpolation instead of linear
        #   centredZ  zero-order hold, the last observation carried forward
        #   centredN  no filling at all: each window's mean is taken over the observations actually inside it
        h = window // 2
        if smooth == "centredN":
            x = np.array([y[(t >= d - h) & (t <= d + h)].mean() if ((t >= d - h) & (t <= d + h)).any() else np.nan
                          for d in days])
            bad = ~np.isfinite(x)
            if bad.all(): return None, None
            if bad.any(): x[bad] = np.interp(days[bad], days[~bad], x[~bad])   # windows with no observation at all
        else:
            if smooth == "centredP":
                x = np.clip(pchip(t.astype(float), y.astype(float))(days.astype(float)), 0.0, None)
            else:
                idx = np.clip(np.searchsorted(t, days, side="right") - 1, 0, len(t) - 1); x = y[idx].astype(float)
            x = np.array([x[max(0, i-h):min(len(x), i+h+1)].mean() for i in range(len(x))])
        m = days >= day(start); return days[m], x[m]
    if smooth == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        z = np.log1p(y); k = ConstantKernel(1.0, (0.05, 20)) * Matern(length_scale=window, length_scale_bounds=(7, 120), nu=1.5) + WhiteKernel(0.3, (0.01, 5))
        gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0, random_state=0)
        gp.fit(t.reshape(-1, 1), z); x = np.clip(np.expm1(gp.predict(days.reshape(-1, 1))), 0, None)
    else:
        x = np.interp(days, t, y)
        if smooth == "trailing":
            x = np.array([x[max(0, i-window+1):i+1].mean() for i in range(len(x))])
        elif smooth == "centred":
            h = window // 2
            x = np.array([x[max(0, i-h):min(len(x), i+h+1)].mean() for i in range(len(x))])
        elif smooth == "log":
            h = window // 2; z = np.log1p(x)
            x = np.expm1(np.array([z[max(0, i-h):min(len(z), i+h+1)].mean() for i in range(len(z))]))
    m = days >= day(start)
    return days[m], x[m]

def a0_rule(cs, R, days, mult=2.0, cap=None, mean_count=None):
    """A0 from the admissibility of J: m = -min(x'/mu + x). Original: mult=2 -> 2*ceil(m) (0 if m<=0).
    cap: A0 <= cap*mean_count (days where J would be negative are reported by admissibility())."""
    jj = cs(days, 1) / R.mu(days) + cs(days); m = jj.min()
    A0 = mult * math.ceil(-m) if m < 0 else 0.0
    if cap is not None and mean_count is not None: A0 = min(A0, cap * mean_count)
    return float(A0), float(m)

def invert(cs, R, days, A0):
    """two-stage inverse: J(t), J'(t), k(t) (verbatim algebra from the generation notebook)"""
    x, x1, x2 = cs(days), cs(days, 1), cs(days, 2)
    d, d1, p, p1, mu, mu1, b = R.d(days), R.d(days, 1), R.pEA(days), R.pEA(days, 1), R.mu(days), R.mu(days, 1), R.b(days)
    A = x + A0
    J = 2*(x1 + mu*A)/d/p
    Jp = 2*(x2 + mu1*A + mu*x1)/d/p - 2*(d1*p + d*p1)*(x1 + mu*A)/d/d/p/p
    k = (b*A - Jp - d*J)/(b*J*A)
    return k, J, Jp

def k_T(R, days, A0=1.0):
    """temperature-only inverse (x == 0): shape independent of A0, scale 1/A0"""
    cs = pchip(days, np.zeros(len(days)))
    return invert(cs, R, days, A0)[0]

def recover(trap, species, zeros="drop", cutoff=None, smooth="trailing", window=14, mult=2.0, cap=None, start="2017-01-01", floor=0.5):
    """full chain for one trap -> dict(days, x, k, J, A0, m, kT, station, mean_count) or None"""
    st = trap_station().get(trap)
    if st is None: return None
    if smooth.startswith("reg"):              # regularised inversion (inverse_reg.py): "reg" = forward filter, "reg<n>" = plus n smoothing sweeps
        import inverse_reg; return inverse_reg.recover_reg(trap, species, cutoff=cutoff, window=window, mult=mult, cap=cap, start=start, floor=floor, sweeps=int(smooth[3:] or 0))
    t, y = trap_observations(trap, species, zeros=zeros, floor=floor, cutoff=cutoff)
    days, x = smooth_series(t, y, smooth=smooth, window=window, start=start)
    if days is None or len(days) < 30: return None
    R = rates(st, species); days = days[days <= R.tmax]; x = x[:len(days)]
    if smooth.startswith("spline"):
        # a C2 cubic spline through the smoothed daily values: both derivatives the algebra needs are exact,
        # where pchip is only C1 and its second derivative is piecewise linear
        from scipy.interpolate import UnivariateSpline
        cs = UnivariateSpline(days.astype(float), x, k=3, s=0)
    else:
        cs = pchip(days, x)
    A0, m = a0_rule(cs, R, days, mult=mult, cap=cap, mean_count=y.mean())
    if mult == 0: A0 = 0.0                      # proportional observation model (F4): counts = q*A, no additive offset; q drops out of the scaled target
    k, J, Jp = invert(cs, R, days, A0)
    if mult == 0:                                 # days where J <= 0 are inadmissible under the proportional model: mark k non-finite (callers interpolate or drop)
        k = np.where(J > 0, k, np.nan)
    return dict(trap=trap, station=st, days=days, x=x, k=k, J=J, A0=A0, m=m, kT=k_T(R, days), mean_count=float(y[y > 0].mean()) if (y > 0).any() else 0.0, n_obs=len(t), t=t, y=y)

def admissibility(res, jfloor=1e-3):
    k, J = res["k"], res["J"]
    return dict(neg_k=float((k < 0).mean()), near_zero_J=float((J < jfloor * np.nanmax(J)).mean()),
                A0_over_mean=res["A0"] / res["mean_count"] if res["mean_count"] > 0 else np.nan)

def winsor(a, lo=1, hi=99):
    return np.clip(a, *np.percentile(a, [lo, hi]))

def r2_on_kT(res, mask=None, robust=True):
    """variance of k explained by a linear fit on k_T (the diagnostic of 2026-09-14).
    robust=True winsorises k at 1-99% first so single-day spikes (J near 0) cannot dominate."""
    k, kT = res["k"], res["kT"]
    if mask is not None: k, kT = k[mask], kT[mask]
    ok = np.isfinite(k) & np.isfinite(kT)
    if ok.sum() < 30: return np.nan
    k = winsor(k[ok]) if robust else k[ok]; kT = kT[ok]
    X = np.c_[np.ones(len(k)), kT]; beta = np.linalg.lstsq(X, k, rcond=None)[0]
    return float(1 - ((k - X @ beta).var() / k.var()))

def spike_fraction(k, nmad=5):
    """fraction of days more than nmad median-absolute-deviations from the median of k"""
    k = k[np.isfinite(k)]; med = np.median(k); mad = np.median(np.abs(k - med))
    return float((np.abs(k - med) > nmad * mad).mean()) if mad > 0 else np.nan

# ---------------- forward ODE (verbatim from 'ML prediction output - single trap' notebook) ----------------
def forward(R, k_of_t, A0, days, y0=None):
    """J' = b(1 - J k)A - dJ ; A' = pEA d J/2 - mu A ; y0 = [A0, A0] as in the original. Returns J, A, count = A - A0."""
    def ode(y, t):
        J, A = y
        return [R.b(t)*(1 - J*k_of_t(t))*A - R.d(t)*J, R.pEA(t)*R.d(t)*J/2 - R.mu(t)*A]
    sol = odeint(ode, y0 if y0 is not None else [A0, A0], days)
    return sol[:, 0], sol[:, 1], sol[:, 1] - A0

def load_label(species, trap):
    """original label rows (unscaled k, A0) for one trap"""
    import glob, csv
    folder = os.path.join(HERE, "..", "ODE-2025", "label", SPECIES[species]["label"] + "_scale_baseline")
    rows = []
    for f in glob.glob(os.path.join(folder, "*.csv")):
        st = os.path.basename(f).split("_")[1]
        with open(f) as fh:
            rd = csv.reader(fh); next(rd)
            for x in rd:
                if x[5] == trap: rows.append((st, float(x[1]), float(x[2]), float(x[3])))
    return pd.DataFrame(rows, columns=["station", "t", "k", "A0"])
