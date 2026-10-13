"""
Phase 0 finalist comparison: regenerate targets (training-only, E0 changes), train CatBoost on weather + site features
(no target-derived features), forecast 180 days from an origin with the two-stage ODE, and score against observed counts.
Origin treatments: D0 = recovered k drives the ODE right up to the origin; D1 = the last `drop_days` days of targets before
the origin are removed from the label set and the ML prediction drives the ODE from origin-drop_days.
Usage: python mlode_pipeline.py [--species aedes|culex] [--origin 2020-06-01] [--quick]
"""
import os, sys, argparse, datetime, math, warnings, numpy as np, pandas as pd
from scipy.interpolate import pchip
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,"..","ODE-2025","src"))
import inverse_step_2stage as inv, inverse_step_3stage as i3

STAGES=[2]      # set to [3] by --stages 3; the recovery worker reads it
DESIGNS={"F0":dict(zeros="drop",smooth="trailing",window=14,mult=2.0,winsor=False),
         "F1":dict(zeros="keep",smooth="centred",window=28,mult=2.0,winsor=False),
         "F2":dict(zeros="keep",smooth="centred",window=28,mult=1.0,winsor=True),
         "F3":dict(zeros="keep",smooth="gp",window=30,mult=2.0,winsor=True),
         "F4":dict(zeros="keep",smooth="gp",window=30,mult=0.0,winsor=True),      # proportional observation model, GP smoothing
         "F4c":dict(zeros="keep",smooth="centred",window=28,mult=0.0,winsor=True), # proportional observation model, centred 28-day smoothing
         "R1":dict(zeros="keep",smooth="reg",window=14,mult=2.0,winsor=False),      # regularised sequential inversion (inverse_reg.py), 14-day windows, forward filter
         "R2":dict(zeros="keep",smooth="reg2",window=14,mult=2.0,winsor=False),     # regularised inversion with two smoothing sweeps (backward, forward)
         "R4":dict(zeros="keep",smooth="reg4",window=14,mult=2.0,winsor=False),     # option 2: four sweeps (block-coordinate joint fit to convergence)
         "P2":dict(zeros="keep",smooth="pspline2",window=28,mult=2.0,winsor=False), # option 3: Poisson P-spline smoother, lambda 1e2, then the algebraic inversion
         "P3":dict(zeros="keep",smooth="pspline3",window=28,mult=2.0,winsor=False),
         "P4":dict(zeros="keep",smooth="pspline4",window=28,mult=2.0,winsor=False),
         "P5":dict(zeros="keep",smooth="pspline5",window=28,mult=2.0,winsor=False),
         # A0 sweep (R1 #4, R2 #6): the offset multiplier, everything else as F1
         "A1":dict(zeros="keep",smooth="centred",window=28,mult=1.0,winsor=False),
         "A3":dict(zeros="keep",smooth="centred",window=28,mult=3.0,winsor=False),
         "A4":dict(zeros="keep",smooth="centred",window=28,mult=4.0,winsor=False),
         # rolling-window sweep (R2 #7): the centred smoothing width, everything else as F1
         "W7":dict(zeros="keep",smooth="centred",window=7,mult=2.0,winsor=False),
         "W14":dict(zeros="keep",smooth="centred",window=14,mult=2.0,winsor=False),
         "W21":dict(zeros="keep",smooth="centred",window=21,mult=2.0,winsor=False),
         "W35":dict(zeros="keep",smooth="centred",window=35,mult=2.0,winsor=False),
         "W42":dict(zeros="keep",smooth="centred",window=42,mult=2.0,winsor=False),
         "W56":dict(zeros="keep",smooth="centred",window=56,mult=2.0,winsor=False),
         "W70":dict(zeros="keep",smooth="centred",window=70,mult=2.0,winsor=False),
         "W84":dict(zeros="keep",smooth="centred",window=84,mult=2.0,winsor=False),
         "W112":dict(zeros="keep",smooth="centred",window=112,mult=2.0,winsor=False),
         "W140":dict(zeros="keep",smooth="centred",window=140,mult=2.0,winsor=False),
         # A0 multipliers between the bound and the published value, and above 4 (E6 asks for 5x and 10x)
         "A11":dict(zeros="keep",smooth="centred",window=28,mult=1.1,winsor=False),
         "A125":dict(zeros="keep",smooth="centred",window=28,mult=1.25,winsor=False),
         "A150":dict(zeros="keep",smooth="centred",window=28,mult=1.5,winsor=False),
         "A175":dict(zeros="keep",smooth="centred",window=28,mult=1.75,winsor=False),
         "A5":dict(zeros="keep",smooth="centred",window=28,mult=5.0,winsor=False),
         "A10":dict(zeros="keep",smooth="centred",window=28,mult=10.0,winsor=False),
         # the same multipliers at the wide window, to test whether the two knobs interact
         "X100":dict(zeros="keep",smooth="centred",window=112,mult=1.0,winsor=False),
         "X125":dict(zeros="keep",smooth="centred",window=112,mult=1.25,winsor=False),
         "X150":dict(zeros="keep",smooth="centred",window=112,mult=1.5,winsor=False),
         "X300":dict(zeros="keep",smooth="centred",window=112,mult=3.0,winsor=False),
         "X400":dict(zeros="keep",smooth="centred",window=112,mult=4.0,winsor=False),
         # cubic smoothing spline fitted straight to the counts, replacing linear interpolation + rolling mean +
         # pchip. The smoothing factor is s = 10^-n * len(t) * var(y); the window argument is unused.
         "SPL3":dict(zeros="keep",smooth="spline3",window=28,mult=2.0,winsor=False),
         "SPL2":dict(zeros="keep",smooth="spline2",window=28,mult=2.0,winsor=False),
         "SPL1":dict(zeros="keep",smooth="spline1",window=28,mult=2.0,winsor=False),
         # roughness-matched splines: the stiffness at which the spline's admissibility bound, and so A0, lands on
         # the same scale as the current chain's, so the comparison isolates the interpolation method from A0
         "SPLM":dict(zeros="keep",smooth="spline0.2",window=28,mult=2.0,winsor=False),   # Ae. aegypti match
         "SPLN":dict(zeros="keep",smooth="spline0.3",window=28,mult=2.0,winsor=False),   # Cx. quinquefasciatus match
         # the crossed A0 x window grid: multipliers 1, 1.5, 3, 4 at every window from 42 to 140 days.
         # The multiplier-2 column is W42..W140 and the 112-day row is X100..X400, so only these 20 are new.
         "M1W42":dict(zeros="keep",smooth="centred",window=42,mult=1.0,winsor=False), "M15W42":dict(zeros="keep",smooth="centred",window=42,mult=1.5,winsor=False), "M3W42":dict(zeros="keep",smooth="centred",window=42,mult=3.0,winsor=False), "M4W42":dict(zeros="keep",smooth="centred",window=42,mult=4.0,winsor=False),
         "M1W56":dict(zeros="keep",smooth="centred",window=56,mult=1.0,winsor=False), "M15W56":dict(zeros="keep",smooth="centred",window=56,mult=1.5,winsor=False), "M3W56":dict(zeros="keep",smooth="centred",window=56,mult=3.0,winsor=False), "M4W56":dict(zeros="keep",smooth="centred",window=56,mult=4.0,winsor=False),
         "M1W70":dict(zeros="keep",smooth="centred",window=70,mult=1.0,winsor=False), "M15W70":dict(zeros="keep",smooth="centred",window=70,mult=1.5,winsor=False), "M3W70":dict(zeros="keep",smooth="centred",window=70,mult=3.0,winsor=False), "M4W70":dict(zeros="keep",smooth="centred",window=70,mult=4.0,winsor=False),
         "M1W84":dict(zeros="keep",smooth="centred",window=84,mult=1.0,winsor=False), "M15W84":dict(zeros="keep",smooth="centred",window=84,mult=1.5,winsor=False), "M3W84":dict(zeros="keep",smooth="centred",window=84,mult=3.0,winsor=False), "M4W84":dict(zeros="keep",smooth="centred",window=84,mult=4.0,winsor=False),
         "M1W140":dict(zeros="keep",smooth="centred",window=140,mult=1.0,winsor=False), "M15W140":dict(zeros="keep",smooth="centred",window=140,mult=1.5,winsor=False), "M3W140":dict(zeros="keep",smooth="centred",window=140,mult=3.0,winsor=False), "M4W140":dict(zeros="keep",smooth="centred",window=140,mult=4.0,winsor=False)}
WCOLS=["p01i","relh_max","relh_ave","relh_min","tmpf_max","tmpf_ave","tmpf_min"]; NLAG=30

def load_features():
    cwd=os.getcwd(); os.chdir(os.path.join(HERE,"..","ODE-2025"))
    import util; feat=util.load_input_feature(); os.chdir(cwd); return feat

def analogue_weather(W, cutoff, shift):
    """copy of W with every row from `cutoff` onward replaced by the row `shift` days earlier, i.e. the observed
    weather of the same calendar period in an earlier year (shift a whole number of 365-day years)."""
    W=W.copy(); c=int(cutoff); n=len(W); k=int(shift)
    if c<n:
        idx=np.arange(c,n)-k; ok=idx>=0
        blk=W[c:]; blk[ok]=W[idx[ok]]; W[c:]=blk
    return W

def climatological_weather(W, cutoff):
    """copy of W with every row from `cutoff` onward replaced by the day-of-year mean of the rows before it"""
    W=W.copy(); c=int(cutoff); n=len(W)
    doy=(np.arange(n)+inv.DAY0.timetuple().tm_yday-1)%365
    tab=pd.DataFrame(W[:c]).groupby(doy[:c]).mean().reindex(range(365)).interpolate(limit_direction="both").bfill().ffill()
    sm=pd.DataFrame(np.vstack([tab.values[-7:],tab.values,tab.values[:7]])).rolling(15,center=True,min_periods=1).mean().values[7:-7]
    if c<n: W[c:]=sm[doy[c:]]
    return W

def weather_matrix(feat, trap):
    """daily weather array indexed by day since DAY0, 2016-08-01 .. 2022-12-31"""
    d0=inv.DAY0; n=(datetime.datetime(2022,12,31)-d0).days+1; W=np.full((n,7),np.nan)
    p=feat["traps_p01i_daily"].get(trap,{}); r=feat["traps_relh_daily"].get(trap,{}); t=feat["traps_tmpf_daily"].get(trap,{})
    for ds,v in p.items():
        i=(datetime.datetime.strptime(ds,"%Y-%m-%d")-d0).days
        if 0<=i<n: W[i,0]=v
    for ds,v in r.items():
        i=(datetime.datetime.strptime(ds,"%Y-%m-%d")-d0).days
        if 0<=i<n: W[i,1],W[i,2],W[i,3]=v["max"],v["ave"],v["min"]
    for ds,v in t.items():
        i=(datetime.datetime.strptime(ds,"%Y-%m-%d")-d0).days
        if 0<=i<n: W[i,4],W[i,5],W[i,6]=v["max"],v["ave"],v["min"]
    # fill gaps with previous day
    for j in range(7):
        col=W[:,j]; idx=np.where(~np.isnan(col))[0]
        if len(idx): W[:,j]=np.interp(np.arange(n),idx,col[idx])
    return W

def static_row(feat, trap, day):
    dt=inv.date(day); y,m=dt.year,dt.month
    try:
        return [feat["poverty_num_yearly"][trap][str(y)],feat["poverty_rate_yearly"][trap][str(y)],feat["trap_elevation"][trap],
                feat["num_sewer_manhole"][trap],feat["nearest_canal_distance"][trap],feat["vegetation_coverage_monthly"][trap][y][m],feat["water_body_percent_monthly"][trap][y][m]]
    except KeyError: return None

LONGW=[(30,59),(60,89),(90,119)]        # extra aggregation windows (days back), Phase 2B
def long_feats(W, d):
    """Weather memory beyond the 30 daily lags: means of each variable over three older windows, the dry-spell length
    (days since the last day with >1 mm), cumulative rainfall over 30/60/90 d, and degree-days above 10 C over 30/90 d."""
    d=int(d); out=[]
    for a,b in LONGW:
        seg=W[max(0,d-b):d-a+1]
        out+= list(seg.mean(0)) if len(seg) else [np.nan]*7
    p=W[max(0,d-119):d+1,0]                                   # precipitation, oldest first
    wet=np.where(p>1.0)[0]; out.append(float(len(p)-1-wet[-1]) if len(wet) else 120.0)
    for n in (30,60,90): out.append(float(W[max(0,d-n+1):d+1,0].sum()))
    tmean=(W[:,5]-32)*5/9
    for n in (30,90): out.append(float(np.clip(tmean[max(0,d-n+1):d+1]-10.0,0,None).sum()))
    return out
def build_X(feat, W, trap, days, long=False):
    rows=[]; keep=[]
    for d in days:
        s=static_row(feat,trap,int(d))
        if s is None or d-NLAG+1<0: continue
        lags=W[int(d)-NLAG+1:int(d)+1][::-1].ravel()          # lag 0 first, then 1..29, each 7 cols
        if np.isnan(lags).any(): continue
        extra=long_feats(W,d) if long else []
        if long and not np.all(np.isfinite(extra)): continue
        rows.append(s+list(lags)+extra); keep.append(d)
    return np.array(rows), np.array(keep)
FEATURE_NAMES=["poverty_num","poverty_rate","elevation","num_sewer_manhole","canal_dist","NDVI_monthly","water_body_monthly"]+[f"{c}_L{l}" for l in range(NLAG) for c in WCOLS]
LONG_NAMES=[f"{c}_m{a}_{b}" for (a,b) in LONGW for c in WCOLS]+["dry_spell","precip_sum30","precip_sum60","precip_sum90","degdays30","degdays90"]

def kclim_table(days, y):
    doy=(days%365.25).astype(int)//7; tab=pd.Series(y).groupby(doy).mean().reindex(range(53)).interpolate(limit_direction="both")
    return pd.Series(np.convolve(np.r_[tab.values[-2:],tab.values,tab.values[:2]],np.ones(5)/5,mode="valid"),index=range(53))
def kclim_at(tab, d): return np.array([tab[int((x%365.25)//7)] for x in np.atleast_1d(d)])
def aggregate_X(X):
    """217 -> 28 features: 7 static + for each of 7 weather variables the mean over lags 0-6, 7-13, 14-29"""
    st=X[:,:7]; W=X[:,7:].reshape(len(X),NLAG,7)
    return np.column_stack([st, W[:,0:7,:].mean(1), W[:,7:14,:].mean(1), W[:,14:30,:].mean(1)])

def make_model(ml, quick=False, seed=0, hp=None):
    if ml in ("catboost","anom","anom_agg","resid_kT","resid_kT_w","inv_resid","inv_k"):
        from catboost import CatBoostRegressor
        params=dict(iterations=300 if quick else 5000,learning_rate=0.05,depth=6,loss_function="RMSE",random_seed=seed,verbose=False,thread_count=int(os.environ.get("CB_THREADS","-1")))
        if hp: params.update(hp)
        return CatBoostRegressor(**params)
    if ml=="rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100 if quick else 300,min_samples_leaf=5,max_features=0.3,n_jobs=-1,random_state=seed)
    if ml=="lasso":
        # alpha chosen by BIC in a single least-angle pass: cross-validated Lasso is far too slow on 158k x 217
        from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LassoLarsIC
        return make_pipeline(StandardScaler(),LassoLarsIC(criterion="bic",max_iter=500))
    if ml=="ada":
        from sklearn.ensemble import AdaBoostRegressor
        return AdaBoostRegressor(n_estimators=100 if quick else 300,learning_rate=0.05,random_state=seed)
    if ml=="ridge":
        from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    if ml=="mlp":
        from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.neural_network import MLPRegressor
        return make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),early_stopping=True,max_iter=200 if not quick else 30,random_state=seed))
    return None

def _recover_one(args):
    t,species,des,origin=args
    try:
        if STAGES[0]==3: return i3.recover3(t,species,zeros=des["zeros"],cutoff=origin,smooth=des["smooth"],window=des["window"],mult=des["mult"])
        return inv.recover(t,species,zeros=des["zeros"],cutoff=origin,smooth=des["smooth"],window=des["window"],mult=des["mult"])
    except Exception: return None

def eligible_traps(species, origin):
    import glob, csv
    folder=os.path.join(HERE,"..","ODE-2025","label",inv.SPECIES[species]["label"]+"_scale_baseline")
    traps=set()
    for f in glob.glob(folder+"/*.csv"):
        with open(f) as fh:
            rd=csv.reader(fh); next(rd)
            for x in rd: traps.add(x[5])
    out=[]
    for t in sorted(traps):
        tt,y=inv.trap_observations(t,species,zeros="drop",cutoff=origin)
        if len(tt)<2: continue
        span=tt[-1]-tt[0]
        if span>180 and len(tt)/(1+span)>=0.09 and tt[0]<origin-180: out.append(t)
    return out

def climatology(trap, species, origin):
    r=inv.raw_counts(); col=inv.SPECIES[species]["col"]
    d=r[(r.TRAPID==trap)&(r.t<origin)].assign(c=pd.to_numeric(r.loc[r.TRAPID==trap,col],errors="coerce")).dropna(subset=["c","t"]).drop_duplicates("t")
    d=d[d.t>=inv.day("2017-01-01")]
    wk=((d.t.values.astype(int))//7)%52
    tab=pd.Series(d.c.values).groupby(wk).mean().reindex(range(52))
    tab=tab.interpolate(limit_direction="both"); tab=pd.Series(np.convolve(np.r_[tab.values[-2:],tab.values,tab.values[:2]],np.ones(5)/5,mode="valid"),index=range(52))   # 5-week centred smoothing, circular
    return tab, float(d.c.mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--species",default="aedes"); ap.add_argument("--origin",default="2020-06-01"); ap.add_argument("--quick",action="store_true")
    ap.add_argument("--drop_static",action="store_true",help="blank the seven static site features, as an ablation"); ap.add_argument("--designs",default="F0,F1,F2,F3,F4,F4c"); ap.add_argument("--drop_days",type=int,default=14); ap.add_argument("--horizon",type=int,default=180); ap.add_argument("--valid_days",type=int,default=122); ap.add_argument("--ml",default="catboost",help="catboost | rf | ridge | mlp | lasso | ada | oracle | clim_k | anom | anom_agg | resid_kT | resid_kT_w (residual winsorised 1-99%) | inv_resid (target 1/k - 1/k_T) | inv_k (target 1/k); both interpretation only"); ap.add_argument("--suffix",default="",help="appended to output file names"); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--stages",type=int,default=2,choices=[2,3],help="2 = juvenile/adult (published); 3 = larva/pupa/adult, using inverse_step_3stage"); ap.add_argument("--save_model",action="store_true"); ap.add_argument("--ml_only",action="store_true",help="fit and score the learner, skip the forward ODE and the per-trap forecast"); ap.add_argument("--hp",default="",help="JSON of CatBoost parameter overrides, e.g. '{\"depth\":8,\"learning_rate\":0.02}'"); ap.add_argument("--climate_scenario",default="",help="'clim': day-of-year climatology over the forecast window (E1 worst case); 'yearN': the observed weather of N years before the origin, an analogue-year ensemble member"); ap.add_argument("--long_feats",action="store_true",help="Phase 2B: add 30/60/90-day weather aggregations, dry-spell length, cumulative rainfall and degree-days"); ap.add_argument("--anom_pool",default="",help="Phase 2B: pool the anomaly target across the station's traps ('station'); prediction adds the trap's own climatological k back"); ap.add_argument("--kclim_pool",type=float,default=0.0,help="option 5: pool the climatological k with the station mean table, weight n_years/(n_years + C); 0 = off"); ap.add_argument("--target_smooth",type=int,default=0,help="step 2: centred window (days) over which the daily anomaly target is averaged before training (0 = off)"); ap.add_argument("--k_last_avg",type=int,default=0,help="step 3: number of last recovered k values averaged to form k_last for persistence (0 = the last value)"); ap.add_argument("--k_persist",type=float,default=0.0,help="time constant (days) for decaying the last recovered k value into the ML k after origin-drop_days; 0 = off"); ap.add_argument("--state_fit",action="store_true",help="rescale the ODE state at origin-drop_days so the simulated adult level matches the last 28 days of pre-origin collections; adds column fc_sf")
    a=ap.parse_args(); STAGES[0]=a.stages; species=a.species; origin=inv.day(a.origin); H=a.horizon
    import json; hp=json.loads(a.hp) if a.hp else None
    feat=load_features(); traps=eligible_traps(species,origin)
    if a.quick: traps=traps[:25]
    print(species,"eligible traps",len(traps),flush=True)
    Wm={t:weather_matrix(feat,t) for t in traps}
    _ana=None
    if a.climate_scenario.startswith("year"):
        _ana=(origin,365*int(a.climate_scenario[4:]))
        Wm={t:analogue_weather(W,origin,_ana[1]) for t,W in Wm.items()}
        print("climate scenario: weather and ODE temperature taken from",a.climate_scenario[4:],"year(s) before the origin",flush=True)
    elif a.climate_scenario=="clim": Wm={t:climatological_weather(W,origin) for t,W in Wm.items()}; print("climate scenario: weather and ODE temperature from day-of-year climatology from the origin onward",flush=True)
    raw=inv.raw_counts(); col=inv.SPECIES[species]["col"]
    results=[]; mlrows=[]; preds_store={}; obsrows=[]
    for dname in a.designs.split(","):
        des=DESIGNS[dname]
        # ---- targets ----
        rec={}
        if des["smooth"].startswith("reg"):
            from multiprocessing import Pool
            with Pool(int(os.environ.get("REC_PROCS","8"))) as pool: rec_list=pool.map(_recover_one,[(t,species,des,float(origin)) for t in traps])
        else: rec_list=[_recover_one((t,species,des,float(origin))) for t in traps]
        for t,res in zip(traps,rec_list):
            if res is None: continue
            k=res["k"].copy(); bad=~np.isfinite(k)
            if bad.mean()>0.05: continue
            if bad.any(): k[bad]=np.interp(res["days"][bad],res["days"][~bad],k[~bad])
            if des["winsor"]: k=inv.winsor(k)
            res["k_use"]=k; res["scale"]=float(np.nanmax(k))
            if STAGES[0]==3:
                L0=float(res["L"][0]) if np.isfinite(res["L"][0]) and res["L"][0]>0 else max(res["A0"],1.0)
                P0=float(res["P"][0]) if np.isfinite(res["P"][0]) and res["P"][0]>0 else max(res["A0"],1.0)
                res["y0"]=[L0,P0,float(res["x"][0]+res["A0"])]
            else:
                J0=float(res["J"][0]) if np.isfinite(res["J"][0]) and res["J"][0]>0 else max(res["A0"],1.0)
                res["y0"]=[J0,float(res["x"][0]+res["A0"])]          # initial state from the recovered series (original used [A0, A0])
            if res["scale"]<=0: continue
            rec[t]=res
        # station-pooled anomaly target (Phase 2B): daily mean anomaly over the station's traps, used as the training target
        pool_series={}
        if a.anom_pool=="station" and a.ml in ("anom","anom_agg"):
            acc={}
            for t,res in rec.items():
                days=res["days"]; y=res["k_use"]/res["scale"]; m_=days<origin-a.valid_days
                if m_.sum()<100: continue
                tab=kclim_table(days[m_],y[m_]); an=pd.Series(y-kclim_at(tab,days),index=days.astype(int))
                acc.setdefault(res["station"],[]).append(an)
            for st_,v in acc.items(): pool_series[st_]=pd.concat(v,axis=1).mean(axis=1)
        station_tab={}
        if a.kclim_pool>0 and a.ml in ("anom","anom_agg"):
            acc={}
            for t,res in rec.items():
                days=res["days"]; y=res["k_use"]/res["scale"]; m_=days<origin-a.valid_days
                if m_.sum()<100: continue
                acc.setdefault(res["station"],[]).append(kclim_table(days[m_],y[m_]).values)
            station_tab={st_:pd.Series(np.mean(np.vstack(v),axis=0),index=range(53)) for st_,v in acc.items()}
        for D in (["D0","D1"]):
            drop=a.drop_days if D=="D1" else 0
            Xtr,ytr,Xva,yva,gva,dva=[],[],[],[],[],[]; ndrop_tot=[0]
            for t,res in rec.items():
                days=res["days"]; y=res["k_use"]/res["scale"]
                m=days<origin-drop
                X,kd=build_X(feat,Wm[t],t,days[m],long=a.long_feats)
                if a.drop_static and len(X): X[:,:7]=0.0     # ablation: blank the seven constant site features
                if len(kd)==0: continue
                yy=np.interp(kd,days,y)
                if a.ml in ("anom","anom_agg"):
                    res["kclim"]=kclim_table(days[days<origin-a.valid_days],y[days<origin-a.valid_days])
                    if a.kclim_pool>0 and res["station"] in station_tab:
                        n_years=float((days<origin-a.valid_days).sum())/365.25; wgt=n_years/(n_years+a.kclim_pool)
                        res["kclim"]=wgt*res["kclim"]+(1-wgt)*station_tab[res["station"]]
                    if a.anom_pool=="station" and res["station"] in pool_series:
                        ps=pool_series[res["station"]]; yy=np.interp(kd,ps.index.values.astype(float),ps.values)
                    elif a.target_smooth>0:
                        an=y-kclim_at(res["kclim"],days); h=a.target_smooth//2; an_s=np.array([an[max(0,i-h):min(len(an),i+h+1)].mean() for i in range(len(an))]); yy=np.interp(kd,days,an_s)
                    else: yy=yy-kclim_at(res["kclim"],kd)
                if a.ml in ("resid_kT","resid_kT_w"):
                    # fit scaled k on the station's temperature-only k_T over the training period (winsorised), residual is the target
                    kT=res["kT"]; trm=days<origin-a.valid_days; ok=trm&np.isfinite(kT)&np.isfinite(y)
                    if ok.sum()<100: continue
                    beta=np.polyfit(kT[ok],inv.winsor(y[ok]),1); res["kT_fit"]=beta
                    yy=yy-np.polyval(beta,np.interp(kd,days,kT))
                    if a.ml=="resid_kT_w": yy=inv.winsor(yy)
                if a.ml=="inv_k":
                    # the carrying capacity itself, 1/k(t), divided by its own training-period median so traps of
                    # very different abundance contribute on a common scale. Interpretation only, as for inv_resid.
                    kk=np.asarray(res["k_use"],dtype=float); trm=days<origin-a.valid_days
                    ia=np.where(np.isfinite(kk)&(kk>0),1.0/np.where(kk>0,kk,np.nan),np.nan)
                    with np.errstate(invalid="ignore"):
                        ma=np.nanmedian(ia[trm]) if trm.any() else np.nan
                    if not (np.isfinite(ma) and ma>0): continue
                    dd_=ia/ma; gd=np.isfinite(dd_)
                    if gd.sum()<100: continue
                    yy=np.interp(kd,days[gd],dd_[gd])
                if a.ml=="inv_resid":
                    # the count-driven part of the carrying capacity, 1/k - 1/k_T, each divided by its own
                    # training-period median so the two are on a common scale (k_T is computed with A0 fixed at 1).
                    # This is exactly the quantity plotted in phase1/improve/cc_thermal_ts.py; here it is the
                    # learning target. Interpretation only -- there is no forward step for this mode.
                    kk=np.asarray(res["k_use"],dtype=float); kT=np.asarray(res["kT"],dtype=float)
                    trm=days<origin-a.valid_days
                    ia=np.where(np.isfinite(kk)&(kk>0),1.0/np.where(kk>0,kk,np.nan),np.nan)
                    ib=np.where(np.isfinite(kT)&(kT>0),1.0/np.where(kT>0,kT,np.nan),np.nan)
                    with np.errstate(invalid="ignore"):
                        ma=np.nanmedian(ia[trm]) if trm.any() else np.nan
                        mb=np.nanmedian(ib[trm]) if trm.any() else np.nan
                    if not (np.isfinite(ma) and ma>0 and np.isfinite(mb) and mb>0): continue
                    dd_=ia/ma-ib/mb; gd=np.isfinite(dd_)
                    if gd.sum()<100: continue
                    yy=np.interp(kd,days[gd],dd_[gd])
                if a.ml=="anom_agg": X=aggregate_X(X)
                tr=kd<origin-a.valid_days
                fin=np.isfinite(yy)          # a long validation slice can leave too little training history to build the
                if not fin.all():            # per-trap climatological k for every day of year; drop those rows rather than fail
                    nbad=int((~fin).sum())
                    if nbad>0.5*len(yy): continue
                    ndrop_tot[0]+=nbad
                trf=tr&fin; vaf=(~tr)&fin
                if trf.sum()==0 or vaf.sum()==0: continue
                Xtr.append(X[trf]); ytr.append(yy[trf]); Xva.append(X[vaf]); yva.append(yy[vaf]); gva+= [t]*int(vaf.sum()); dva.append(kd[vaf])
            if ndrop_tot[0]: print(f"  {dname} {D}: dropped {ndrop_tot[0]} rows with a non-finite target (short training history)",flush=True)
            if not Xtr or not Xva: print(f"  {dname} {D}: no usable traps, skipped",flush=True); continue
            Xtr=np.vstack(Xtr); ytr=np.concatenate(ytr); Xva=np.vstack(Xva); yva=np.concatenate(yva)
            model=make_model(a.ml,a.quick,a.seed,hp); best_iter=-1; fam={}
            if a.ml in ("catboost","anom","anom_agg","resid_kT","resid_kT_w","inv_resid","inv_k"):
                model.fit(Xtr,ytr,eval_set=(Xva,yva),use_best_model=True); best_iter=model.get_best_iteration()
                if a.ml!="anom_agg":
                    for nm,v in zip(FEATURE_NAMES+(LONG_NAMES if a.long_feats else []),model.get_feature_importance()): g=nm.split("_L")[0] if "_L" in nm else nm; fam[g]=fam.get(g,0)+v
            elif a.ml in ("rf","ridge","mlp","lasso","ada"): model.fit(Xtr,ytr)
            if a.ml in ("oracle","clim_k"): pva=np.full(len(yva),np.nan); mae=medae=np.nan
            else: pva=model.predict(Xva); mae=float(np.mean(np.abs(pva-yva))); medae=float(np.median(np.abs(pva-yva)))
            null=float(np.mean(np.abs(ytr.mean()-yva)))
            if a.save_model and a.ml not in ("oracle","clim_k"):
                import pickle
                with open(f"model_{species}_{a.origin}_{a.ml}{a.suffix}_{dname}_{D}.pkl","wb") as fh: pickle.dump(dict(model=model,Xva=Xva,yva=yva,gva=np.array(gva),dva=np.concatenate(dva),feature_names=FEATURE_NAMES+(LONG_NAMES if a.long_feats else []),design=dname,D=D,ml=a.ml,species=species,origin=a.origin),fh)
            mlrows.append(dict(species=species,design=dname,ml=a.ml,D=D,n_train=len(ytr),n_valid=len(yva),best_iter=best_iter,valid_mae=mae,valid_medae=medae,valid_mae_null=null,
                               top_families=", ".join(f"{k}={v:.1f}" for k,v in sorted(fam.items(),key=lambda x:-x[1])[:6])))
            print(f"  {dname} {D} {a.ml}: train {len(ytr)} valid {len(yva)} best_iter {best_iter} valid MAE {mae:.4f} (null {null:.4f})",flush=True)
            if a.ml_only: continue      # learner-comparison runs need only the validation error, not the forecast
            # ---- forecast + score per trap ----
            for t,res in rec.items():
                R=inv.rates(res["station"],species,clim_from=origin if a.climate_scenario=="clim" else None,analogue=_ana); days=res["days"]; kuse=res["k_use"]
                pdays=np.arange(max(days[0],origin-drop-1),origin+H,dtype=float)
                Xp,kp=build_X(feat,Wm[t],t,pdays,long=a.long_feats)
                if a.drop_static and len(Xp): Xp[:,:7]=0.0
                if len(kp)<H*0.8: continue
                if a.ml=="oracle":
                    full=(i3.recover3 if STAGES[0]==3 else inv.recover)(t,species,zeros=des["zeros"],cutoff=origin+H+40,smooth=des["smooth"],window=des["window"],mult=des["mult"])
                    if full is None: continue
                    kf_=full["k"].copy(); bad=~np.isfinite(kf_); kf_[bad]=np.interp(full["days"][bad],full["days"][~bad],kf_[~bad]) if (~bad).any() else 0
                    if des["winsor"]: kf_=inv.winsor(kf_)
                    khat=np.interp(kp,full["days"],kf_)
                elif a.ml=="clim_k":
                    y=res["k_use"]/res["scale"]; doy=(days%365.25).astype(int)//7; tab=pd.Series(y).groupby(doy).mean().reindex(range(53)).interpolate(limit_direction="both")
                    tab=pd.Series(np.convolve(np.r_[tab.values[-2:],tab.values,tab.values[:2]],np.ones(5)/5,mode="valid"),index=range(53))
                    khat=np.array([tab[int((d%365.25)//7)] for d in kp])*res["scale"]
                elif a.ml in ("resid_kT","resid_kT_w"):
                    if "kT_fit" not in res: continue
                    kTw=inv.k_T(R,kp)                                      # temperature-only k over the window from the window's own temperature
                    khat=(model.predict(Xp)+np.polyval(res["kT_fit"],kTw))*res["scale"]
                elif a.ml in ("anom","anom_agg"):
                    if "kclim" not in res: continue
                    Xq=aggregate_X(Xp) if a.ml=="anom_agg" else Xp
                    khat=(model.predict(Xq)+kclim_at(res["kclim"],kp))*res["scale"]
                else: khat=model.predict(Xp)*res["scale"]
                # combined k: recovered where allowed, ML elsewhere; optional persistence of the last recovered k into the ML k (gap-1 test)
                if a.k_persist>0 and (days<origin-drop).any():
                    kk=kuse[days<origin-drop]; k_last=float(kk[-max(1,a.k_last_avg):].mean()) if a.k_last_avg>0 else float(kk[-1]); wk=np.exp(-np.clip(kp-(origin-drop),0,None)/a.k_persist); khat=np.where(kp>=origin-drop,wk*k_last+(1-wk)*khat,khat)
                allk_days=np.concatenate([days[days<origin-drop],kp[kp>=origin-drop]]); allk=np.concatenate([kuse[days<origin-drop],khat[kp>=origin-drop]])
                order=np.argsort(allk_days); allk_days,allk=allk_days[order],allk[order]
                kf=lambda tt: np.interp(tt,allk_days,allk)
                sim_days=np.arange(days[0],origin+H,dtype=float)
                try:
                    if STAGES[0]==3: _L,_P,A,cnt=i3.forward3(R,kf,res["A0"],sim_days,y0=res["y0"]); J=_P
                    else: J,A,cnt=inv.forward(R,kf,res["A0"],sim_days,y0=res["y0"])
                except Exception: continue
                cnt=np.clip(cnt,0,None)
                obs=raw[(raw.TRAPID==t)&(raw.t>=origin)&(raw.t<origin+H)].assign(c=pd.to_numeric(raw.loc[raw.TRAPID==t,col],errors="coerce")).dropna(subset=["c","t"]).drop_duplicates("t")
                if len(obs)<5: continue
                yo=obs.c.values.astype(float); to=obs.t.values.astype(float); yf=np.interp(to,sim_days,cnt)
                # ---- origin-state fit (item 1a): rescale (J, A) at origin-drop so the simulated adult level matches the last 28 pre-origin days ----
                fs=yf.copy(); s_scale=1.0
                if a.state_fit:
                    pre_=raw[(raw.TRAPID==t)&(raw.t<origin)&(raw.t>=origin-28)].assign(c=pd.to_numeric(raw.loc[raw.TRAPID==t,col],errors="coerce")).dropna(subset=["c","t"]).drop_duplicates("t")
                    if len(pre_)>=2:
                        tp=pre_.t.values.astype(float); Asim=np.interp(tp,sim_days,A)
                        s_scale=float(np.clip((pre_.c.mean()+res["A0"])/max(Asim.mean(),1e-6),0.5,2.0))
                        i0=int(np.searchsorted(sim_days,origin-drop)); i0=min(max(i0,0),len(sim_days)-2); days2=sim_days[i0:]
                        try:
                            if STAGES[0]==3: _l,_p,A2,cnt2=i3.forward3(R,kf,res["A0"],days2,y0=[float(res["y0"][0])*s_scale,float(J[i0])*s_scale,float(A[i0])*s_scale])
                            else: J2,A2,cnt2=inv.forward(R,kf,res["A0"],days2,y0=[float(J[i0])*s_scale,float(A[i0])*s_scale])
                            fs=np.interp(to,days2,np.clip(cnt2,0,None))
                        except Exception: fs=yf.copy(); s_scale=1.0
                clim,cmean=climatology(t,species,origin); yc=np.array([clim.get((int(x)//7)%52,cmean) for x in to]); yk=np.full(len(to),cmean)
                def sc(yp,mask=None):
                    o,p=(yo,yp) if mask is None else (yo[mask],yp[mask])
                    if len(o)<3 or o.mean()==0: return np.nan,np.nan,np.nan
                    return float(np.sqrt(np.mean((p-o)**2))),float(np.sqrt(np.mean((p-o)**2))/o.mean()),float(np.mean(np.abs(p-o)))
                rm,rr,ma=sc(yf); rc,_,_=sc(yc); rk,_,_=sc(yk); nz=yo>0; _,rr_nz,_=sc(yf,nz)
                thr=10; hit=int(((yo>=thr)&(yf>=thr)).sum()); miss=int(((yo>=thr)&(yf<thr)).sum()); fa=int(((yo<thr)&(yf>=thr)).sum()); cn=int(((yo<thr)&(yf<thr)).sum())
                hitc=int(((yo>=thr)&(yc>=thr)).sum()); fac=int(((yo<thr)&(yc>=thr)).sum())
                results.append(dict(species=species,design=dname,D=D,trap=t,station=res["station"],A0=res["A0"],A0_over_mean=res["A0"]/res["mean_count"],n_obs=len(yo),obs_mean=float(yo.mean()),
                    rmse=rm,rel_rmse=rr,rel_rmse_nonzero=rr_nz,mae=ma,rmse_clim=rc,rmse_const=rk,skill_clim=1-rm/rc if rc>0 else np.nan,skill_const=1-rm/rk if rk>0 else np.nan,
                    spearman=float(spearmanr(yo,yf).correlation) if len(set(yo))>2 else np.nan,state_scale=s_scale,hit=hit,miss=miss,fa=fa,cn=cn,hit_clim=hitc,fa_clim=fac,fc_mean=float(yf.mean())))
                pre=raw[(raw.TRAPID==t)&(raw.t<origin)&(raw.t>=origin-21)].assign(c=pd.to_numeric(raw.loc[raw.TRAPID==t,col],errors="coerce")).dropna(subset=["c","t"])
                if len(pre): tl=float(pre.t.max()); anom=float(pre.loc[pre.t==tl,"c"].mean())-float(clim.get((int(tl)//7)%52,cmean))
                else: anom=0.0
                ypers=np.clip(yc+anom*np.exp(-(to-origin)/21.0),0,None)
                obsrows+= [dict(design=dname,D=D,trap=t,t=float(tt),lead=float(tt-origin),obs=float(o),fc=float(f),fc_sf=float(f2),clim=float(c),const=float(cmean),pers=float(pp)) for tt,o,f,f2,c,pp in zip(to,yo,yf,fs,yc,ypers)]
                if t in ("MB55B","CW57","CW5") and D=="D0": preds_store[(dname,t)]=(sim_days,cnt,to,yo,yc)
        print("design done",dname,flush=True)
    df=pd.DataFrame(results); ml=pd.DataFrame(mlrows)
    tag=f"{species}_{a.origin}"+("" if a.ml=="catboost" else f"_{a.ml}")+a.suffix+("_quick" if a.quick else "")
    df.to_csv(f"mlode_results_{tag}.csv",index=False); ml.to_csv(f"mlode_models_{tag}.csv",index=False); ob=pd.DataFrame(obsrows); ob.to_csv(f"mlode_obs_{tag}.csv",index=False)
    def alert(g):
        H_=g.hit.sum(); M=g.miss.sum(); F=g.fa.sum(); C=g.cn.sum(); Hc=g.hit_clim.sum(); Fc=g.fa_clim.sum()
        hr=H_/max(H_+M,1); far=F/max(F+C,1); hrc=Hc/max(H_+M,1); farc=Fc/max(F+C,1)
        return pd.Series(dict(hit_rate=hr,false_alarm=far,peirce=hr-far,peirce_clim=hrc-farc,precision=H_/max(H_+F,1),event_rate=(H_+M)/max(H_+M+F+C,1)))
    with open(f"mlode_summary_{tag}.md","w") as fh:
        fh.write(f"# Phase 0 finalist comparison: {species}, origin {a.origin}, horizon {H} d, valid = last {a.valid_days} d before origin, D1 drops last {a.drop_days} d of targets\n\n")
        fh.write("Targets training-only; no target-derived features; one station per trap; ODE initial state taken from the recovered series at its first day (original used [A0, A0]); CatBoost 5000 it, lr 0.05, depth 6, early stopping on validation. Climatology = trap week-of-year mean over training years, 5-week centred smoothing. Scores on all observations in the window (zeros included) unless the column says nonzero. Skill = 1 - RMSE/RMSE_baseline (climatology = trap week-of-year mean over training years; const = trap training mean). Medians over traps.\n\n")
        fh.write("## ML step\n\n"+ml.round(4).to_markdown(index=False)+"\n\n")
        g=df.groupby(["design","D"]).agg(traps=("trap","size"),A0_over_mean=("A0_over_mean","median"),rel_rmse=("rel_rmse","median"),rel_rmse_nonzero=("rel_rmse_nonzero","median"),mae=("mae","median"),
            skill_clim=("skill_clim","median"),frac_pos_skill_clim=("skill_clim",lambda s:(s>0).mean()),skill_const=("skill_const","median"),spearman=("spearman","median"),fc_over_obs=("fc_mean",lambda s: np.nan)).reset_index()
        g["fc_over_obs"]=df.groupby(["design","D"]).apply(lambda d:(d.fc_mean/d.obs_mean).median()).values
        fh.write("## Forecast skill by design and origin treatment\n\n"+g.round(3).to_markdown(index=False)+"\n\n")
        fh.write("## Alert skill at threshold 10 (pooled over traps and collections)\n\n"+df.groupby(["design","D"]).apply(alert).round(3).to_markdown()+"\n\n")
        # lead-time breakdown: per trap RMSE within lead bins, skill vs climatology, then median over traps; pooled alert skill per bin
        ob["bin"]=pd.cut(ob.lead,[-1,30,90,180],labels=["1-30 d","31-90 d","91-180 d"]); ob["cum90"]=ob.lead<=90
        def trap_skill(g):
            r=np.sqrt(np.mean((g.fc-g.obs)**2)); rc=np.sqrt(np.mean((g.clim-g.obs)**2)); rk=np.sqrt(np.mean((g.const-g.obs)**2))
            rp=np.sqrt(np.mean((g.pers-g.obs)**2))
            return pd.Series(dict(n=len(g),rel_rmse=r/g.obs.mean() if g.obs.mean()>0 else np.nan,skill_clim=1-r/rc if rc>0 else np.nan,skill_const=1-r/rk if rk>0 else np.nan,skill_pers=1-r/rp if rp>0 else np.nan))
        rows=[]
        for (dn,D_,b),g in ob.groupby(["design","D","bin"],observed=True):
            ts=g.groupby("trap").apply(trap_skill); ts=ts[ts.n>=3]
            hit=((g.obs>=10)&(g.fc>=10)).sum(); miss=((g.obs>=10)&(g.fc<10)).sum(); fa=((g.obs<10)&(g.fc>=10)).sum(); cn=((g.obs<10)&(g.fc<10)).sum()
            hc=((g.obs>=10)&(g.clim>=10)).sum(); fc_=((g.obs<10)&(g.clim>=10)).sum()
            rows.append(dict(design=dn,D=D_,lead=b,traps=len(ts),rel_rmse=ts.rel_rmse.median(),skill_clim=ts.skill_clim.median(),frac_pos=(ts.skill_clim>0).mean(),skill_const=ts.skill_const.median(),skill_pers=ts.skill_pers.median(),
                             peirce=hit/max(hit+miss,1)-fa/max(fa+cn,1),peirce_clim=hc/max(hit+miss,1)-fc_/max(fa+cn,1)))
        for (dn,D_),g in ob[ob.cum90].groupby(["design","D"]):
            ts=g.groupby("trap").apply(trap_skill); ts=ts[ts.n>=3]
            hit=((g.obs>=10)&(g.fc>=10)).sum(); miss=((g.obs>=10)&(g.fc<10)).sum(); fa=((g.obs<10)&(g.fc>=10)).sum(); cn=((g.obs<10)&(g.fc<10)).sum()
            hc=((g.obs>=10)&(g.clim>=10)).sum(); fc_=((g.obs<10)&(g.clim>=10)).sum()
            rows.append(dict(design=dn,D=D_,lead="1-90 d (cumulative)",traps=len(ts),rel_rmse=ts.rel_rmse.median(),skill_clim=ts.skill_clim.median(),frac_pos=(ts.skill_clim>0).mean(),skill_const=ts.skill_const.median(),skill_pers=ts.skill_pers.median(),
                             peirce=hit/max(hit+miss,1)-fa/max(fa+cn,1),peirce_clim=hc/max(hit+miss,1)-fc_/max(fa+cn,1)))
        fh.write("## Skill by lead time (per-trap medians within each lead bin; alert skill pooled)\n\n"+pd.DataFrame(rows).round(3).to_markdown(index=False)+"\n\n")
        base=df[df.design=="F0"] if (df.design=="F0").any() else df[df.design==df.design.iloc[0]]
        fh.write("## Baselines (same traps, F0 rows)\n\n"+pd.DataFrame(dict(rel_rmse_clim=[(base.rmse_clim/base.obs_mean).median()],rel_rmse_const=[(base.rmse_const/base.obs_mean).median()])).round(3).to_markdown(index=False)+"\n")
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys=[k for k in preds_store]; 
    if keys:
        traps_f=sorted(set(k[1] for k in keys)); fig,ax=plt.subplots(len(traps_f),1,figsize=(12,3.2*len(traps_f)),squeeze=False)
        for i,t in enumerate(traps_f):
            for dname in a.designs.split(","):
                if (dname,t) in preds_store:
                    sd,cnt,to,yo,yc=preds_store[(dname,t)]; m=sd>=origin-60; ax[i,0].plot([inv.date(x) for x in sd[m]],cnt[m],label=f"{dname} forecast")
            ax[i,0].scatter([inv.date(x) for x in to],yo,color="k",s=12,label="observed"); ax[i,0].plot([inv.date(x) for x in to],yc,ls=":",color="grey",label="climatology"); ax[i,0].axvline(inv.date(origin),color="k",lw=0.8)
            ax[i,0].set_title(f"{t} ({species}), origin {a.origin}, D0",fontsize=10); ax[i,0].legend(fontsize=7,ncol=3)
        plt.tight_layout(); plt.savefig(f"mlode_forecasts_{tag}.png",dpi=120)
    print("saved",tag)
if __name__=="__main__": main()
