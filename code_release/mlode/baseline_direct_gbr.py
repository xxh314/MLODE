"""
E2b: direct gradient boosting on trap counts (no ODE), same features as MLODE, same traps, window and scoring.
Variant A 'weather': site + weather lags + week-of-year, Poisson loss, count target on real collection dates.
Variant B 'persist': A + last observed count before the origin (and its climatological anomaly) + lead time, trained by sampling origins.
Usage: python baseline_baseline_direct_gbr.py --species aedes --origin 2020-06-01
"""
import os, sys, argparse, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import inverse_step_2stage as inv, mlode_pipeline as P
from catboost import CatBoostRegressor
rng=np.random.default_rng(0)

def collections(trap,species,t0,t1):
    r=inv.raw_counts(); col=inv.SPECIES[species]["col"]
    d=r[(r.TRAPID==trap)&(r.t>=t0)&(r.t<t1)].assign(c=pd.to_numeric(r.loc[r.TRAPID==trap,col],errors="coerce")).dropna(subset=["c","t"]).drop_duplicates("t").sort_values("t")
    return d.t.values.astype(float), d.c.values.astype(float)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--species",default="aedes"); ap.add_argument("--origin",default="2020-06-01"); ap.add_argument("--horizon",type=int,default=180); ap.add_argument("--valid_days",type=int,default=122); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--suffix",default=""); ap.add_argument("--climate_scenario",default="",help="'clim': replace the weather features over the forecast window by their day-of-year climatology, as in mlode_pipeline.py")
    a=ap.parse_args(); species=a.species; origin=inv.day(a.origin); H=a.horizon
    feat=P.load_features(); traps=P.eligible_traps(species,origin); print(species,"traps",len(traps),flush=True)
    Wm={t:P.weather_matrix(feat,t) for t in traps}
    if a.climate_scenario=="clim":
        Wm={t:P.climatological_weather(W,origin) for t,W in Wm.items()}
        print("climate scenario: weather features from day-of-year climatology from the origin onward",flush=True)
    rows=[]   # training/validation examples
    test=[]
    for t in traps:
        tt,cc=collections(t,species,inv.day("2017-01-01"),origin)
        if len(tt)<20: continue
        X,kd=P.build_X(feat,Wm[t],t,tt)
        if len(kd)==0: continue
        cmap=dict(zip(tt,cc)); clim,cmean=P.climatology(t,species,origin)
        for x,d in zip(X,kd):
            doy=2*np.pi*(d%365.25)/365.25
            # persistence features: sample a lead 1..H, take last observation at least `lead` days before d
            lead=float(rng.integers(1,H+1)); prev=tt[tt<=d-lead]
            if len(prev): lastc=cmap[prev[-1]]; lastanom=lastc-clim.get((int(prev[-1])//7)%52,cmean)
            else: lastc,lastanom=cmean,0.0
            rows.append(dict(trap=t,t=d,y=cmap[d],split="valid" if d>=origin-a.valid_days else "train",x=x,doy_s=np.sin(doy),doy_c=np.cos(doy),clim=clim.get((int(d)//7)%52,cmean),lead=lead,lastc=lastc,lastanom=lastanom))
        # test rows
        to,yo=collections(t,species,origin,origin+H)
        if len(to)<5: continue
        Xt,kt=P.build_X(feat,Wm[t],t,to)
        prev=tt[tt<origin]; lastc=cmap[prev[-1]] if len(prev) else cmean; lastanom=(lastc-clim.get((int(prev[-1])//7)%52,cmean)) if len(prev) else 0.0
        for x,d in zip(Xt,kt):
            doy=2*np.pi*(d%365.25)/365.25
            test.append(dict(trap=t,t=d,lead=d-origin,obs=dict(zip(to,yo))[d],x=x,doy_s=np.sin(doy),doy_c=np.cos(doy),clim=clim.get((int(d)//7)%52,cmean),const=cmean,lastc=lastc,lastanom=lastanom,
                             pers=max(clim.get((int(d)//7)%52,cmean)+lastanom*np.exp(-(d-origin)/21.0),0)))
    tr=pd.DataFrame(rows); te=pd.DataFrame(test); print("train rows",(tr.split=="train").sum(),"valid",(tr.split=="valid").sum(),"test",len(te),flush=True)
    def mat(df,variant):
        base=np.vstack(df.x.values); extra=[df.doy_s.values,df.doy_c.values]
        if variant=="persist": extra+= [df.lead.values,df.lastc.values,df.lastanom.values]
        return np.column_stack([base]+extra)
    out=te[["trap","t","lead","obs","clim","const","pers"]].copy(); summ=[]
    for variant in ["weather","persist"]:
        Xtr=mat(tr[tr.split=="train"],variant); ytr=tr[tr.split=="train"].y.values; Xva=mat(tr[tr.split=="valid"],variant); yva=tr[tr.split=="valid"].y.values
        m=CatBoostRegressor(iterations=3000,learning_rate=0.05,depth=6,loss_function="Poisson",random_seed=a.seed,verbose=False,thread_count=-1)
        m.fit(Xtr,ytr,eval_set=(Xva,yva),use_best_model=True)
        out[variant]=np.clip(m.predict(mat(te,variant)),0,None)
        summ.append(dict(variant=variant,best_iter=m.get_best_iteration(),valid_mae=float(np.mean(np.abs(m.predict(Xva)-yva))),valid_mae_clim=float(np.mean(np.abs(tr[tr.split=="valid"].clim.values-yva)))))
        print(variant,summ[-1],flush=True)
    tag=f"{species}_{a.origin}"+a.suffix; out.to_csv(f"direct_gbr_obs_{tag}.csv",index=False)
    def trap_scores(g,col):
        o=g.obs.values; p=g[col].values
        if len(o)<3 or o.mean()==0: return None
        r=np.sqrt(np.mean((p-o)**2)); rc=np.sqrt(np.mean((g.clim-o)**2)); rk=np.sqrt(np.mean((g.const-o)**2)); rp=np.sqrt(np.mean((g.pers-o)**2))
        return dict(rel_rmse=r/o.mean(),mae=np.mean(np.abs(p-o)),skill_clim=1-r/rc if rc>0 else np.nan,skill_const=1-r/rk if rk>0 else np.nan,skill_pers=1-r/rp if rp>0 else np.nan,spearman=spearmanr(o,p).correlation if len(set(o))>2 else np.nan,fc_over_obs=p.mean()/o.mean())
    def alert(g,col):
        o=g.obs.values; p=g[col].values; hit=((o>=10)&(p>=10)).sum(); miss=((o>=10)&(p<10)).sum(); fa=((o<10)&(p>=10)).sum(); cn=((o<10)&(p<10)).sum()
        return hit/max(hit+miss,1)-fa/max(fa+cn,1)
    out["bin"]=pd.cut(out.lead,[-1,30,90,180],labels=["1-30 d","31-90 d","91-180 d"])
    lines=[f"# Direct gradient boosting on counts (E2b): {species}, origin {a.origin}, {out.trap.nunique()} traps\n",
           "Same traps, window, observations (zeros included) and baselines as the MLODE Phase 0 comparison. Poisson CatBoost, 3000 it, lr 0.05, depth 6, early stopping on the last 122 d before the origin. 'weather' = site + weather lags + week-of-year; 'persist' adds last count before the origin, its anomaly and the lead. Damped persistence baseline: climatology + last anomaly decayed with 21 d.\n",
           "## ML step\n", pd.DataFrame(summ).round(4).to_markdown(index=False), "\n## 180-day window, medians over traps\n"]
    rows=[]
    for col in ["weather","persist","clim","pers","const"]:
        ts=pd.DataFrame([s for s in (trap_scores(g,col) for _,g in out.groupby("trap")) if s]); rows.append(dict(model=col,**ts.median().round(3).to_dict(),frac_pos_skill_clim=round((ts.skill_clim>0).mean(),3),peirce=round(alert(out,col),3)))
    lines.append(pd.DataFrame(rows).to_markdown(index=False)); lines.append("\n## By lead time (per-trap medians within bin; Peirce pooled)\n")
    rows=[]
    for b,gb in out.groupby("bin",observed=True):
        for col in ["weather","persist","clim","pers"]:
            ts=pd.DataFrame([s for s in (trap_scores(g,col) for _,g in gb.groupby("trap")) if s]); rows.append(dict(lead=b,model=col,traps=len(ts),rel_rmse=round(ts.rel_rmse.median(),3),skill_clim=round(ts.skill_clim.median(),3),skill_pers=round(ts.skill_pers.median(),3),spearman=round(ts.spearman.median(),3),peirce=round(alert(gb,col),3)))
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    open(f"direct_gbr_summary_{tag}.md","w").write("\n".join(lines)); print("saved",tag)
if __name__=="__main__": main()
