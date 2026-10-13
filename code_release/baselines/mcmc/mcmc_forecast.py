"""Forward simulation for the weather-driven variant: tables from data/{name}_obs.json, linear interpolation, 500 draws,
same outputs as predict.py (forecasts/{name}_obs.csv, design = MCMC_obs). Usage: python mcmc_forecast.py <name> [ndraw]"""
import os, sys, json, numpy as np, pandas as pd
from scipy.integrate import odeint
from scipy.stats import poisson
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,"..","..","mlode"))
import inverse_step_2stage as inv, mlode_pipeline as P
name=sys.argv[1]; ndraw=int(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2].isdigit() else 500; variant=(sys.argv[sys.argv.index("--variant")+1] if "--variant" in sys.argv else "obs")
man=pd.read_csv(os.path.join(HERE,"manifest.csv")).set_index("name").loc[name]; sp,trap,origin,H=man.species,man.trap,man.origin,int(man.horizon); o=inv.day(origin)
dsuf=("_obsclim" if variant=="obsclim" else "_obs")
data=json.load(open(os.path.join(HERE,f"data/{name}{dsuf}.json"))); tab={k:np.array(data[k]) for k in ["ovi","pea","dev","mu","rain"]}; T=data["T"]
fitvar=("obs" if variant=="obsclim" else variant)   # obsclim reuses the obs posterior: the fit sees only pre-origin data
draws=pd.read_csv(os.path.join(HERE,f"runs/{name}_{fitvar}/draws.csv")); sub=draws.sample(n=min(ndraw,len(draws)),random_state=0) if len(draws)>ndraw else draws
time=np.arange(0,o+H+1,dtype=float); K=1000.0; grid=np.arange(T,dtype=float)
def at(k,t): return np.interp(t,grid,tab[k])
def rhs(x,tt):
    J,A=x; return [A*(1-J/(K+K*alpha*at("rain",tt)))*at("ovi",tt)-at("dev",tt)*J, 0.5*at("dev",tt)*at("pea",tt)*J-at("mu",tt)*A]
lam=[]
for _,r in sub.iterrows():
    alpha=float(r["alpha"]) if "alpha" in r else 0.0; sol=odeint(rhs,(r["J0"]*1000,r["A0"]*1000),time); lam.append(np.clip(sol[:,1]*r["q"]/100,0,None))
lam=np.array(lam); fc_mean=lam.mean(0)
obs=pd.read_csv(os.path.join(HERE,f"data/{name}_window_obs.csv")); to=obs.t.values.astype(float); yo=obs.c.values.astype(float)
idx=np.clip(np.round(to).astype(int),0,len(time)-1); fc=fc_mean[idx]
lo=np.array([np.percentile(poisson.rvs(lam[:,i],random_state=1),5) for i in idx]); hi=np.array([np.percentile(poisson.rvs(lam[:,i],random_state=1),95) for i in idx])
clim,cmean=P.climatology(trap,sp,o); yc=np.array([clim.get((int(x)//7)%52,cmean) for x in to])
raw=inv.raw_counts(); col=inv.SPECIES[sp]["col"]
pre=raw[(raw.TRAPID==trap)&(raw.t<o)&(raw.t>=o-21)].assign(c=pd.to_numeric(raw.loc[raw.TRAPID==trap,col],errors="coerce")).dropna(subset=["c","t"])
anom=(float(pre.loc[pre.t==pre.t.max(),"c"].mean())-float(clim.get((int(pre.t.max())//7)%52,cmean))) if len(pre) else 0.0
pers=np.clip(yc+anom*np.exp(-(to-o)/21.0),0,None)
out=pd.DataFrame(dict(design="MCMC_"+variant,D="D0",trap=trap,t=to,lead=to-o,obs=yo,fc=fc,clim=yc,const=cmean,pers=pers,lo=lo,hi=hi))
out.to_csv(os.path.join(HERE,f"forecasts/{name}_{variant}.csv"),index=False)
cov=float(((yo>=lo)&(yo<=hi)).mean()); rmse=float(np.sqrt(np.mean((fc-yo)**2))); rc=float(np.sqrt(np.mean((yc-yo)**2)))
print(f"{name}_{variant}: n_window {len(yo)}, rel RMSE {rmse/max(yo.mean(),1e-9):.3f}, skill vs clim {1-rmse/rc:+.3f}, 90% interval coverage {cov:.2f}, draws used {len(sub)}")
