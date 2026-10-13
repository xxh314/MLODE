"""Data for the weather-driven MCMC variant: for every Ae. aegypti fit at the June 2021 origin (batches 1, 2, q_aedes), write
data/{name}_obs.json = the original counts/times plus daily tables (t = 0 .. origin + horizon + 1) of the four thermal rates
(inverse.rates, 28-day trailing station temperature) and the scaled 7-day cumulative rainfall at the trap (mean 1 over
1 Jan 2017 .. origin - 1). Usage: python mcmc_prepare_weather.py [species origin]"""
import os, sys, json, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,"..","..","mlode")); import inverse_step_2stage as inv, mlode_pipeline as P
sp=sys.argv[1] if len(sys.argv)>1 else "aedes"; origin=sys.argv[2] if len(sys.argv)>2 else "2021-06-01"; CLIM="--clim" in sys.argv; SUF="_obsclim" if CLIM else "_obs"
man=pd.read_csv(os.path.join(HERE,"manifest.csv")); sel=man[(man.species==sp)&(man.origin==origin)&(man.batch.astype(str).isin(["1","2","q_aedes","rest_aedes","rest_culex"]))]
feat=P.load_features(); o=inv.day(origin); names=[]
for _,r in sel.iterrows():
    base=json.load(open(os.path.join(HERE,f"data/{r['name']}.json"))); T=int(o+r.horizon+2); R=inv.rates(r.station,sp); tt=np.arange(T,dtype=float); tt=np.minimum(tt,R.tmax)
    W=P.weather_matrix(feat,r.trap); p=W[:T,0]; p=np.nan_to_num(p,nan=0.0); c7=np.convolve(p,np.ones(7),mode="full")[:T]
    m=c7[int(inv.day("2017-01-01")):int(o)].mean(); rain=(c7/m if m>0 else c7)
    tabs=[np.asarray(x,dtype=float) for x in (R.b(tt),R.pEA(tt),R.d(tt),R.mu(tt),rain)]
    if CLIM:
        # E1 climatology scenario: every daily value from the origin onward becomes its day-of-year mean over the training
        # years (15-day circular smoothing). Values before the origin are untouched, so the posterior is unchanged and reused.
        c=int(o); doy=(np.arange(T)+inv.DAY0.timetuple().tm_yday-1)%365
        def climify(v):
            tab=pd.Series(v[:c]).groupby(doy[:c]).mean().reindex(range(365)).interpolate(limit_direction="both")
            tab=pd.Series(np.convolve(np.r_[tab.values[-7:],tab.values,tab.values[:7]],np.ones(15)/15,mode="valid"),index=range(365))
            out=v.copy(); out[c:]=tab.reindex(doy[c:]).values; return out
        tabs=[climify(v) for v in tabs]
    data=dict(n_obs=base["n_obs"],yy=base["yy"],t0=0.0,ts=base["ts"],T=T,ovi=[float(x) for x in tabs[0]],pea=[float(x) for x in tabs[1]],dev=[float(x) for x in tabs[2]],mu=[float(x) for x in tabs[3]],rain=[float(x) for x in tabs[4]])
    json.dump(data,open(os.path.join(HERE,"data/"+r["name"]+SUF+".json"),"w")); names.append(r["name"])
open(os.path.join(HERE,"manifest_obs_"+sp+("_clim" if CLIM else "")+".txt"),"w").write("\n".join(names)+"\n"); print("prepared",len(names),"traps; rain index mean over training 1.0; example rain max",round(max(rain),2),"rates at t=1000: ovi %.2f pea %.2f dev %.3f mu %.4f"%(data["ovi"][1000],data["pea"][1000],data["dev"][1000],data["mu"][1000]))
