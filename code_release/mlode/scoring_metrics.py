"""Paired statistical comparison of two forecast models on the same traps and window, from per-observation files.
Usage: python scoring_metrics.py A.csv colA B.csv colB [--label ...]. Per trap: window RMSE, skill vs climatology.
Reports median paired difference with a 95% bootstrap CI, Wilcoxon signed-rank p, share of traps where A wins with CI,
a within-station bootstrap, and the pooled alert (Peirce) scores with trap-bootstrap CIs."""
import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
def load(path, col, design=None, D=None):
    df=pd.read_csv(path)
    if design is not None: df=df[(df.design==design)&(df.D==D)]
    return df[["trap","t","obs","clim",col]].rename(columns={col:"fc"})
def per_trap(df):
    g=df.groupby("trap"); r=g.apply(lambda x: pd.Series(dict(n=len(x),rmse=np.sqrt(np.mean((x.fc-x.obs)**2)),rmse_clim=np.sqrt(np.mean((x.clim-x.obs)**2)),obs_mean=x.obs.mean())))
    r=r[r.n>=5]; r["skill"]=1-r.rmse/r.rmse_clim; return r
def peirce(df,col="fc",thr=10):
    o=df.obs.values; p=df[col].values; hit=((o>=thr)&(p>=thr)).sum(); miss=((o>=thr)&(p<thr)).sum(); fa=((o<thr)&(p>=thr)).sum(); cn=((o<thr)&(p<thr)).sum()
    return hit/max(hit+miss,1)-fa/max(fa+cn,1)
def compare(A,B,nameA,nameB,station=None,nboot=2000,seed=0):
    rng=np.random.default_rng(seed)
    a=per_trap(A); b=per_trap(B); traps=a.index.intersection(b.index); a=a.loc[traps]; b=b.loc[traps]
    d=(a.skill-b.skill).values; win=(a.rmse<b.rmse).values.astype(float)
    boots=np.array([np.median(d[rng.integers(0,len(d),len(d))]) for _ in range(nboot)]); wb=np.array([win[rng.integers(0,len(d),len(d))].mean() for _ in range(nboot)])
    p=wilcoxon(d).pvalue
    out=dict(traps=len(traps),median_skill_A=float(a.skill.median()),median_skill_B=float(b.skill.median()),median_diff=float(np.median(d)),ci=(float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))),
             wilcoxon_p=float(p),share_A_wins=float(win.mean()),share_ci=(float(np.percentile(wb,2.5)),float(np.percentile(wb,97.5))))
    if station is not None:
        st=station.reindex(traps).values; groups={s:np.where(st==s)[0] for s in set(st)}
        sb=[]
        for _ in range(nboot):
            idx=np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups.values()]); sb.append(np.median(d[idx]))
        out["ci_within_station"]=(float(np.percentile(sb,2.5)),float(np.percentile(sb,97.5)))
    # alert scores with trap bootstrap
    def pb(df):
        tr=df.trap.unique(); vals=[]
        for _ in range(500):
            samp=rng.choice(tr,len(tr),replace=True); vals.append(peirce(pd.concat([df[df.trap==t] for t in samp])))
        return float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))
    out["peirce_A"]=(peirce(A),pb(A)); out["peirce_B"]=(peirce(B),pb(B)); out["peirce_clim"]=peirce(A,"clim")
    print(f"{nameA} vs {nameB}: traps {out['traps']}; median skill {out['median_skill_A']:.3f} vs {out['median_skill_B']:.3f}; median paired diff {out['median_diff']:+.3f} (95% CI {out['ci'][0]:+.3f}, {out['ci'][1]:+.3f})"+(f"; within-station CI ({out['ci_within_station'][0]:+.3f}, {out['ci_within_station'][1]:+.3f})" if 'ci_within_station' in out else "")+f"; Wilcoxon p {out['wilcoxon_p']:.3g}; share {nameA} wins {out['share_A_wins']:.2f} (CI {out['share_ci'][0]:.2f}, {out['share_ci'][1]:.2f})")
    print(f"   Peirce: {nameA} {out['peirce_A'][0]:.3f} (CI {out['peirce_A'][1][0]:.3f}, {out['peirce_A'][1][1]:.3f}); {nameB} {out['peirce_B'][0]:.3f} (CI {out['peirce_B'][1][0]:.3f}, {out['peirce_B'][1][1]:.3f}); climatology {out['peirce_clim']:.3f}")
    return out
if __name__=="__main__":
    import inverse_step_2stage as inv
    st=pd.Series(inv.trap_station())
    for sp in ["aedes","culex"]:
        print("=====",sp,"June 2020 origin")
        direct=load(f"direct_gbr_obs_{sp}_2020-06-01.csv","weather"); mlode=load(f"mlode_obs_{sp}_2020-06-01.csv","fc",design="F1",D="D1" if sp=="aedes" else "D0"); orig=load(f"mlode_obs_{sp}_2020-06-01.csv","fc",design="F0",D="D0")
        compare(direct,mlode,"direct GBR","MLODE F1",station=st); compare(mlode,orig,"MLODE F1","MLODE original",station=st)
        clim=mlode.assign(fc=mlode.clim); compare(mlode,clim,"MLODE F1","climatology",station=st); compare(direct,clim,"direct GBR","climatology",station=st)
