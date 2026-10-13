"""Fit mcmc_2stage_ode.stan for one trap/origin. Usage: python mcmc_fit_trap.py data.json outdir [chains] [warmup] [sampling] [thin]"""
import sys, json, time, os, pandas as pd, cmdstanpy
data_path, outdir = sys.argv[1], sys.argv[2]; chains=int(sys.argv[3]) if len(sys.argv)>3 else 4; warm=int(sys.argv[4]) if len(sys.argv)>4 else 2000; samp=int(sys.argv[5]) if len(sys.argv)>5 else 2000; thin=int(sys.argv[6]) if len(sys.argv)>6 else 2
os.makedirs(outdir, exist_ok=True); data=json.load(open(data_path))
m=cmdstanpy.CmdStanModel(stan_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("STAN_FILE","mcmc_2stage_ode.stan")))
t0=time.time(); fit=m.sample(data=data,chains=chains,parallel_chains=chains,iter_warmup=warm,iter_sampling=samp,thin=thin,seed=20260916,show_progress=False,show_console=False,output_dir=outdir)
el=time.time()-t0
draws=fit.draws_pd(); draws.to_csv(os.path.join(outdir,"draws.csv"),index=False)
summ=fit.summary(); summ.to_csv(os.path.join(outdir,"summary.csv"))
diag=fit.diagnose(); open(os.path.join(outdir,"diagnose.txt"),"w").write(diag)
json.dump(dict(elapsed_s=el,chains=chains,warmup=warm,sampling=samp,thin=thin,n_obs=data["n_obs"],divergences=int(draws["divergent__"].sum()) if "divergent__" in draws else None),open(os.path.join(outdir,"timing.json"),"w"))
print("done",outdir,"elapsed s",round(el,1)); print(summ.loc[[p for p in ["J0","A0","alpha","q"] if p in summ.index],["Mean","StdDev","5%","95%","R_hat","ESS_bulk"]].round(3))
