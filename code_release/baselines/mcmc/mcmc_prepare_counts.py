"""Build the Stan data files for the E2c MCMC fits: one JSON per trap x origin, all collections (zeros included)
from 1 Jan 2017 to the day before the origin, times in days since 1 Aug 2016 (t0 = 0, as in the original),
station harmonic coefficients from coefficients.json using the temperature-file trap-to-station mapping."""
import os, sys, json, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mlode"))
import inverse_step_2stage as inv
SETS = {"aedes": dict(origin="2021-06-01", horizon=153, traps=["MB55B","MB51","MB53","CW77","CW57","CW5","CW6","CW7","CW16","CW55","CW20","R1","CW51","LR1"]),
        "culex": dict(origin="2021-01-01", horizon=180, traps=["MB51","MB53","MB55B","CW75","CW11","CW15","CW49","CW71","CW57","CW56","CW1","CW2"])}
coef = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coefficients.json")))
st = inv.trap_station(); raw = inv.raw_counts(); os.makedirs("data", exist_ok=True); manifest = []
for sp, cfg in SETS.items():
    o = inv.day(cfg["origin"]); col = inv.SPECIES[sp]["col"]
    for t in cfg["traps"]:
        d = raw[raw.TRAPID == t].assign(c=pd.to_numeric(raw.loc[raw.TRAPID == t, col], errors="coerce")).dropna(subset=["c", "t"]).drop_duplicates("t").sort_values("t")
        tr = d[(d.t >= inv.day("2017-01-01")) & (d.t < o)]; te = d[(d.t >= o) & (d.t < o + cfg["horizon"])]
        station = st[t]; c = coef[sp][station]
        data = dict(n_obs=len(tr), yy=tr.c.astype(int).tolist(), t0=0.0, ts=tr.t.astype(float).tolist(), ovi=c["ovi"], pea=c["pea"], dev=c["dev"], mu=c["mu"], rain=c["rain"])
        name = f"{sp}_{t}_{cfg['origin']}"; json.dump(data, open(f"data/{name}.json", "w"))
        te[["t", "c"]].to_csv(f"data/{name}_window_obs.csv", index=False)
        manifest.append(dict(name=name, species=sp, trap=t, station=station, origin=cfg["origin"], horizon=cfg["horizon"], n_train=len(tr), n_zero_train=int((tr.c == 0).sum()), n_window=len(te)))
pd.DataFrame(manifest).to_csv("manifest.csv", index=False); print(pd.DataFrame(manifest).to_string(index=False))
