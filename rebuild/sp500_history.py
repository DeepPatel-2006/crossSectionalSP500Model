# Rebuilds the final book (5-seed ensemble, K=5 factor-neutral) and writes the dated monthly
# return series to reports/sp500_history.json for the site. Same construction as sp500_holdout.py,
# plus an equal-weight universe return as a market reference.
import sys, warnings, json
import numpy as np, lightgbm as lgb
warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0, "rebuild"); sys.path.insert(0, "newcycle")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import best_model as BM
import featlab as fl
H = BM.H; F = BM.F; NAMES = BM.NAMES; P = BM.P; MCAP = BM.MCAP; R = H.R; dvol = H.dvol; rebs = H.rebs; n, S = H.n, H.S
YR = np.asarray(fl.YR); idx = fl.idx; beta = F["beta_126"]; rv = F["rv_63"]; ret = fl.Cn / fl.shift(fl.Cn, 1) - 1.0; PS = 0.0005
K = 5
def fwd_h(t, h):
    if t + h >= n: return np.full(S, np.nan)
    e = H.O[t + 1]; x = H.Cff[t + h]; r = np.where(np.isfinite(e) & (e > 0) & np.isfinite(x) & (x > 0), x / np.where(e > 0, e, np.nan) - 1.0, np.nan)
    return np.where(r < -0.5, -1.0, r)
SP = np.zeros((n, S), bool)
for t in rebs:
    mc = MCAP[t]; v = np.where(np.isfinite(mc) & (R[t] > 1) & (dvol[t] >= 5e6))[0]
    if len(v) > 500: SP[t, v[np.argsort(-mc[v])[:500]]] = True
    else: SP[t, v] = True
groups, rawf, f63, yrg = [], [], [], []
for i, t in enumerate(rebs):
    if not (2012 <= int(YR[i]) <= 2026): continue
    s = np.where(SP[t])[0]
    if len(s) < 100 or t + 22 >= n: continue
    groups.append((t, s)); rawf.append(H.fwd21(t)[s]); f63.append(fwd_h(t, 63)[s]); yrg.append(int(YR[i]))
X = np.vstack([np.column_stack([F[k][t, s] for k in NAMES]) for (t, s) in groups])
yr = np.concatenate([np.full(len(s), yrg[gi]) for gi, (t, s) in enumerate(groups)]); grp = np.concatenate([np.full(len(g[1]), gi) for gi, g in enumerate(groups)])
yk = np.full(len(yr), np.nan)
for gi in range(len(groups)):
    m = grp == gi; y = f63[gi] - np.nanmean(f63[gi]); o = np.argsort(np.argsort(np.nan_to_num(y, nan=-9))); yk[m] = (o + 1) / m.sum()
def rank01g(p):
    r = np.full(len(p), np.nan)
    for gi in range(len(groups)):
        m = grp == gi; v = p[m]; ok = np.isfinite(v)
        if ok.sum() > 5: rr = np.full(len(v), np.nan); rr[ok] = np.argsort(np.argsort(v[ok])) / ok.sum(); r[m] = rr
    return r
print("training 5-seed ensemble through 2026...", flush=True)
ens = np.zeros(len(yr))
for ki in range(5):
    Pk = dict(P, bagging_fraction=0.8, bagging_freq=1, seed=ki, bagging_seed=ki, feature_fraction_seed=ki + 7); p = np.full(len(yr), np.nan)
    for Y in range(2016, 2027):
        tr = (yr < Y) & np.isfinite(yk); te = yr == Y
        if tr.sum() < 3000 or te.sum() == 0: continue
        m = lgb.train(Pk, lgb.Dataset(X[tr], yk[tr]), num_boost_round=300); p[te] = m.predict(X[te])
    ens += np.nan_to_num(rank01g(p), nan=0.5)
    print(f"  seed {ki} done", flush=True)
pe = ens / 5.0; scE = [pe[grp == gi] for gi in range(len(groups))]
def facneut(w, s, t):
    if t < 130: return w
    Rw = np.nan_to_num(ret[t - 125:t + 1, s]); Rw = Rw - Rw.mean(0)
    try:
        _, _, Vt = np.linalg.svd(Rw, full_matrices=False); Vk = Vt[:K].T; return w - Vk @ (Vk.T @ w)
    except Exception:
        return w
rows, prevw = [], {}
for gi in range(len(groups)):
    if yrg[gi] < 2016: continue
    t, s = groups[gi]; sci = scE[gi]; m = np.isfinite(sci)
    if m.sum() < 50: continue
    ivw = 1.0 / np.clip(rv[t, s], 0.02, None); order = np.argsort(-np.where(m, sci, -9)); ln = order[:30]; sh = order[-30:]
    w = np.zeros(len(s)); w[ln] = ivw[ln] / ivw[ln].sum(); w[sh] = -ivw[sh] / ivw[sh].sum(); w = facneut(w, s, t)
    f = np.nan_to_num(rawf[gi], nan=0.0); wd = {int(s[j]): w[j] for j in np.where(np.abs(w) > 1e-5)[0]}
    dturn = sum(abs(wd.get(k, 0) - prevw.get(k, 0)) for k in set(wd) | set(prevw))
    net = float(np.nansum(w * f)) - dturn * PS; prevw = wd
    rows.append({"d": str(idx[t].date()), "model": round(net, 6), "mkt": round(float(np.nanmean(rawf[gi])), 6),
                 "holdout": yrg[gi] >= 2024})
def stats(rs):
    r = np.array(rs); eq = np.cumprod(1 + r); dd = eq / np.maximum.accumulate(eq) - 1
    return {"ann": round(float(eq[-1] ** (12 / len(r)) - 1), 4), "sharpe": round(float(r.mean() / (r.std() + 1e-9) * np.sqrt(12)), 2),
            "maxdd": round(float(dd.min()), 4), "months": len(r)}
dev = [r["model"] for r in rows if not r["holdout"]]; hold = [r["model"] for r in rows if r["holdout"]]
out = {"built": str(idx[-1].date()), "config": "5-seed LightGBM, 63-day signal, top/bot-30 inverse-vol, K=5 factor-neutral, 5 bps/side",
       "dev": stats(dev), "holdout": stats(hold), "series": rows}
import os; os.makedirs("reports", exist_ok=True)
open("reports/sp500_history.json", "w").write(json.dumps(out, indent=1))
print(f"wrote reports/sp500_history.json ({len(rows)} months)")
print("dev", out["dev"]); print("holdout", out["holdout"])
