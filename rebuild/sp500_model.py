# S&P-500 book. Ranks a top-500-by-cap universe with the shared GBM (63-day target) and forms a
# long-only top-50 and a market-neutral long/short. Costs at 5 bps per side; 2024+ is left untouched.
# Market cap stands in for exact index membership, which we don't have point-in-time.
import sys, warnings
import numpy as np, lightgbm as lgb
warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0, "rebuild"); sys.path.insert(0, "newcycle")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import best_model as BM
H = BM.H; F = BM.F; NAMES = BM.NAMES; P = BM.P; MCAP = BM.MCAP; R = H.R; dvol = H.dvol; rebs = H.rebs; YR = np.asarray(BM.YR) if hasattr(BM, "YR") else np.asarray([d.year for d in H.idx[rebs]])
import featlab as fl
YR = np.asarray(fl.YR); n, S = H.n, H.S
def fwd_h(t, h):
    if t + h >= n: return np.full(S, np.nan)
    e = H.O[t + 1]; x = H.Cff[t + h]; r = np.where(np.isfinite(e) & (e > 0) & np.isfinite(x) & (x > 0), x / np.where(e > 0, e, np.nan) - 1.0, np.nan)
    return np.where(r < -0.5, -1.0, r)
# S&P-500 proxy: top-500 by market cap, liquid
SP = np.zeros((n, S), bool)
for t in rebs:
    mc = MCAP[t].copy(); valid = np.where(np.isfinite(mc) & (R[t] > 1) & (dvol[t] >= 5e6))[0]
    if len(valid) > 500: SP[t, valid[np.argsort(-mc[valid])[:500]]] = True
    else: SP[t, valid] = True
groups, rawf, f63, yrg = [], [], [], []
for i, t in enumerate(rebs):
    if not (2012 <= int(YR[i]) <= 2023): continue
    s = np.where(SP[t])[0]
    if len(s) < 100: continue
    groups.append((t, s)); rawf.append(H.fwd21(t)[s]); f63.append(fwd_h(t, 63)[s]); yrg.append(int(YR[i]))
print(f"S&P-500 proxy: {len(groups)} rebalances, ~{int(np.mean([len(s) for _, s in groups]))} names/date")
X = np.vstack([np.column_stack([F[k][t, s] for k in NAMES]) for (t, s) in groups])
yr = np.concatenate([np.full(len(s), yrg[gi]) for gi, (t, s) in enumerate(groups)]); grp = np.concatenate([np.full(len(g[1]), gi) for gi, g in enumerate(groups)])
yk = np.full(len(yr), np.nan)
for gi in range(len(groups)):
    m = grp == gi; y = f63[gi] - np.nanmean(f63[gi]); o = np.argsort(np.argsort(np.nan_to_num(y, nan=-9))); yk[m] = (o + 1) / m.sum()
pred = np.full(len(yr), np.nan)
for Y in H.DEV_YEARS:
    tr = yr < Y; te = yr == Y
    if tr.sum() < 3000 or te.sum() == 0: continue
    mdl = lgb.train(P, lgb.Dataset(X[tr], yk[tr]), num_boost_round=300); pred[te] = mdl.predict(X[te])
sc = [pred[grp == gi] for gi in range(len(groups))]
ics = []
for gi in range(len(groups)):
    if yrg[gi] < 2016: continue
    yv = rawf[gi] - np.nanmean(rawf[gi]); ok = np.isfinite(sc[gi]) & np.isfinite(yv)
    if ok.sum() < 50: continue
    a = np.argsort(np.argsort(sc[gi][ok])).astype(float); b = np.argsort(np.argsort(yv[ok])).astype(float); a -= a.mean(); b -= b.mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
    if d > 0: ics.append((a @ b) / d)
rv = BM.F["rv_63"]; beta = BM.F["beta_126"]; PS = 0.0005   # 5 bps/side large-cap
def iv(t, names): return H.capw(1.0 / np.clip(rv[t, names], 0.02, None))
def book_long(topn):
    rets, prevw, yrs = [], {}, []
    for gi in range(len(groups)):
        if yrg[gi] < 2016: continue
        t, s = groups[gi]; sci = sc[gi]; order = np.argsort(-np.where(np.isfinite(sci), sci, -9))[:topn]; names = s[order]
        w = iv(t, names); f = np.nan_to_num(rawf[gi][order], nan=0.0); wd = {int(a): b for a, b in zip(names, w)}
        dturn = sum(abs(wd.get(k, 0) - prevw.get(k, 0)) for k in set(wd) | set(prevw))
        rets.append(float(np.sum(w * f)) - dturn * PS); prevw = wd; yrs.append(yrg[gi])
    rets = np.array(rets); eq = np.cumprod(1 + rets); dd = eq / np.maximum.accumulate(eq) - 1
    return eq[-1] ** (12 / len(rets)) - 1, rets.mean() / (rets.std() + 1e-9) * np.sqrt(12), dd.min()
def book_ls(nn):
    rets, pL, pS, yrs, bnet = [], {}, {}, [], []
    for gi in range(len(groups)):
        if yrg[gi] < 2016: continue
        t, s = groups[gi]; sci = sc[gi]; order = np.argsort(-np.where(np.isfinite(sci), sci, -9))
        ln = s[order[:nn]]; sn = s[order[-nn:]]; wl = iv(t, ln); ws = iv(t, sn)
        fl_ = np.nan_to_num(rawf[gi][order[:nn]], nan=0.0); fs = np.nan_to_num(rawf[gi][order[-nn:]], nan=0.0)
        wdL = {int(a): b for a, b in zip(ln, wl)}; wdS = {int(a): b for a, b in zip(sn, ws)}
        cL = sum(abs(wdL.get(k, 0) - pL.get(k, 0)) for k in set(wdL) | set(pL)) * PS
        cS = sum(abs(wdS.get(k, 0) - pS.get(k, 0)) for k in set(wdS) | set(pS)) * PS
        rets.append(float(np.sum(wl * fl_)) - float(np.sum(ws * fs)) - cL - cS)
        bnet.append(float(np.nansum(wl * np.nan_to_num(beta[t, ln]))) - float(np.nansum(ws * np.nan_to_num(beta[t, sn]))))
        pL, pS = wdL, wdS; yrs.append(yrg[gi])
    rets = np.array(rets); eq = np.cumprod(1 + rets); dd = eq / np.maximum.accumulate(eq) - 1
    return eq[-1] ** (12 / len(rets)) - 1, rets.mean() / (rets.std() + 1e-9) * np.sqrt(12), dd.min(), float(np.mean(bnet))
IC = float(np.mean(ics))
if __name__ == "__main__":
    print("S&P-500 book - 49 features, 63-day target, 2016-2023")
    print(f"  IC {IC:+.4f}")
    a, sh, md = book_long(50); print(f"  long-only top-50      net ann {a:+6.1%}  Sharpe {sh:.2f}  maxDD {md:+5.0%}")
    for nn in (30, 50):
        a, sh, md, bn = book_ls(nn); print(f"  long/short top/bot{nn:<3d} net ann {a:+6.1%}  Sharpe {sh:.2f}  maxDD {md:+5.0%}  (net beta {bn:+.2f})")
