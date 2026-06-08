# Out-of-sample check for the final market-neutral model (5-seed ensemble, 63-day target, top/bot-30,
# beta + statistical-factor neutralized). Trains walk-forward through 2026, then compares the net book on
# the development years against the 2024-2026 holdout across a few neutralization strengths (K).
import sys, warnings
import numpy as np, lightgbm as lgb
warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0, "rebuild"); sys.path.insert(0, "newcycle")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import best_model as BM
import featlab as fl
H = BM.H; F = BM.F; NAMES = BM.NAMES; P = BM.P; MCAP = BM.MCAP; R = H.R; dvol = H.dvol; rebs = H.rebs; n, S = H.n, H.S
YR = np.asarray(fl.YR); beta = F["beta_126"]; rv = F["rv_63"]; ret = fl.Cn / fl.shift(fl.Cn, 1) - 1.0; PS = 0.0005
DEVY = list(range(2016, 2024)); HOLD = [2024, 2025, 2026]
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
print("training 5-seed ensemble through 2026 (predict DEV+HOLDOUT)...", flush=True)
ens = np.zeros(len(yr))
for ki in range(5):
    Pk = dict(P, bagging_fraction=0.8, bagging_freq=1, seed=ki, bagging_seed=ki, feature_fraction_seed=ki + 7); p = np.full(len(yr), np.nan)
    for Y in range(2016, 2027):
        tr = (yr < Y) & np.isfinite(yk); te = yr == Y
        if tr.sum() < 3000 or te.sum() == 0: continue
        m = lgb.train(Pk, lgb.Dataset(X[tr], yk[tr]), num_boost_round=300); p[te] = m.predict(X[te])
    ens += np.nan_to_num(rank01g(p), nan=0.5)
pe = ens / 5.0; scE = [pe[grp == gi] for gi in range(len(groups))]
def facneut(w, s, t, K):
    if K <= 0:
        bb = np.nan_to_num(beta[t, s]); bb = bb - bb.mean(); return w - ((w @ bb) / (bb @ bb + 1e-9)) * bb
    if t < 130: return w
    Rw = np.nan_to_num(ret[t - 125:t + 1, s]); Rw = Rw - Rw.mean(0)
    try:
        _, _, Vt = np.linalg.svd(Rw, full_matrices=False); Vk = Vt[:K].T; return w - Vk @ (Vk.T @ w)
    except Exception:
        return w
def book(K, years):
    rets, prevw = [], {}
    for gi in range(len(groups)):
        if yrg[gi] not in years: continue
        t, s = groups[gi]; sci = scE[gi]; m = np.isfinite(sci)
        if m.sum() < 50: continue
        ivw = 1.0 / np.clip(rv[t, s], 0.02, None); order = np.argsort(-np.where(m, sci, -9)); ln = order[:30]; sh = order[-30:]
        w = np.zeros(len(s)); w[ln] = ivw[ln] / ivw[ln].sum(); w[sh] = -ivw[sh] / ivw[sh].sum(); w = facneut(w, s, t, K)
        f = np.nan_to_num(rawf[gi], nan=0.0); wd = {int(s[j]): w[j] for j in np.where(np.abs(w) > 1e-5)[0]}
        dturn = sum(abs(wd.get(k, 0) - prevw.get(k, 0)) for k in set(wd) | set(prevw))
        rets.append(float(np.nansum(w * f)) - dturn * PS); prevw = wd
    rets = np.array(rets); eq = np.cumprod(1 + rets); dd = eq / np.maximum.accumulate(eq) - 1
    return eq[-1] ** (12 / len(rets)) - 1, rets.mean() / (rets.std() + 1e-9) * np.sqrt(12), dd.min()
print("S&P-500 market-neutral - development vs 2024-2026 holdout")
print(f"  {'neutralize':16s} | {'dev ann':>8s} {'dev Sh':>7s} {'dev DD':>7s} | {'hold ann':>9s} {'hold Sh':>8s} {'hold DD':>8s}")
for K, lbl in [(1, "K=1 (return)"), (5, "K=5 (balanced)"), (10, "K=10 (Sharpe)")]:
    da, dsh, ddd = book(K, DEVY); ha, hsh, hdd = book(K, HOLD)
    print(f"  {lbl:16s} | {da:+8.1%} {dsh:7.2f} {ddd:+7.0%} | {ha:+9.1%} {hsh:8.2f} {hdd:+8.0%}")