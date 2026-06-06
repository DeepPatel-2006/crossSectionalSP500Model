# Shared backtest plumbing: point-in-time universe, cost model, a reserved 2024+ holdout
# the book refuses to touch until the final run, and the net-of-cost book evaluator.
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0, "newcycle")
import featlab as fl
C, O, R, V, DV, idx, syms = fl.Cn, fl.On, fl.Rn, fl.Vn, fl.DVn, fl.idx, fl.syms
n, S = C.shape; rebs = fl.rebs; YR = np.asarray(fl.YR)
rm, rstd = fl.roll_mean, fl.roll_std
DEV_YEARS = list(range(2016, 2024))        # 2016-2023 used for development
HOLDOUT_YEARS = list(range(2024, 2027))     # 2024-2026 reserved, evaluated ONCE
def sd(a, b): return a / np.where(np.abs(b) > 1e-12, b, np.nan)
dlog = np.log(np.clip(C, 1e-6, None)) - np.log(np.clip(fl.shift(C, 1), 1e-6, None))
rv63 = rstd(dlog, 63)
dvol = pd.DataFrame(DV, index=idx).rolling(63, min_periods=50).median().to_numpy()
uni = np.isfinite(C) & (R >= 1.0) & (R < 5.0) & (dvol >= 5e5)   # PIT sub-$5 universe
U = {t: np.where(uni[t])[0] for t in rebs}
Cff = pd.DataFrame(C, index=idx).ffill(limit=26).to_numpy()
def fwd21(t):
    e = O[t + 1]; x = Cff[t + 21]; r_ = np.where(np.isfinite(e) & (e > 0) & np.isfinite(x) & (x > 0), x / np.where(e > 0, e, np.nan) - 1.0, np.nan)
    return np.where(r_ < -0.5, -1.0, r_)   # delisting-safe

def sample(year_lo=2012, year_hi=2026, min_names=60):
    """Return (groups[(t,s)], rawf[list], yrg[list]) for rebalances in [year_lo,year_hi]."""
    groups, rawf, yrg = [], [], []
    for i, t in enumerate(rebs):
        if not (year_lo <= int(YR[i]) <= year_hi): continue
        s = U[t]
        if len(s) < min_names: continue
        groups.append((t, s)); rawf.append(fwd21(t)[s]); yrg.append(int(YR[i]))
    return groups, rawf, yrg

def cost_perside(price, level="real"):
    """Per-side trading cost (fraction). 'real' = sub-$5 half-spread (~$0.01) + 10bps impact, floor 20bps cap 4%."""
    if isinstance(level, (int, float)): return float(level)
    hs = np.clip(0.01 / np.clip(price, 0.5, None), 0.002, 0.04)
    return hs + 0.001

def capw(iv, cap=0.06):
    w = iv / iv.sum()
    for _ in range(50):
        over = w > cap
        if not over.any(): break
        ex = (w[over] - cap).sum(); w[over] = cap; un = ~over
        if un.any(): w[un] += ex * w[un] / w[un].sum()
    return w / w.sum()

def book(score, groups, rawf, yrg, topn=30, weighting="invvol", perside="real",
         exclude=None, years=None, allow_holdout=False):
    # score is a list indexed by group -> array of per-name scores for that rebalance's universe.
    if years is None: years = DEV_YEARS
    if (not allow_holdout) and any(y in HOLDOUT_YEARS for y in years):
        raise RuntimeError("HOLDOUT is reserved. Pass allow_holdout=True only for the final one-shot evaluation.")
    rets, yrs, caps, prevw, drops = [], [], [], {}, 0
    for gi, (t, s) in enumerate(groups):
        if yrg[gi] not in years: continue
        sc = score[gi]
        if not np.isfinite(sc).any(): continue
        order = np.argsort(-np.where(np.isfinite(sc), sc, -9))
        keep = []
        for j in order:
            if exclude is not None and exclude[t, s[j]]: drops += 1; continue
            keep.append(j)
            if len(keep) >= topn: break
        keep = np.array(keep); names = s[keep]; pr = R[t, names]
        if weighting == "invvol": w = capw(1.0 / np.clip(rv63[t, names], 0.02, None))
        elif weighting == "score": w = capw(np.clip(sc[keep] - sc[keep].min() + 1e-6, 1e-6, None))
        else: w = np.full(len(keep), 1.0 / len(keep))   # equal
        f = np.nan_to_num(rawf[gi][keep], nan=0.0)
        ps = cost_perside(pr, perside)
        wd = {int(nm): wt for nm, wt in zip(names, w)}
        dturn = np.array([abs(wd.get(k, 0) - prevw.get(k, 0)) for k in set(wd) | set(prevw)])
        # cost: turnover weight * per-side (approx per-name cost at this rebalance's avg)
        gcost = float(np.sum(np.abs([wd.get(k, 0) - prevw.get(k, 0) for k in set(wd) | set(prevw)])) * np.mean(ps))
        rets.append(float(np.sum(w * f)) - gcost); yrs.append(yrg[gi]); prevw = wd
        caps.append(float(np.min(0.06 ** -1 * 0.01 * dvol[t, names] * 21)))  # ADV-based crude capacity proxy
    rets = np.array(rets); yrs = np.array(yrs)
    if len(rets) == 0: return None
    eq = np.cumprod(1 + rets); dd = eq / np.maximum.accumulate(eq) - 1
    return dict(total=eq[-1] - 1, ann=eq[-1] ** (12 / len(rets)) - 1, sharpe=rets.mean() / (rets.std() + 1e-9) * np.sqrt(12),
                maxDD=dd.min(), n=len(rets), drops=drops, by={y: float(np.prod(1 + rets[yrs == y]) - 1) for y in sorted(set(yrs))},
                cap=float(np.median(caps)) if caps else np.nan)
