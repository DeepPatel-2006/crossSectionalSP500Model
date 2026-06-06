# Feature construction and the gradient-boosted ranker. Builds the cross-sectional feature
# matrices (volatility, momentum, low-risk, fundamentals, path-shape, peer-relative) and a
# shallow LightGBM model; importing exposes the features, running backtests the top-30 book.
import sys, warnings, importlib.util
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0, "rebuild"); sys.path.insert(0, "newcycle")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import harness as H
import featlab as fl
s2 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("s2", "newcycle/s2_build_features.py"))
importlib.util.spec_from_file_location("s2", "newcycle/s2_build_features.py").loader.exec_module(s2)
C, O, V, R, idx, syms, rebs, YR = fl.Cn, fl.On, fl.Vn, fl.Rn, fl.idx, fl.syms, fl.rebs, np.asarray(fl.YR); n, S = C.shape
rm, rstd = fl.roll_mean, fl.roll_std; rext = lambda a, w, k: s2.roll_extreme(a, w, k); dvol = H.dvol
def sd(a, b): return a / np.where(np.abs(b) > 1e-12, b, np.nan)
ret = C / fl.shift(C, 1) - 1.0; dlog = np.log(np.clip(C, 1e-6, None)) - np.log(np.clip(fl.shift(C, 1), 1e-6, None))
mkt = np.array([np.nanmean(ret[t, (R[t] >= 1) & (R[t] < 50)]) if np.isfinite(ret[t]).any() else np.nan for t in range(n)])[:, None]
beta = (rm(ret * mkt, 126) - rm(ret, 126) * rm(mkt, 126)) / np.clip(rm(mkt * mkt, 126) - rm(mkt, 126) ** 2, 1e-12, None)
m1 = rm(ret, 63); m2 = rm(ret * ret, 63) - m1 * m1; m3 = rm(ret ** 3, 63) - 3 * m1 * rm(ret * ret, 63) + 2 * m1 ** 3
rv63 = rstd(dlog, 63); dist252 = sd(C, rext(C, 252, "max")) - 1; ret126 = sd(fl.shift(C, 21), fl.shift(C, 126)) - 1
downsev = -rm(np.minimum(ret, 0.0), 63); gapf = rm((np.abs(sd(O, fl.shift(C, 1)) - 1) > 0.03).astype(float), 63)
ddf = rm((C < 0.9 * rext(C, 63, "max")).astype(float), 63); bounce = sd(C, rext(C, 63, "min")) - 1
F = {  # price, risk, momentum
    "rv_63": rv63, "rv_126": rstd(dlog, 126), "idiovol_63": rstd(ret - beta * mkt, 63), "MAX_21": rext(ret, 21, "max"),
    "skew_63": m3 / np.clip(m2, 1e-12, None) ** 1.5, "beta_126": beta, "dist_252h": dist252, "ret_126": ret126,
    "ret126_21": ret126 - (sd(C, fl.shift(C, 21)) - 1), "pctup_252": rm((ret > 0).astype(np.float32), 252),
    "price": R, "log_dvol": np.log(np.clip(dvol, 1, None)), "amihud_21": np.log1p(rm(np.abs(ret) / np.clip(dvol, 1e4, None), 21)),
    # path-shape risk
    "downday_sev_63": downsev, "gap_freq_63": gapf, "drawdown_freq_63": ddf, "bounce_off_low_63": bounce,
    "at_high_flag": (dist252 > -0.02).astype(float),   # near but not at the high
}
fp = pd.read_parquet("data/newcycle/fund_panel_pit.parquet"); fp["signal_date"] = pd.to_datetime(fp["signal_date"])
dm = {pd.Timestamp(d).normalize(): i for i, d in enumerate(idx)}; sm = {tk: j for j, tk in enumerate(syms)}
fp["t"] = fp["signal_date"].map(lambda d: dm.get(pd.Timestamp(d).normalize(), -1)); fp["gid"] = fp["symbol"].map(lambda s: sm.get(str(s), -1)); fp = fp[(fp.t >= 0) & (fp.gid >= 0)]
sh = "EntityCommonStockSharesOutstanding"; piv = fp.pivot_table(index="signal_date", columns="symbol", values=sh, aggfunc="last").sort_index()
gr = (piv / piv.shift(12) - 1.0).stack().rename("issuance").reset_index(); fp = fp.merge(gr, on=["signal_date", "symbol"], how="left")
ta, ga = fp.t.to_numpy(), fp.gid.to_numpy(); mc = R[ta, ga] * fp[sh].to_numpy()
def scat(v): M = np.full((n, S), np.nan); M[ta, ga] = np.asarray(v, float); return M
A, NI, OCF, GP, REVv, EQ, LI, CASH = fp["Assets"], fp["NetIncomeLoss"], fp["NetCashProvidedByUsedInOperatingActivities"], fp["GrossProfit"], fp["Revenues"], fp["StockholdersEquity"], fp["Liabilities"], fp["CashAndCashEquivalentsAtCarryingValue"]
F.update({"f_ROA": scat(sd(NI, A)), "f_OCF_A": scat(sd(OCF, A)), "f_GP_A": scat(sd(GP, A)), "f_EY": scat(sd(NI.to_numpy(), mc)),
          "f_BM": scat(sd(EQ.to_numpy(), mc)), "f_SP": scat(sd(REVv.to_numpy(), mc)), "f_issuance": scat(fp["issuance"].to_numpy()),
          "f_log_mktcap": scat(np.log(np.clip(mc, 1, None))), "f_log_assets": scat(np.log(np.clip(A.to_numpy(), 1, None))),
          "f_cash_ratio": scat(sd(CASH, A)), "f_leverage": scat(sd(LI, A)),
          "f_junk": scat(-((EQ < 0).astype(float).to_numpy() + (sd(NI, A) < 0).astype(float).to_numpy())),
          "f_cash_runway": scat(np.where(OCF < 0, np.clip(sd(CASH, -OCF), 0, 20), 20.0))})
SH = np.full((n, S), np.nan); SH[ta, ga] = pd.to_numeric(fp[sh], errors="coerce").to_numpy(); SHf = pd.DataFrame(SH, index=idx).ffill(limit=300).to_numpy()
F["turnover_63"] = np.log1p(sd(rm(V, 63), SHf)); MCAP = R * SHf
uni = np.isfinite(C) & (R >= 1.0) & (R < 50.0) & (dvol >= 5e5)
def rank_reb(M):
    Rk = np.full((n, S), np.nan)
    for t in rebs:
        s = np.where(uni[t] & np.isfinite(M[t]))[0]
        if len(s) > 20: Rk[t, s] = np.argsort(np.argsort(M[t, s])) / len(s)
    return Rk
def peer_reb(M):  # demean within a size octile each rebalance
    P = np.full(n * 0 + n, 0); Pm = np.full((n, S), np.nan)
    for t in rebs:
        s = np.where(uni[t] & np.isfinite(M[t]) & np.isfinite(MCAP[t]))[0]
        if len(s) < 80: continue
        bk = np.clip(np.argsort(np.argsort(MCAP[t, s])) * 8 // len(s), 0, 7)
        out = M[t, s].copy()
        for b in range(8):
            mb = bk == b
            if mb.sum() > 3: out[mb] = M[t, s][mb] - np.median(M[t, s][mb])
        Pm[t, s] = out
    return Pm
# peer-relative factors
F["rv_vs_peers"] = -peer_reb(rv63); F["ROA_vs_peers"] = peer_reb(F["f_ROA"])
# multi-horizon volatility, momentum, trend
slope126, r2_126 = s2.slope_r2(np.log(np.clip(C, 1e-6, None)), 126)
for w in (21, 42): F[f"rv_{w}"] = rstd(dlog, w)
F["dvn_63"] = np.sqrt(rm(np.where(dlog < 0, dlog * dlog, 0.0), 63)); F["rv21_63"] = sd(rstd(dlog, 21), rstd(dlog, 63)); F["rv63_126"] = sd(rstd(dlog, 63), rstd(dlog, 126))
F["volofvol"] = rstd(rstd(dlog, 21), 63); F["slope126"] = slope126; F["r2_126"] = r2_126
for w in (21, 42, 63, 252): F[f"ret_{w}"] = sd(C, fl.shift(C, w)) - 1
for w in (21, 63, 126): F[f"dist_{w}h"] = sd(C, rext(C, w, "max")) - 1
def z(M, t, s):
    v = M[t, s]; o = np.zeros(len(s)); mm = np.isfinite(v)
    if mm.sum() > 5: o[mm] = (v[mm] - np.nanmean(v[mm])) / (np.nanstd(v[mm]) + 1e-9)
    return o
NAMES = list(F)
# names that just broke down on volume (excluded from the book)
mv = pd.DataFrame(V, index=idx).rolling(63, min_periods=30).median().to_numpy(); p20l = rext(fl.shift(C, 1), 20, "min")
wd = np.where(np.isfinite(ret) & np.isfinite(p20l), (ret <= -0.08) & (V >= 3 * mv) & (C < p20l), False)
washex = pd.DataFrame(wd.astype(float), index=idx).rolling(10, min_periods=1).max().to_numpy() > 0
groups, rawf, yrg = [], [], []
for i, t in enumerate(rebs):
    if not (2012 <= int(YR[i]) <= 2023): continue
    s = np.where(uni[t])[0]
    if len(s) < 60: continue
    groups.append((t, s)); rawf.append(H.fwd21(t)[s]); yrg.append(int(YR[i]))
X = np.vstack([np.column_stack([F[k][t, s] for k in NAMES]) for (t, s) in groups])
yr = np.concatenate([np.full(len(s), yrg[gi]) for gi, (t, s) in enumerate(groups)]); grp = np.concatenate([np.full(len(g[1]), gi) for gi, g in enumerate(groups)])
yrk = np.full(len(yr), np.nan)
for gi in range(len(groups)):
    m = grp == gi; y = rawf[gi] - np.nanmean(rawf[gi]); o = np.argsort(np.argsort(np.nan_to_num(y, nan=-9))); yrk[m] = (o + 1) / m.sum()
P = dict(num_leaves=7, max_depth=3, min_child_samples=300, learning_rate=0.03, feature_fraction=0.7, lambda_l2=10.0, verbose=-1, objective="regression")
if __name__ == "__main__":
    pred = np.full(len(yr), np.nan); imp = np.zeros(len(NAMES))
    for Y in H.DEV_YEARS:
        tr = yr < Y; te = yr == Y
        if tr.sum() < 2000 or te.sum() == 0: continue
        mdl = lgb.train(P, lgb.Dataset(X[tr], yrk[tr]), num_boost_round=300); pred[te] = mdl.predict(X[te]); imp += mdl.feature_importance("gain")
    sc = [pred[grp == gi] for gi in range(len(groups))]
    ics = []
    for gi in range(len(groups)):
        if yrg[gi] < 2016: continue
        yv = rawf[gi] - np.nanmean(rawf[gi]); ok = np.isfinite(sc[gi]) & np.isfinite(yv)
        if ok.sum() < 40: continue
        a = np.argsort(np.argsort(sc[gi][ok])).astype(float); b = np.argsort(np.argsort(yv[ok])).astype(float); a -= a.mean(); b -= b.mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
        if d > 0: ics.append((a @ b) / d)
    b_no = H.book(sc, groups, rawf, yrg, topn=30, weighting="invvol", perside="real", years=H.DEV_YEARS)
    b_wx = H.book(sc, groups, rawf, yrg, topn=30, weighting="invvol", perside="real", years=H.DEV_YEARS, exclude=washex)
    print(f"development backtest - {len(NAMES)} features, 2016-2023 | IC {np.mean(ics):+.4f}")
    print(f"  {'book':24s} {'net_total':>10s} {'net_ann':>8s} {'Sharpe':>7s} {'maxDD':>7s} {'cap$':>7s}")
    print(f"  {'top30 invvol':24s} {b_no['total']:+10.1%} {b_no['ann']:+8.1%} {b_no['sharpe']:7.2f} {b_no['maxDD']:+7.0%} {b_no['cap']/1e6:5.1f}M")
    print(f"  {'+ washout-exclusion':24s} {b_wx['total']:+10.1%} {b_wx['ann']:+8.1%} {b_wx['sharpe']:7.2f} {b_wx['maxDD']:+7.0%} {b_wx['cap']/1e6:5.1f}M")
    print("year-by-year net (washout-excl): " + "  ".join(f"{y}:{b_wx['by'].get(y,0):+.0%}" for y in H.DEV_YEARS))
    print("top 12 features by gain:", ", ".join(np.array(NAMES)[np.argsort(-imp)[:12]]))
