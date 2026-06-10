# Runs the model on the latest rebalance and writes its current longs/shorts to
# reports/sp500_predictions.json, with a few SHAP-based reasons per name. 5-seed ensemble,
# 63-day target, trained on all history whose forward return is already known.
import sys, warnings, json
import numpy as np, lightgbm as lgb
warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0, "rebuild"); sys.path.insert(0, "newcycle")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import best_model as BM
import featlab as fl
H = BM.H; F = BM.F; NAMES = BM.NAMES; P = BM.P; MCAP = BM.MCAP; R = H.R; dvol = H.dvol; rebs = H.rebs; n, S = H.n, H.S
syms = fl.syms; idx = fl.idx; YR = np.asarray(fl.YR)
def fwd_h(t, h):
    if t + h >= n: return np.full(S, np.nan)
    e = H.O[t + 1]; x = H.Cff[t + h]; r = np.where(np.isfinite(e) & (e > 0) & np.isfinite(x) & (x > 0), x / np.where(e > 0, e, np.nan) - 1.0, np.nan)
    return np.where(r < -0.5, -1.0, r)
SP = np.zeros((n, S), bool)
for t in rebs:
    v = np.where(np.isfinite(MCAP[t]) & (R[t] > 1) & (dvol[t] >= 5e6))[0]
    if len(v) > 500: SP[t, v[np.argsort(-MCAP[t, v])[:500]]] = True
    else: SP[t, v] = True
groups = []
for i, t in enumerate(rebs):
    s = np.where(SP[t])[0]
    if len(s) >= 100: groups.append((i, t, s))
X = np.vstack([np.column_stack([F[k][t, s] for k in NAMES]) for (_, t, s) in groups])
gi_ix = np.concatenate([np.full(len(s), gi) for gi, (_, t, s) in enumerate(groups)])
yrg = [int(YR[i]) for (i, t, s) in groups]
# 63d rank target (forward-complete rows only)
yk = np.full(len(X), np.nan)
for gi, (i, t, s) in enumerate(groups):
    f63 = fwd_h(t, 63)[s]
    if np.isfinite(f63).sum() < 40: continue
    m = gi_ix == gi; y = f63 - np.nanmean(f63); o = np.argsort(np.argsort(np.nan_to_num(y, nan=-9))); yk[m] = (o + 1) / m.sum()
last = len(groups) - 1                      # most recent rebalance
while last > 0 and (gi_ix == last).sum() < 100: last -= 1
_, tL, sL = groups[last]; Xl = X[gi_ix == last]
train = np.isfinite(yk)                      # all forward-complete history (strictly earlier than tL's unknown fwd)
print(f"as-of {idx[tL].date()} | universe {len(sL)} | train rows {int(train.sum()):,}", flush=True)
scores = np.zeros(len(Xl)); contrib = np.zeros((len(Xl), len(NAMES) + 1))
for ki in range(5):
    Pk = dict(P, bagging_fraction=0.8, bagging_freq=1, seed=ki, bagging_seed=ki, feature_fraction_seed=ki + 7)
    mdl = lgb.train(Pk, lgb.Dataset(X[train], yk[train]), num_boost_round=300)
    p = mdl.predict(Xl); scores += np.argsort(np.argsort(p)) / len(p)
    contrib += mdl.predict(Xl, pred_contrib=True)
scores /= 5.0; contrib /= 5.0
# z of each feature within the current universe
Z = np.zeros_like(Xl)
for j in range(len(NAMES)):
    v = Xl[:, j]; mu = np.nanmean(v); sd = np.nanstd(v) + 1e-9; Z[:, j] = (np.nan_to_num(v, nan=mu) - mu) / sd
PH = {
 "rv_63":("elevated 3-month volatility","unusually calm price action"),"rv_126":("elevated volatility","calm long-run volatility"),
 "rv_21":("choppy recent trading","calm recent trading"),"rv_42":("elevated volatility","calm price action"),
 "idiovol_63":("high stock-specific risk","low stock-specific noise"),"volofvol":("unstable volatility","stable volatility"),
 "MAX_21":("lottery-like recent spikes","no lottery-style spikes"),"skew_63":("positively skewed returns","negatively skewed returns"),
 "beta_126":("high market beta","low market beta"),"dist_252h":("near its 52-week high","well below its highs"),
 "dist_126h":("near recent highs","below recent highs"),"dist_63h":("near 3-month highs","below 3-month highs"),
 "at_high_flag":("sitting right at its high","not extended at highs"),"ret_126":("strong 6-month momentum","weak 6-month momentum"),
 "ret_63":("strong 3-month momentum","weak 3-month momentum"),"ret_252":("strong 12-month momentum","weak 12-month momentum"),
 "ret126_21":("strong intermediate momentum","fading intermediate momentum"),"ret_21":("strong 1-month move","weak 1-month move"),
 "ret_42":("firm 2-month trend","soft 2-month trend"),"pctup_252":("consistent up-days","few up-days"),
 "slope126":("clean uptrend","downward drift"),"r2_126":("smooth, trending path","choppy path"),
 "price":("higher share price","lower share price"),"log_dvol":("very liquid","thinner liquidity"),
 "amihud_21":("high price impact","liquid, low impact"),"turnover_63":("high share turnover","low, sticky turnover"),
 "downday_sev_63":("severe down-days","mild down-days"),"gap_freq_63":("frequent price gaps","smooth, gap-free"),
 "drawdown_freq_63":("frequent drawdowns","few drawdowns"),"bounce_off_low_63":("recent bounce off lows","no dead-cat bounce"),
 "rv_vs_peers":("calmer than size peers","more volatile than peers"),"ROA_vs_peers":("more profitable than peers","less profitable than peers"),
 "f_ROA":("strongly profitable (ROA)","unprofitable (weak ROA)"),"f_OCF_A":("high operating cash flow","weak cash generation"),
 "f_GP_A":("high gross profitability","thin gross margins"),"f_EY":("cheap - high earnings yield","expensive - low earnings yield"),
 "f_BM":("cheap on book value","richly valued on book"),"f_SP":("cheap on sales","expensive on sales"),
 "f_issuance":("diluting its share count","not diluting / buying back"),"f_log_mktcap":("large cap","smaller cap"),
 "f_log_assets":("large asset base","smaller asset base"),"f_cash_ratio":("cash-heavy balance sheet","low cash reserves"),
 "f_leverage":("highly levered","conservatively financed"),"f_junk":("solvent and profitable","distressed (losses / neg. equity)"),
 "f_cash_runway":("long cash runway","short cash runway"),"dvn_63":("high downside volatility","low downside volatility"),
 "rv21_63":("volatility rising","volatility easing"),"rv63_126":("volatility elevated vs trend","volatility subdued"),
}
def phrase(row_i, j):
    nm = NAMES[j]; hi, lo = PH.get(nm, (nm.replace("_", " "), "low " + nm.replace("_", " ")))
    return hi if Z[row_i, j] > 0 else lo
def pick_row(row_i, direction):
    c = contrib[row_i, :len(NAMES)]; order = np.argsort(-c * direction)   # features agreeing with the pick
    drivers = []
    mx = np.max(np.abs(c)) + 1e-9
    for j in order[:3]:
        if c[j] * direction <= 0: break
        drivers.append({"text": phrase(row_i, int(j)), "mag": round(float(abs(c[j]) / mx), 3)})
    gid = sL[row_i]
    return {"ticker": str(syms[gid]), "price": round(float(R[tL, gid]), 2),
            "rank": None, "drivers": drivers}
order = np.argsort(-scores)
NL = 15
longs = [pick_row(int(i), +1) for i in order[:NL]]
shorts = [pick_row(int(i), -1) for i in order[-NL:][::-1]]
for k, p in enumerate(longs): p["rank"] = k + 1
for k, p in enumerate(shorts): p["rank"] = k + 1
out = {"as_of": str(idx[tL].date()), "universe": "Top-500 by market cap (S&P-500 proxy)",
       "model": "5-seed LightGBM ensemble, 63-day signal, market-neutral, beta + factor-neutral",
       "book": "Long top-30 / short bottom-30 (inverse-vol, monthly). Shown: highest-conviction 15 per side.",
       "n_features": len(NAMES), "longs": longs, "shorts": shorts}
import os; os.makedirs("reports", exist_ok=True)
open("reports/sp500_predictions.json", "w").write(json.dumps(out, indent=1))
print("longs: " + ", ".join(p["ticker"] for p in longs))
print("shorts: " + ", ".join(p["ticker"] for p in shorts))
print("wrote reports/sp500_predictions.json")
