# Fast cross-sectional feature scoring. The heavy substrate (per-date universe, rebalance rows,
# delisting-safe forward returns and their centered ranks) is computed once and cached to npz;
# scoring is then matrix algebra -- rank-IC as a cosine between centered feature ranks and forward
# ranks, over a whole batch at once. Set FEATLAB_GPU=1 (with cupy) to run the argsort/matmul on GPU.
# Exposes raw matrices (Cn, On, Rn, Vn, DVn) and rolling helpers for building features.
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
GPU = os.environ.get("FEATLAB_GPU", "0") == "1"
try:
    import cupy as cp; xp = cp if GPU else np
except Exception:
    cp = None; xp = np; GPU = False

REBAL, HORIZON = 21, 21
MIN_PRICE, MIN_DVOL, COV_WIN, COV_MIN, DVOL_WIN = 1.0, 1e6, 63, 0.80, 63
DISTRESS_RET, WIPEOUT = -0.50, -1.0
PRICE_CACHE = "data/newcycle/broad_universe.npz"
SUB_CACHE = "data/newcycle/featlab_substrate.npz"

# raw matrices (for building features)
def _load_prices():
    if os.path.exists(PRICE_CACHE):
        z = np.load(PRICE_CACHE, allow_pickle=True)
        return (z["Cn"], z["On"], z["Rn"], z["Vn"], pd.DatetimeIndex(z["idx"]), list(z["syms"]))
    allf = sorted(glob.glob("data/prices/clean/symbol=*/data.parquet"))
    def sof(f): return f.split("symbol=", 1)[1].split("\\")[0].split("/")[0]
    dts = set()
    for f in allf: dts.update(pd.to_datetime(pd.read_parquet(f, columns=["date"])["date"]).tolist())
    idx = pd.DatetimeIndex(sorted(d for d in dts if d >= pd.Timestamp("2007-06-01"))); pos = {d: i for i, d in enumerate(idx)}
    syms = [sof(f) for f in allf]; n, ns = len(idx), len(syms)
    Cn = np.full((n, ns), np.nan, np.float32); On = Cn.copy(); Rn = Cn.copy(); Vn = Cn.copy()
    for j, f in enumerate(allf):
        d = pd.read_parquet(f, columns=["date", "open", "close", "raw_close", "volume"]); d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"].isin(pos)].drop_duplicates("date"); ii = d["date"].map(pos).to_numpy()
        Cn[ii, j] = d["close"]; On[ii, j] = d["open"]; Rn[ii, j] = d["raw_close"]; Vn[ii, j] = d["volume"]
    np.savez(PRICE_CACHE, Cn=Cn, On=On, Rn=Rn, Vn=Vn, idx=idx.values, syms=np.array(syms, object))
    return Cn, On, Rn, Vn, idx, syms

print("featlab: loading price matrices...", flush=True)
Cn, On, Rn, Vn = (a.astype(np.float32) for a in _load_prices()[:4])
_, _, _, _, idx, syms = _load_prices()
n, S = Cn.shape; years = np.asarray(idx.year); DVn = (Rn * Vn).astype(np.float32)

# feature-building helpers
def shift(a, k):
    out = np.full_like(a, np.nan); out[k:] = a[:-k] if k > 0 else a; return out
def _cumsum_nan(a):
    m = np.isfinite(a).astype(np.float32); x = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    return np.cumsum(x, axis=0), np.cumsum(m, axis=0)
def roll_sum(a, w, minp=None):
    cs, cm = _cumsum_nan(a); s = cs.copy(); c = cm.copy()
    s[w:] = cs[w:] - cs[:-w]; c[w:] = cm[w:] - cm[:-w]
    minp = minp or max(2, int(w * 0.6)); return np.where(c >= minp, s, np.nan)
def roll_mean(a, w, minp=None):
    cs, cm = _cumsum_nan(a); s = cs.copy(); c = cm.copy()
    s[w:] = cs[w:] - cs[:-w]; c[w:] = cm[w:] - cm[:-w]
    minp = minp or max(2, int(w * 0.6)); return np.where(c >= minp, s / np.maximum(c, 1), np.nan)
def roll_std(a, w, minp=None):
    m1 = roll_mean(a, w, minp); m2 = roll_mean(a * a, w, minp); v = m2 - m1 * m1; return np.sqrt(np.clip(v, 0, None))
def xs_z(a):                                                            # cross-sectional z per date (common feature norm)
    mu = np.nanmean(a, axis=1, keepdims=True); sd = np.nanstd(a, axis=1, keepdims=True)
    return (a - mu) / np.where(sd > 1e-12, sd, np.nan)

# cached substrate: universe, rebalances, forward-rank
def _build_substrate():
    dvol = pd.DataFrame(DVn, index=idx).rolling(DVOL_WIN, min_periods=int(DVOL_WIN * .8)).median().to_numpy()
    cov = pd.DataFrame(np.isfinite(Cn), index=idx).rolling(COV_WIN, min_periods=1).mean().to_numpy()
    e0 = np.isfinite(Cn) & (Rn >= MIN_PRICE) & (dvol >= MIN_DVOL) & (cov >= COV_MIN)
    rebs = [r for r in range(max(273, int(np.searchsorted(idx.values, np.datetime64("2009-06-01")))), n - HORIZON - 2, REBAL)]
    Cff = pd.DataFrame(Cn, index=idx).ffill(limit=HORIZON + 5).to_numpy()
    eidx = []; frc = []; counts = []
    for r in rebs:
        e = On[r + 1]; x = Cff[r + HORIZON]
        good = e0[r] & np.isfinite(e) & (e > 0) & np.isfinite(x) & (x > 0)
        ix = np.where(good)[0]; fwd = x[ix] / e[ix] - 1.0
        fwd = np.where(fwd < DISTRESS_RET, WIPEOUT, fwd)
        rk = np.argsort(np.argsort(fwd)).astype(np.float32); rk -= rk.mean()
        nrm = np.sqrt((rk * rk).sum()); frc.append((rk / nrm).astype(np.float32)); eidx.append(ix.astype(np.int32)); counts.append(len(ix))
    yr = np.array([years[r] for r in rebs]); return rebs, eidx, frc, counts, yr

if os.path.exists(SUB_CACHE):
    z = np.load(SUB_CACHE, allow_pickle=True)
    rebs = list(z["rebs"]); EIDX = list(z["eidx"]); FRC = list(z["frc"]); CNT = list(z["cnt"]); YR = z["yr"]
else:
    print("featlab: building substrate (one-time)...", flush=True)
    rebs, EIDX, FRC, CNT, YR = _build_substrate()
    np.savez(SUB_CACHE, rebs=np.array(rebs), eidx=np.array(EIDX, object), frc=np.array(FRC, object), cnt=np.array(CNT), yr=YR)
NR = len(rebs); rebpos = np.array(rebs)
ISM = YR <= 2018; OOSM = YR >= 2019
if GPU:
    EIDX = [cp.asarray(e) for e in EIDX]; FRC = [cp.asarray(f) for f in FRC]
print(f"featlab READY: {NR} rebalances, ~{int(np.median(CNT))} names/date, S={S}. backend={'GPU(cupy)' if GPU else 'CPU(numpy)'}", flush=True)

def to_rebal(full):                                                    # slice a [dates x S] matrix to [NR x S]
    return full[rebpos]

# scorer
def _rank_axis1(X):                                                    # rank along axis1; NaN -> lowest (cheap, no nanmedian)
    Xf = xp.where(xp.isfinite(X), X, -xp.inf)
    return xp.argsort(xp.argsort(Xf, axis=1), axis=1).astype(xp.float32)

def score_batch(F, method="pearson"):                                  # F:(B,NR,S). 'pearson'=fast screen (no sort), 'rank'=robust
    F = xp.asarray(F); B = F.shape[0]; IC = xp.empty((B, NR), xp.float32)
    for ri in range(NR):
        ei = EIDX[ri]; fr = FRC[ri]; X = F[:, ri, ei]                  # fr: unit-norm centered fwd ranks (k,)
        if method == "rank": X = _rank_axis1(X)
        fin = xp.isfinite(X); Xf = xp.where(fin, X, 0.0)               # NaN-robust: missing stocks drop out neutrally
        mu = Xf.sum(axis=1, keepdims=True) / xp.maximum(fin.sum(axis=1, keepdims=True), 1)
        Xc = xp.where(fin, X - mu, 0.0)
        nrm = xp.sqrt((Xc * Xc).sum(axis=1)); IC[:, ri] = (Xc @ fr) / xp.where(nrm > 0, nrm, xp.nan)
    IC_np = cp.asnumpy(IC) if GPU else IC
    m = np.nanmean(IC_np, 1); sd = np.nanstd(IC_np, 1); nb = np.isfinite(IC_np).sum(1)
    pos = np.nansum(IC_np > 0, 1) / np.maximum(nb, 1)
    return dict(ic=m, ic_ir=m / np.where(sd > 0, sd, np.nan), ic_t=m / np.where(sd > 0, sd, np.nan) * np.sqrt(np.maximum(nb, 1)),
                pct_pos=pos, ic_is=np.nanmean(IC_np[:, ISM], 1), ic_oos=np.nanmean(IC_np[:, OOSM], 1))

def score_stream(gen, batch=256, method="pearson"):                    # gen yields full [dates x S] OR [NR x S] matrices
    out = {k: [] for k in ("ic", "ic_ir", "ic_t", "pct_pos", "ic_is", "ic_oos")}; names = []; buf = []; nm = []
    def flush():
        if not buf: return
        F = np.stack([f if f.shape[0] == NR else to_rebal(f) for f in buf]); r = score_batch(F, method)
        for k in out: out[k].append(r[k])
        names.extend(nm); buf.clear(); nm.clear()
    for item in gen:
        name, feat = item if isinstance(item, tuple) else (str(len(names) + len(nm)), item)
        buf.append(feat.astype(np.float32)); nm.append(name)
        if len(buf) >= batch: flush()
    flush()
    df = pd.DataFrame({k: np.concatenate(v) for k, v in out.items()}); df.insert(0, "feature", names); return df

def null_bar(n_features=100000, n_perm=200, seed=0):                    # permutation-null: |IC-t| a random feature clears
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n_perm, NR, S)).astype(np.float32); t = np.abs(score_batch(F)["ic_t"])
    print(f"\nPermutation null ({n_perm} random features): |IC-t| 50/95/99/max = "
          f"{np.nanpercentile(t,50):.2f}/{np.nanpercentile(t,95):.2f}/{np.nanpercentile(t,99):.2f}/{np.nanmax(t):.2f}")
    import math
    bar = abs(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - 0.05 / max(n_features, 1) / 2))
    print(f"With {n_features:,} features, Bonferroni 5% bar |IC-t| ~ {bar:.2f}  (a feature below this is noise).")

if __name__ == "__main__":
    print("\n=== THROUGHPUT BENCHMARK (pure scorer) ===", flush=True)
    rng = np.random.default_rng(0); F = rng.standard_normal((512, NR, S)).astype(np.float32)
    for method in ("pearson", "rank"):
        if GPU: score_batch(F, method)                                  # warm up
        t0 = time.time(); score_batch(F, method); dt = time.time() - t0; rate = 512 / dt
        print(f"  {method:>8}: {dt*1000:7.0f} ms / 512 feats -> {rate:8.0f} feats/sec -> 100k in ~{100000/rate/60:5.1f} min "
              f"(8-core mp ~{100000/rate/60/7:.1f} min)", flush=True)
    print("\n=== real volume-feature demo (built with fast helpers) ===", flush=True)
    feats = []
    for w in (5, 10, 21, 42, 63, 126):
        feats.append((f"vol_z_{w}", xs_z(roll_mean(Vn, w))))
        feats.append((f"vol_surge_{w}", roll_mean(Vn, w) / roll_mean(Vn, 252)))
        feats.append((f"dvol_z_{w}", xs_z(roll_mean(DVn, w))))
    df = score_stream(iter(feats), batch=64).sort_values("ic_t", key=np.abs, ascending=False)
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False), flush=True)
    null_bar(100000)
    print("\nTo build YOUR features: make [dates x stocks] matrices from fl.Vn/DVn + roll_*/xs_z, pass to "
          "score_stream(generator) or stack to (B,NR,S) for score_batch. GPU: set FEATLAB_GPU=1 (needs cupy).")
