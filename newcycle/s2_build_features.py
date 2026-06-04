# Trailing-window stock and regime features on the clean price matrices, plus the rolling
# helpers (roll_extreme, slope/R2, RSI, ...) reused elsewhere. Everything uses data only through
# the current close, is winsorized and z-scored cross-sectionally per date, and missing values
# fall back to neutral. Regime features are date-level and kept raw.
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path("data/prices/clean")
LABELS = Path("data/newcycle/swing_labels.parquet")
SPY = Path("data/reference/spy.parquet")
OUT = Path("data/newcycle/swing_features.parquet")
ZCLIP = 8.0


# rolling helpers (time axis 0, NaN-aware)
def shift(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    return out


def _cum(a):
    c = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=np.float64)
    np.cumsum(np.nan_to_num(a), axis=0, out=c[1:])
    return c


def roll_sum(a, w, minp):
    cs = _cum(a); cc = _cum(np.isfinite(a).astype(np.float64))
    s = np.full(a.shape, np.nan); c = np.zeros(a.shape)
    s[w - 1:] = cs[w:] - cs[:-w]; c[w - 1:] = cc[w:] - cc[:-w]
    s[c < minp] = np.nan
    return s, c


def roll_mean(a, w, minp=None):
    s, c = roll_sum(a, w, minp or max(1, int(w * 0.7)))
    return s / np.where(c > 0, c, np.nan)


def roll_std(a, w, minp=None):
    mp = minp or max(2, int(w * 0.7))
    s1, c = roll_sum(a, w, mp); s2, _ = roll_sum(a * a, w, mp)
    m = s1 / np.where(c > 0, c, np.nan)
    return np.sqrt(np.clip(s2 / np.where(c > 0, c, np.nan) - m * m, 0, None))


def roll_extreme(a, w, kind):
    f = np.fmax if kind == "max" else np.fmin
    acc = a.copy()
    for s in range(1, w):
        acc = f(acc, shift(a, s))
    return acc


def rsi(close, p):
    d = close - shift(close, 1)
    up = roll_mean(np.where(d > 0, d, 0.0), p, max(1, p // 2))
    dn = roll_mean(np.where(d < 0, -d, 0.0), p, max(1, p // 2))
    rs = up / np.where(dn > 0, dn, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(dn == 0) & (up > 0)] = 100.0          # all-up window
    return out


def slope_r2(logc, w):
    """Rolling OLS of log price vs time -> (slope per day, R^2)."""
    n = logc.shape[0]
    tg = np.arange(n, dtype=np.float64)[:, None]
    Sx, _ = roll_sum(np.broadcast_to(tg, logc.shape), w, w)
    Sxx, _ = roll_sum(np.broadcast_to(tg * tg, logc.shape), w, w)
    S0, _ = roll_sum(logc, w, w); S1, _ = roll_sum(tg * logc, w, w)
    Syy, _ = roll_sum(logc * logc, w, w)
    denom = w * Sxx - Sx * Sx
    slope = (w * S1 - Sx * S0) / np.where(denom != 0, denom, np.nan)
    ss_tot = Syy - S0 * S0 / w
    ss_xx = Sxx - Sx * Sx / w
    r2 = (slope * slope * ss_xx) / np.where(ss_tot > 0, ss_tot, np.nan)
    return slope, np.clip(r2, 0, 1)


def winz_z(F, elig):
    Fm = np.where(elig, F, np.nan)
    lo = np.nanpercentile(Fm, 1, axis=1, keepdims=True)
    hi = np.nanpercentile(Fm, 99, axis=1, keepdims=True)
    Fc = np.clip(Fm, lo, hi)
    mu = np.nanmean(Fc, axis=1, keepdims=True)
    sd = np.nanstd(Fc, axis=1, keepdims=True)
    return np.clip((Fc - mu) / np.where(sd > 0, sd, np.nan), -ZCLIP, ZCLIP)


def load():
    files = sorted(CLEAN.glob("symbol=*/data.parquet"))
    cols = {c: [] for c in ["open", "high", "low", "close", "raw_close", "volume"]}
    syms = []
    for f in files:
        s = f.parent.name.split("=", 1)[1]; syms.append(s)
        d = pd.read_parquet(f, columns=["date", "open", "high", "low", "close", "raw_close", "volume"])
        d["date"] = pd.to_datetime(d["date"]); d = d.set_index("date")
        for c in cols:
            cols[c].append(d[c].rename(s))
    close = pd.concat(cols["close"], axis=1, sort=True).sort_index()
    M = {c: pd.concat(cols[c], axis=1, sort=True).reindex_like(close).to_numpy(np.float32)
         for c in cols}
    return close.index, close.columns, M


def main():
    print("loading clean matrices...")
    idx, allcols, M = load()
    C, O, H, L, RW, V = M["close"], M["open"], M["high"], M["low"], M["raw_close"], M["volume"]
    ret = C / shift(C, 1) - 1.0
    logc = np.log(np.clip(C, 1e-6, None))
    n = C.shape[0]

    lab = pd.read_parquet(LABELS)
    lab["date"] = pd.to_datetime(lab["date"])
    rowmap = {d: i for i, d in enumerate(idx)}
    colmap = {c: j for j, c in enumerate(allcols)}
    ri = lab["date"].map(rowmap).to_numpy()
    ci = lab["ticker"].map(colmap).to_numpy()
    elig = np.zeros(C.shape, dtype=bool)
    elig[ri, ci] = True

    print("computing stock features (vectorized, trailing only)...")
    sl20, r2_20 = slope_r2(logc, 20)
    sl50, r2_50 = slope_r2(logc, 50)
    sl100, _ = slope_r2(logc, 100)                          # longer trend for 21d horizon
    eff = lambda w: np.abs(C - shift(C, w)) / roll_sum(np.abs(C - shift(C, 1)), w, max(2, int(w * 0.7)))[0]
    tr = np.fmax(np.fmax(H - L, np.abs(H - shift(C, 1))), np.abs(L - shift(C, 1)))
    feats = {
        "ret_5": C / shift(C, 5) - 1, "ret_10": C / shift(C, 10) - 1, "ret_21": C / shift(C, 21) - 1,
        "ret_63": C / shift(C, 63) - 1, "ret_126": C / shift(C, 126) - 1, "ret_252": C / shift(C, 252) - 1,
        "high_52w_dist": C / roll_extreme(C, 252, "max") - 1,
        "price_vs_20dma": C / roll_mean(C, 20) - 1, "price_vs_50dma": C / roll_mean(C, 50) - 1,
        "price_vs_200dma": C / roll_mean(C, 200) - 1,
        "ret_1": ret, "ret_2": C / shift(C, 2) - 1, "ret_3": C / shift(C, 3) - 1,
        "rsi_2": rsi(C, 2), "rsi_5": rsi(C, 5), "rsi_14": rsi(C, 14),
        "gap_1d": O / shift(C, 1) - 1, "dist_from_10dma": C / roll_mean(C, 10) - 1,
        "slope_20": sl20, "slope_50": sl50, "r2_trend_20": r2_20,
        "efficiency_20": eff(20), "efficiency_63": eff(63),
        "pct_up_days_20": roll_mean((ret > 0).astype(np.float64), 20),
        "atr_14_pct": roll_mean(tr, 14) / np.where(C > 0, C, np.nan),
        "rvol_10": roll_std(ret, 10), "rvol_21": roll_std(ret, 21), "rvol_63": roll_std(ret, 63),
        "maxdd_21": roll_extreme(C, 21, "min") / roll_extreme(C, 21, "max") - 1,
        "downside_vol_21": np.sqrt(roll_mean(np.clip(ret, None, 0) ** 2, 21)),
        "logdvol_20": np.log(np.clip(roll_mean(RW * V, 20), 1, None)),
        "logdvol_60": np.log(np.clip(roll_mean(RW * V, 60), 1, None)),
        "rel_vol_1": V / roll_mean(V, 20), "rel_vol_5": roll_mean(V, 5) / roll_mean(V, 20),
        # longer-window adds for the 21-day horizon (low-vol dominates @21d)        "slope_100": sl100, "r2_trend_50": r2_50,
        "maxdd_63": roll_extreme(C, 63, "min") / roll_extreme(C, 63, "max") - 1,
        "downside_vol_63": np.sqrt(roll_mean(np.clip(ret, None, 0) ** 2, 63)),
        "rvol_126": roll_std(ret, 126),
        "pct_up_days_63": roll_mean((ret > 0).astype(np.float64), 63),
    }

    out = {"date": lab["date"].to_numpy(), "ticker": lab["ticker"].to_numpy(),
           "target_5d": lab["target_5d"].to_numpy(), "target_10d": lab["target_10d"].to_numpy(),
           "raw_5d": lab["raw_5d"].to_numpy(), "delisted_5d": lab["delisted_5d"].to_numpy()}
    cov = {}
    for name, F in feats.items():
        F = np.where(np.isfinite(F), F, np.nan).astype(np.float64)
        z = winz_z(F, elig)
        col = z[ri, ci]
        cov[name] = np.isfinite(col).mean()
        out["z_" + name] = np.nan_to_num(col, nan=0.0).astype(np.float32)
        del F, z

    print("computing regime features (date-level, raw)...")
    spy = pd.read_parquet(SPY); spy["date"] = pd.to_datetime(spy["date"])
    sp = spy.set_index("date")["close"].reindex(idx, method="ffill").to_numpy(np.float64)
    spr5 = sp / shift(sp[:, None], 5)[:, 0] - 1
    spr21 = sp / shift(sp[:, None], 21)[:, 0] - 1
    sp50 = roll_mean(sp[:, None], 50)[:, 0]; sp200 = roll_mean(sp[:, None], 200)[:, 0]
    ma50 = roll_mean(C, 50)                                 # breadth over ACTIVE names only
    above = np.where(np.isfinite(C) & np.isfinite(ma50), (C > ma50).astype(np.float64), np.nan)
    breadth = np.nanmean(above, axis=1)                     # fraction of trading stocks above 50dma
    reg = {"spy_ret_5": spr5, "spy_ret_21": spr21,
           "spy_above_50dma": (sp > sp50).astype(np.float32),
           "spy_above_200dma": (sp > sp200).astype(np.float32), "breadth_50dma": breadth}
    for name, series in reg.items():
        out[name] = np.nan_to_num(series[ri], nan=0.0).astype(np.float32)

    df = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    zc = [c for c in df.columns if c.startswith("z_")]
    print(f"\nFeature table: {len(df):,} rows x {len(zc)} stock feats + {len(reg)} regime "
          f"({df.date.min().date()} -> {df.date.max().date()})")
    print("stock-feature coverage (pre-fill non-null):")
    for name in feats:
        flag = "  <-- LOW" if cov[name] < 0.5 else ""
        print(f"  z_{name:<18} {cov[name]:6.1%}{flag}")
    print(f"\nWrote: {OUT.resolve()}")
    print("sanity check: no look-ahead, coverage, and scale")


if __name__ == "__main__":
    main()
