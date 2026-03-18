"""
================================================================
  Junior Data Scientist Assignment
  Trader Behavior Insights x Bitcoin Fear & Greed Index
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
OUTPUT_DIR = "output_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DARK_BG     = "#0f1117"
CARD_BG     = "#161b22"
BORDER      = "#30363d"
TEXT        = "#c9d1d9"
MUTED       = "#8b949e"
FEAR_COL    = "#ff4d4d"
GREED_COL   = "#00e396"
NEUTRAL_COL = "#ffa500"
ACCENT      = "#7c3aed"

SENTIMENT_PALETTE = {
    "Extreme Fear": "#ff1744",
    "Fear":         "#ff4d4d",
    "Neutral":      "#ffa500",
    "Greed":        "#00e396",
    "Extreme Greed":"#00bfa5",
}

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  BORDER,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titlepad":     12,
})

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ───────────────────────────────────────────────
# 1. LOAD DATA
# ───────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 1 - Loading Datasets")
print("="*55)

all_csvs = [f for f in os.listdir(".") if f.endswith(".csv")]
print(f"  CSV files found: {all_csvs}")

trader_candidates = [f for f in all_csvs if any(
    k in f.lower() for k in ["trade", "hyper", "histor", "position"]
)]
fear_candidates = [f for f in all_csvs if any(
    k in f.lower() for k in ["fear", "greed", "sentiment", "index"]
)]

# Fallback: pick by file size if names don't match
if not trader_candidates or not fear_candidates:
    sizes = sorted(all_csvs, key=lambda f: os.path.getsize(f), reverse=True)
    if not trader_candidates:
        trader_candidates = [sizes[0]]
    if not fear_candidates:
        fear_candidates = [sizes[-1]]

TRADER_FILE = trader_candidates[0]
FEAR_FILE   = fear_candidates[0]

print(f"\n  Trader file : {TRADER_FILE}")
print(f"  F&G file    : {FEAR_FILE}")

trades_raw = pd.read_csv(TRADER_FILE)
fg_raw     = pd.read_csv(FEAR_FILE)

print(f"\n  Trader shape : {trades_raw.shape}")
print(f"  F&G shape    : {fg_raw.shape}")
print(f"\n  Trader columns : {list(trades_raw.columns)}")
print(f"  F&G columns    : {list(fg_raw.columns)}")


# ───────────────────────────────────────────────
# 2. CLEAN TRADER DATA
# ───────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 2 - Cleaning Trader Data")
print("="*55)

df = trades_raw.copy()
df.columns = (df.columns.str.strip()
                         .str.lower()
                         .str.replace(" ", "_")
                         .str.replace("/", "_"))

# ── Find columns by keyword ────────────────────
def find_col(dataframe, keywords):
    for k in keywords:
        for c in dataframe.columns:
            if k in c.lower():
                return c
    return None

time_col  = find_col(df, ["time", "date", "timestamp"])
pnl_col   = find_col(df, ["closedpnl", "pnl", "profit", "realized"])
size_col  = find_col(df, ["sz", "size", "qty", "quantity", "amount"])
price_col = find_col(df, ["px", "price", "exec"])
side_col  = find_col(df, ["side", "direction"])
sym_col   = find_col(df, ["coin", "symbol", "asset", "market"])
acct_col  = find_col(df, ["account", "trader", "user", "address", "wallet"])
lev_col   = find_col(df, ["lev", "leverage"])

print(f"  time_col  = {time_col}")
print(f"  pnl_col   = {pnl_col}")
print(f"  size_col  = {size_col}")
print(f"  price_col = {price_col}")
print(f"  side_col  = {side_col}")
print(f"  sym_col   = {sym_col}")
print(f"  acct_col  = {acct_col}")
print(f"  lev_col   = {lev_col}")

if time_col is None:
    raise ValueError("Cannot find a time/date column in trader data.")

# ── Parse dates ───────────────────────────────
sample = df[time_col].dropna().iloc[0]
try:
    sample_float = float(str(sample).replace(",", ""))
    if sample_float > 1e12:
        df["date"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
    else:
        df["date"] = pd.to_datetime(df[time_col], unit="s", utc=True)
    df["date"] = df["date"].dt.tz_localize(None)
except (ValueError, TypeError):
    df["date"] = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

df["date"] = df["date"].dt.normalize()
print(f"\n  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

# ── Convert numerics ──────────────────────────
for col in [pnl_col, size_col, price_col, lev_col]:
    if col:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Rename to standard names
if pnl_col:
    df = df.rename(columns={pnl_col: "pnl"})
    pnl_col = "pnl"
if size_col:
    df = df.rename(columns={size_col: "size"})
    size_col = "size"

print(f"  Total trades: {len(df):,}")


# ───────────────────────────────────────────────
# 3. CLEAN FEAR & GREED DATA
# ───────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 3 - Cleaning Fear & Greed Data")
print("="*55)

fg = fg_raw.copy()
fg.columns = (fg.columns.str.strip()
                         .str.lower()
                         .str.replace(" ", "_"))

fg_date_col = find_col(fg, ["date", "time", "timestamp"])
fg_cls_col  = find_col(fg, ["classif", "sentiment", "label", "category", "fear", "greed"])
fg_val_col  = find_col(fg, ["value", "score", "index"])

print(f"  fg_date_col = {fg_date_col}")
print(f"  fg_cls_col  = {fg_cls_col}")
print(f"  fg_val_col  = {fg_val_col}")

if fg_date_col is None:
    raise ValueError("Cannot find date column in Fear & Greed data.")

fg["date"] = pd.to_datetime(fg[fg_date_col], dayfirst=True, errors="coerce")
if fg["date"].dt.tz is not None:
    fg["date"] = fg["date"].dt.tz_localize(None)
fg["date"] = fg["date"].dt.normalize()

# Classification: use existing column or derive from numeric
if fg_cls_col:
    fg = fg.rename(columns={fg_cls_col: "classification"})
elif fg_val_col:
    fg[fg_val_col] = pd.to_numeric(fg[fg_val_col], errors="coerce")
    def classify(v):
        if pd.isna(v):    return np.nan
        if v <= 24:       return "Extreme Fear"
        elif v <= 44:     return "Fear"
        elif v <= 54:     return "Neutral"
        elif v <= 74:     return "Greed"
        else:             return "Extreme Greed"
    fg["classification"] = fg[fg_val_col].apply(classify)
else:
    raise ValueError("Cannot find classification or numeric value column in F&G data.")

if fg_val_col:
    fg = fg.rename(columns={fg_val_col: "fg_value"})

fg["classification"] = fg["classification"].astype(str).str.strip().str.title()

keep_cols = ["date", "classification"]
if "fg_value" in fg.columns:
    keep_cols.append("fg_value")
fg = fg[keep_cols]

print(f"\n  F&G date range: {fg['date'].min().date()} -> {fg['date'].max().date()}")
print(f"  Sentiment value counts:\n{fg['classification'].value_counts()}")


# ───────────────────────────────────────────────
# 4. MERGE
# ───────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 4 - Merging Datasets")
print("="*55)

merged = df.merge(fg, on="date", how="left")
print(f"  Total merged rows    : {len(merged):,}")
print(f"  Missing sentiment    : {merged['classification'].isna().sum():,}")

merged = merged.dropna(subset=["classification"])
print(f"  Final working rows   : {len(merged):,}")

SENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
existing_sentiments = [s for s in SENT_ORDER if s in merged["classification"].unique()]
print(f"  Sentiments present   : {existing_sentiments}")


# ───────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ───────────────────────────────────────────────
if pnl_col:
    merged["profitable"] = merged["pnl"] > 0

if side_col:
    merged["side_clean"] = merged[side_col].astype(str).str.strip().str.upper()

merged["broad_sentiment"] = merged["classification"].apply(
    lambda s: "Fear" if "Fear" in s else ("Greed" if "Greed" in s else "Neutral")
)

print("\n  Feature engineering done.")


# ═══════════════════════════════════════════════
#   CHARTS
# ═══════════════════════════════════════════════
print("\n" + "="*55)
print("  Generating Charts...")
print("="*55)


# ── Chart 1: Sentiment Distribution ───────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Fear & Greed Sentiment Distribution", fontsize=15, color=TEXT, y=1.01)

sent_counts = (merged["classification"]
               .value_counts()
               .reindex(existing_sentiments)
               .dropna())
colors = [SENTIMENT_PALETTE.get(s, ACCENT) for s in sent_counts.index]

ax = axes[0]
bars = ax.bar(sent_counts.index, sent_counts.values, color=colors,
              edgecolor=BORDER, linewidth=0.8)
ax.set_title("Trade Count by Sentiment")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Trades")
for bar, val in zip(bars, sent_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.01,
            f"{val:,}", ha="center", va="bottom", fontsize=9)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

wedges, texts, autotexts = axes[1].pie(
    sent_counts.values, labels=sent_counts.index, colors=colors,
    autopct="%1.1f%%", startangle=140,
    wedgeprops=dict(edgecolor=DARK_BG, linewidth=1.5)
)
for t in texts + autotexts:
    t.set_color(TEXT)
axes[1].set_title("Proportion of Sentiment Periods")
plt.tight_layout()
save(fig, "01_sentiment_distribution.png")


# ── Chart 2: Avg PnL by Sentiment ─────────────
if pnl_col:
    pnl_stats = (merged.groupby("classification")["pnl"]
                       .agg(["mean", "median", "sum", "count"])
                       .reindex(existing_sentiments)
                       .dropna())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("PnL Performance vs Market Sentiment", fontsize=15, y=1.01)
    metrics = [("mean", "Avg PnL per Trade"),
               ("median", "Median PnL per Trade"),
               ("sum", "Total PnL")]
    for ax, (metric, label) in zip(axes, metrics):
        bar_colors = [GREED_COL if v >= 0 else FEAR_COL
                      for v in pnl_stats[metric]]
        bars = ax.bar(pnl_stats.index, pnl_stats[metric],
                      color=bar_colors, edgecolor=BORDER, linewidth=0.8)
        ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Sentiment")
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        for bar, val in zip(bars, pnl_stats[metric]):
            offset = abs(val) * 0.02
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + offset if val >= 0 else val - offset,
                    f"{val:,.1f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8)
    plt.tight_layout()
    save(fig, "02_pnl_by_sentiment.png")
    print(f"\n  PnL by Sentiment:\n{pnl_stats.to_string()}")


# ── Chart 3: Win Rate by Sentiment ────────────
if pnl_col:
    win_rate = (merged.groupby("classification")["profitable"]
                      .mean()
                      .reindex(existing_sentiments)
                      .dropna() * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [SENTIMENT_PALETTE.get(s, ACCENT) for s in win_rate.index]
    bars = ax.bar(win_rate.index, win_rate.values,
                  color=bar_colors, edgecolor=BORDER, linewidth=0.8)
    ax.axhline(50, color=NEUTRAL_COL, linewidth=1.2,
               linestyle="--", label="50% breakeven")
    ax.set_title("Win Rate (% Profitable Trades) by Market Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for bar, val in zip(bars, win_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    save(fig, "03_win_rate_by_sentiment.png")


# ── Chart 4: PnL Violin ────────────────────────
if pnl_col:
    p2, p98 = merged["pnl"].quantile(0.02), merged["pnl"].quantile(0.98)
    plot_data = merged[merged["pnl"].between(p2, p98)]
    order = [s for s in existing_sentiments
             if s in plot_data["classification"].unique()]
    palette = {s: SENTIMENT_PALETTE.get(s, ACCENT) for s in order}

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(data=plot_data, x="classification", y="pnl",
                   order=order, palette=palette,
                   inner="quartile", linewidth=0.8, ax=ax)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_title("PnL Distribution by Market Sentiment (2nd-98th percentile)")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Closed PnL")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    save(fig, "04_pnl_distribution_violin.png")


# ── Chart 5: Long vs Short by Sentiment ───────
if side_col:
    side_sent = (merged.groupby(["classification", "side_clean"])
                       .size()
                       .unstack(fill_value=0)
                       .reindex(existing_sentiments)
                       .dropna())
    fig, ax = plt.subplots(figsize=(12, 6))
    side_colors = [GREED_COL, FEAR_COL, ACCENT, NEUTRAL_COL,
                   "#4fc3f7", "#ff8a65"][:len(side_sent.columns)]
    side_sent.plot(kind="bar", ax=ax, color=side_colors,
                   edgecolor=BORDER, linewidth=0.6)
    ax.set_title("Long vs Short Trade Count by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Trades")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="Side", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    save(fig, "05_long_short_by_sentiment.png")


# ── Chart 6: Daily PnL Timeline ───────────────
if pnl_col:
    daily_pnl = (merged.groupby("date")["pnl"]
                       .sum()
                       .reset_index()
                       .merge(fg[["date", "classification"]], on="date", how="left"))

    fig, ax = plt.subplots(figsize=(16, 6))

    # Sentiment background shading
    prev_date = daily_pnl["date"].iloc[0]
    prev_sent = daily_pnl["classification"].iloc[0]
    for _, row in daily_pnl.iterrows():
        if row["classification"] != prev_sent:
            ax.axvspan(prev_date, row["date"],
                       alpha=0.15,
                       color=SENTIMENT_PALETTE.get(prev_sent, DARK_BG),
                       linewidth=0)
            prev_date = row["date"]
            prev_sent = row["classification"]
    ax.axvspan(prev_date, daily_pnl["date"].iloc[-1],
               alpha=0.15,
               color=SENTIMENT_PALETTE.get(prev_sent, DARK_BG),
               linewidth=0)

    ax.plot(daily_pnl["date"], daily_pnl["pnl"],
            color=ACCENT, linewidth=1.2, label="Daily PnL")
    ax.fill_between(daily_pnl["date"], daily_pnl["pnl"], 0,
                    where=daily_pnl["pnl"] >= 0, alpha=0.3, color=GREED_COL)
    ax.fill_between(daily_pnl["date"], daily_pnl["pnl"], 0,
                    where=daily_pnl["pnl"] < 0, alpha=0.3, color=FEAR_COL)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_title("Daily Total PnL Over Time  (background = market sentiment)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total PnL")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "06_daily_pnl_timeline.png")


# ── Chart 7: Top Traders ───────────────────────
if pnl_col and acct_col:
    top_n = 15
    top_traders = (merged.groupby(acct_col)["pnl"]
                         .sum()
                         .sort_values(ascending=False)
                         .head(top_n))

    fig, ax = plt.subplots(figsize=(13, 7))
    bar_colors = [GREED_COL if v >= 0 else FEAR_COL for v in top_traders.values]
    ax.barh(top_traders.index.astype(str).str[:14],
            top_traders.values,
            color=bar_colors, edgecolor=BORDER, linewidth=0.6)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_title(f"Top {top_n} Traders by Total Realized PnL")
    ax.set_xlabel("Total PnL")
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    plt.tight_layout()
    save(fig, "07_top_traders_pnl.png")


# ── Chart 8: Leverage vs PnL ──────────────────
if pnl_col and lev_col:
    scatter_data = merged[[lev_col, "pnl", "classification"]].dropna()
    p2, p98 = scatter_data["pnl"].quantile(0.02), scatter_data["pnl"].quantile(0.98)
    scatter_data = scatter_data[scatter_data["pnl"].between(p2, p98)]

    fig, ax = plt.subplots(figsize=(11, 7))
    for sent in existing_sentiments:
        sub = scatter_data[scatter_data["classification"] == sent]
        ax.scatter(sub[lev_col], sub["pnl"],
                   color=SENTIMENT_PALETTE.get(sent, ACCENT),
                   alpha=0.35, s=15, label=sent)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_title("Leverage vs PnL coloured by Market Sentiment")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Closed PnL")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1))
    ax.yaxis.grid(True, alpha=0.4)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "08_leverage_vs_pnl.png")


# ── Chart 9: Trade Volume Timeline ────────────
daily_vol = (merged.groupby(["date", "classification"])
                   .size()
                   .reset_index(name="count"))

fig, ax = plt.subplots(figsize=(16, 6))
for sent in existing_sentiments:
    sub = daily_vol[daily_vol["classification"] == sent]
    ax.plot(sub["date"], sub["count"],
            color=SENTIMENT_PALETTE.get(sent, ACCENT),
            label=sent, linewidth=1.2, alpha=0.85)
ax.set_title("Daily Trade Volume by Market Sentiment")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Trades")
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
save(fig, "09_trade_volume_timeline.png")


# ── Chart 10: Symbol x Sentiment Heatmap ──────
if pnl_col and sym_col:
    top_syms = merged[sym_col].value_counts().head(10).index
    hmap = (merged[merged[sym_col].isin(top_syms)]
            .groupby([sym_col, "classification"])["pnl"]
            .mean()
            .unstack()
            .reindex(columns=existing_sentiments))

    if not hmap.empty:
        fig, ax = plt.subplots(figsize=(13, 7))
        sns.heatmap(hmap, annot=True, fmt=".1f",
                    cmap="RdYlGn", center=0,
                    linewidths=0.5, linecolor=BORDER, ax=ax,
                    cbar_kws={"label": "Avg PnL"})
        ax.set_title("Avg PnL Heatmap: Top Symbols x Sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Symbol")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        plt.tight_layout()
        save(fig, "10_symbol_sentiment_heatmap.png")


# ───────────────────────────────────────────────
# STATISTICAL TESTS
# ───────────────────────────────────────────────
print("\n" + "="*55)
print("  STATISTICAL ANALYSIS")
print("="*55)

if pnl_col:
    fear_pnl  = merged[merged["broad_sentiment"] == "Fear"]["pnl"].dropna()
    greed_pnl = merged[merged["broad_sentiment"] == "Greed"]["pnl"].dropna()

    if len(fear_pnl) > 1 and len(greed_pnl) > 1:
        t_stat, p_val = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
        print(f"\n  Welch T-test: Fear PnL vs Greed PnL")
        print(f"    t-statistic : {t_stat:.4f}")
        print(f"    p-value     : {p_val:.4f}")
        sig = "Significant" if p_val < 0.05 else "Not significant"
        print(f"    Result      : {sig} at p < 0.05")

    if "fg_value" in merged.columns:
        daily_corr = merged.groupby("date").agg(
            total_pnl=("pnl", "sum"),
            fg_value=("fg_value", "first")
        ).dropna()
        if len(daily_corr) > 2:
            r, p = stats.pearsonr(daily_corr["fg_value"], daily_corr["total_pnl"])
            print(f"\n  Pearson r (F&G score vs daily PnL): {r:.4f}  (p={p:.4f})")


# ───────────────────────────────────────────────
# SUMMARY
# ───────────────────────────────────────────────
print("\n" + "="*55)
print("  KEY INSIGHTS SUMMARY")
print("="*55)

if pnl_col:
    best  = pnl_stats["mean"].idxmax()
    worst = pnl_stats["mean"].idxmin()
    print(f"\n  Best avg PnL sentiment  : {best}  (${pnl_stats.loc[best,  'mean']:,.2f})")
    print(f"  Worst avg PnL sentiment : {worst} (${pnl_stats.loc[worst, 'mean']:,.2f})")
    print(f"\n  Highest win rate        : {win_rate.idxmax()}  ({win_rate.max():.1f}%)")
    print(f"  Lowest win rate         : {win_rate.idxmin()}  ({win_rate.min():.1f}%)")
    print(f"\n  Total trades analyzed   : {len(merged):,}")
    print(f"  Overall total PnL       : ${merged['pnl'].sum():,.2f}")
    print(f"  Overall win rate        : {merged['profitable'].mean()*100:.1f}%")

print(f"\n  All charts saved to -> ./{OUTPUT_DIR}/")
print("\n" + "="*55)
print("  ANALYSIS COMPLETE")
print("="*55 + "\n")
