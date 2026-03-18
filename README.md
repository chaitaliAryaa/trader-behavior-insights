# Trader Behavior Insights × Bitcoin Fear & Greed Index
### Junior Data Scientist Assignment — PrimeTrade.ai

---

## Overview

This project explores the relationship between **crypto trader performance** on Hyperliquid and the **Bitcoin Fear & Greed Index**. The goal is to uncover hidden patterns in trader behavior across different market sentiment regimes and deliver insights that can drive smarter trading strategies.

---

## Datasets

| Dataset | Source | Rows |
|---|---|---|
| Historical Trader Data (Hyperliquid) | Google Drive | 211,224 trades |
| Bitcoin Fear & Greed Index | Google Drive | 2,644 days |

**Trader Data columns:** Account, Coin, Execution Price, Size Tokens, Size USD, Side, Timestamp, Start Position, Direction, Closed PnL, Fee, Trade ID

**Fear & Greed columns:** Date, Value (0–100), Classification (Extreme Fear → Extreme Greed)

---

## Key Questions Answered

- Do traders make more profit during **Fear** or **Greed** periods?
- Does market sentiment affect **win rate**?
- How does **leverage usage** change across sentiment regimes?
- Which **coins/symbols** perform best under each sentiment?
- Who are the **top performing traders** and when do they trade?
- Is there a **statistically significant** difference in PnL between Fear and Greed markets?

---

## Charts Generated

| # | Chart | Description |
|---|---|---|
| 01 | Sentiment Distribution | Trade count and proportion across all sentiment categories |
| 02 | PnL by Sentiment | Avg, Median and Total PnL for each sentiment |
| 03 | Win Rate by Sentiment | % of profitable trades per sentiment category |
| 04 | PnL Distribution Violin | Full PnL spread and quartiles per sentiment |
| 05 | Long vs Short by Sentiment | Buy/Sell trade volume across sentiment periods |
| 06 | Daily PnL Timeline | Total daily PnL over time with sentiment background shading |
| 07 | Top 15 Traders | Highest total realized PnL accounts |
| 08 | Leverage vs PnL | Scatter of leverage against PnL coloured by sentiment |
| 09 | Trade Volume Timeline | Daily trade count broken down by sentiment |
| 10 | Symbol × Sentiment Heatmap | Avg PnL for top 10 coins across each sentiment |

---

## Key Findings

- **Win rates and average PnL differ meaningfully across sentiment regimes** — traders tend to perform differently in Fear vs Greed markets
- **Trade volume spikes** during Extreme Fear and Extreme Greed periods, suggesting sentiment extremes drive activity
- **Top traders** show consistent PnL regardless of sentiment, indicating systematic strategies
- **Statistical t-test** confirms whether the PnL difference between Fear and Greed periods is significant
- Certain **coins outperform** consistently in Greed periods while others show resilience in Fear

---

## How to Run

**1. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scipy
```

**2. Place both CSV files in the same folder as `analysis.py`**
```
trading/
├── analysis.py
├── historical_data.csv
└── fear_greed_index.csv
```

**3. Run the script**
```bash
python analysis.py
```

**4. View output**

All 10 charts are saved to the `output_charts/` folder.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas | Data loading, cleaning, merging |
| NumPy | Numerical operations |
| Matplotlib | Chart generation |
| Seaborn | Statistical visualizations |
| SciPy | T-test and correlation analysis |

---

## Project Structure

```
trader-behavior-insights/
├── analysis.py          # Main analysis script
├── README.md            # This file
└── output_charts/
    ├── 01_sentiment_distribution.png
    ├── 02_pnl_by_sentiment.png
    ├── 03_win_rate_by_sentiment.png
    ├── 04_pnl_distribution_violin.png
    ├── 05_long_short_by_sentiment.png
    ├── 06_daily_pnl_timeline.png
    ├── 07_top_traders_pnl.png
    ├── 08_leverage_vs_pnl.png
    ├── 09_trade_volume_timeline.png
    └── 10_symbol_sentiment_heatmap.png
```

---

*Submitted as part of the Junior Data Scientist hiring process at PrimeTrade.ai*
