# 📊 Pairs Trading (Statistical Arbitrage) Project

An interactive statistical arbitrage research tool built in Python.

This project implements a pairs trading framework using cointegration testing,
regression-based hedge ratios, and mean-reversion signals. It includes both a
command-line research interface and an interactive Panel dashboard.

---

## ✨ Features

- Engle–Granger cointegration test
- OLS hedge ratio (beta) estimation
- Z-score mean reversion signals
- Backtested equity curve
- Sharpe ratio calculation
- Interactive dashboard (HoloViz Panel)
- Adjustable tickers, date range, and plot size
- Plot + Data tabs

---

## 🧠 Strategy Overview

1. Download historical price data using `yfinance`
2. Test whether two equities are cointegrated
3. Estimate hedge ratio via linear regression
4. Construct the spread
5. Generate z-score trading signals
6. Backtest the strategy
7. Visualize equity curve and underlying data

---

## 📁 Project Structure

```
data.py         # Price downloading
signals.py      # Hedge ratio, z-score, cointegration
backtest.py     # PnL + Sharpe calculation
plots.py        # Plotly equity curve visualization
main.py         # CLI research tool
app.py          # Interactive Panel dashboard
requirements.txt
```

---

## 🚀 Installation

Install required packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy yfinance statsmodels plotly panel
```

---

## ▶ Run (CLI Version)

Quick research test from terminal:

```bash
python main.py V MA
```

Example pairs:

```bash
python main.py KO PEP
python main.py XOM CVX
python main.py JPM BAC
```

The CLI prints:

- Date range
- Cointegration p-value
- Beta
- Sharpe ratio
- Displays equity curve

---

## 🖥 Interactive Dashboard

Run the dashboard:

```bash
python app.py
```

The dashboard allows users to:

- Input custom ticker pairs
- Select start and end dates
- Toggle log prices
- Adjust plot width and height
- View Plot and Data tabs

---

## 📊 Interpretation

- **Cointegration p-value < 0.05** → strong statistical evidence of mean reversion  
- **Higher Sharpe ratio** → better risk-adjusted performance  
- **Upward equity curve** → positive strategy edge  
- Large drawdowns → instability  

---

## ⚠ Limitations

This is a research prototype and does NOT include:

- Transaction costs
- Slippage
- Borrow fees
- Capital-based position sizing
- Risk management constraints

---

## 🔮 Future Improvements

- Rolling hedge ratio
- Rolling cointegration testing
- Transaction cost modeling
- Dollar-neutral position sizing
- Automated pair scanning across a universe
- Half-life estimation of mean reversion

---

## 📌 Key Insight

Economic similarity alone does not guarantee statistical mean reversion.
Pairs must be validated using formal cointegration testing before deployment.