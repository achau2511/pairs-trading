# Pairs Trading (Statistical Arbitrage) Project

This project is a small research tool for testing equity pairs trading strategies in Python.

It implements:

- Regression-based hedge ratio (beta)
- Engle–Granger cointegration test
- Z-score mean reversion signals
- Simple backtest
- Equity curve visualization
- Sharpe ratio calculation

---

## How It Works

1. Download historical price data using `yfinance`
2. Test if two stocks are cointegrated
3. Estimate hedge ratio using linear regression
4. Construct the spread
5. Generate z-score trading signals
6. Backtest the strategy
7. Plot the equity curve

---

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install yfinance pandas numpy statsmodels matplotlib
```

---

## How to Run

From the project root:

```bash
cd src
python main.py TICKER1 TICKER2
```

Examples:

```bash
python main.py V MA
python main.py KO PEP
python main.py XOM CVX
python main.py JPM BAC
```

---

## Output

The program prints:

- Number of rows and date range
- Cointegration p-value
- Hedge ratio (beta)
- Sharpe ratio

It also displays an equity curve plot.

---

## Interpretation

- Cointegration p-value < 0.05 suggests statistical validity.
- Higher Sharpe ratio indicates better risk-adjusted performance.
- Upward sloping equity curve suggests positive strategy edge.
- Large drawdowns indicate instability.

---

## Limitations

This is a research prototype and does NOT include:

- Transaction costs
- Slippage
- Borrow fees
- Capital-based position sizing
- Risk controls

---

## Future Improvements

Potential upgrades:

- Rolling hedge ratio
- Rolling cointegration testing
- Transaction cost modeling
- Dollar-neutral position sizing
- Automatic pair scanning across multiple stocks
- Half-life estimation of mean reversion