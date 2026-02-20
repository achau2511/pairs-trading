import sys
import matplotlib.pyplot as plt
import numpy as np
from data import load_prices
from signals import hedge_ratio, compute_zscore, generate_signals, cointegration_test
from backtest import backtest

# ----------------------------------
# Read tickers from command line
# ----------------------------------

if len(sys.argv) != 3:
    print("Usage: python main.py TICKER1 TICKER2")
    sys.exit()

ticker1 = sys.argv[1].upper()
ticker2 = sys.argv[2].upper()

print(f"\nRunning pairs trading on: {ticker1} & {ticker2}\n")

# ----------------------------------
# Load data
# ----------------------------------

data = load_prices([ticker1, ticker2])

y = data[ticker1]
x = data[ticker2]

y = np.log(y)
x = np.log(x)

print("Rows:", len(data), "Start:", data.index.min().date(), "End:", data.index.max().date())

# ----------------------------------
# Cointegration test
# ----------------------------------

pvalue = cointegration_test(y, x)
print("Cointegration p-value:", pvalue)

# ----------------------------------
# Hedge ratio + spread
# ----------------------------------

beta = hedge_ratio(y, x)
print("Beta:", beta)

spread = y - beta * x

# ----------------------------------
# Signals + Backtest
# ----------------------------------

z = compute_zscore(spread)
position = generate_signals(z)

equity, sharpe = backtest(spread, position)

print("Sharpe:", sharpe)

# ----------------------------------
# Plot equity curve
# ----------------------------------

equity.plot(title=f"Equity Curve: {ticker1}-{ticker2}")
plt.show()