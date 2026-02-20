import yfinance as yf
import pandas as pd

def load_prices(tickers, start="2018-01-01"):
    df = yf.download(tickers, start=start, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            prices = df["Adj Close"]
        else:
            prices = df["Close"]
    else:
        prices = df

    prices = prices.dropna()
    return prices