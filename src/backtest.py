import numpy as np

def backtest(spread, position):
    returns = spread.diff()

    pnl = position.shift(1) * returns

    equity = pnl.cumsum()

    sharpe = np.sqrt(252) * pnl.mean() / pnl.std()

    return equity, sharpe