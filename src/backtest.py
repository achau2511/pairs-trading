import numpy as np

def backtest(spread, position, cost_bps=5):
    returns = spread.diff()

    # Charged on each change in position (a +1 -> -1 flip costs 2x an entry)
    costs = cost_bps / 10000 * position.diff().abs()

    pnl = position.shift(1) * returns - costs

    equity = pnl.cumsum()

    sharpe = np.sqrt(252) * pnl.mean() / pnl.std()

    return equity, sharpe
