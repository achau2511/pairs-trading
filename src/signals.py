import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint


def hedge_ratio(y, x):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model.params.iloc[1]

def compute_zscore(spread, window=60):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z = (spread - mean) / std
    return z

def generate_signals(z):
    pos = pd.Series(0, index=z.index)

    pos[z > 2] = -1
    pos[z < -2] = 1
    pos[abs(z) < 0.5] = 0

    return pos.ffill()

def cointegration_test(y, x):
    score, pvalue, _ = coint(y, x)
    return pvalue