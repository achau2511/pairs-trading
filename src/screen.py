"""
Sector cointegration screener.

Tests every unique pair in a ticker universe for Engle-Granger cointegration
over a fixed formation period, then backtests the p < 0.05 pairs out of sample
using parameters frozen from the formation period.

Usage:
    python screen.py                      # uses TICKERS below
    python screen.py XOM CVX COP EOG ...  # overrides TICKERS
    python screen.py XOM CVX COP --out results.csv
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from data import load_prices
from signals import hedge_ratio, generate_signals
from backtest import backtest

# ----------------------------------
# Config
# ----------------------------------

TICKERS = []  # [PASTE TICKERS HERE]

FORMATION_START = "2015-01-01"
FORMATION_END = "2019-12-31"
TEST_START = "2020-01-01"

ALPHA = 0.05


# ----------------------------------
# Data
# ----------------------------------

def load_universe(tickers):
    """
    Download each ticker separately via load_prices.

    load_prices() drops any row with a NaN in *any* column, so downloading the
    whole universe at once would let one late-listing ticker silently truncate
    every other ticker's history. Per-ticker downloads avoid that.
    """
    series = {}
    failed = []
    for t in tickers:
        try:
            px = load_prices([t], start=FORMATION_START)
        except Exception as e:
            print(f"  {t}: download error ({e})")
            failed.append(t)
            continue
        if px.empty or t not in px.columns:
            failed.append(t)
            continue
        series[t] = px[t]

    prices = pd.DataFrame(series).sort_index()
    return prices, failed


def check_formation_coverage(prices):
    """
    A ticker has "full history" if it has a price on every trading day in the
    formation window, where the trading calendar is the union of dates
    observed across the universe.
    """
    form = prices.loc[FORMATION_START:FORMATION_END]
    missing = form.isna().sum()
    keep = missing[missing == 0].index.tolist()
    dropped = {}
    for t in missing[missing > 0].index:
        first = form[t].first_valid_index()
        dropped[t] = (int(missing[t]), first.date() if first is not None else None)
    return keep, dropped, form.index


# ----------------------------------
# Screen
# ----------------------------------

def screen_pairs(form_logpx):
    rows = []
    for t1, t2 in combinations(sorted(form_logpx.columns), 2):
        y = form_logpx[t1]
        x = form_logpx[t2]

        # signals.cointegration_test() only returns the p-value; the test
        # statistic is also required, so call statsmodels' coint directly
        # (it's the same call cointegration_test wraps).
        stat, pvalue, _ = coint(y, x)
        beta = hedge_ratio(y, x)

        spread = y - beta * x
        rows.append({
            "y": t1,
            "x": t2,
            "pvalue": float(pvalue),
            "test_stat": float(stat),
            "beta": float(beta),
            "spread_mean": float(spread.mean()),
            "spread_std": float(spread.std()),
        })
    return pd.DataFrame(rows)


# ----------------------------------
# Out-of-sample backtest (frozen params)
# ----------------------------------

def oos_backtest(test_logpx, row, cost_bps):
    """
    Spread, z-score, and signals on test data use only beta / mean / std
    estimated on the formation period. Nothing is re-estimated here.
    """
    pair = test_logpx[[row["y"], row["x"]]].dropna()
    if len(pair) < 2:
        return np.nan, np.nan, 0, 0.0, 0

    spread = pair[row["y"]] - row["beta"] * pair[row["x"]]
    z = (spread - row["spread_mean"]) / row["spread_std"]
    position = generate_signals(z)

    _, gross = backtest(spread, position, cost_bps=0)
    _, net = backtest(spread, position, cost_bps=cost_bps)
    n_trades = int((position.diff().fillna(0) != 0).sum())
    return float(gross), float(net), len(pair), float((position != 0).mean()), n_trades


# ----------------------------------
# Main
# ----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Screen a ticker universe for cointegrated pairs.")
    parser.add_argument("tickers", nargs="*", help="Ticker universe (overrides TICKERS in screen.py)")
    parser.add_argument("--out", default="screen_results.csv", help="Output CSV path")
    parser.add_argument("--cost-bps", type=float, default=5, help="Transaction cost per unit position change, in bps")
    args = parser.parse_args()

    tickers = [t.upper() for t in (args.tickers or TICKERS)]
    tickers = list(dict.fromkeys(tickers))  # dedupe, keep order
    if len(tickers) < 2:
        print("Need at least 2 tickers. Edit TICKERS in screen.py or pass them on the command line.")
        sys.exit(1)

    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Formation: {FORMATION_START} -> {FORMATION_END}   Test: {TEST_START} -> latest\n")

    # --- Load + coverage check ---
    prices, failed = load_universe(tickers)
    keep, dropped, form_dates = check_formation_coverage(prices) if not prices.empty else ([], {}, [])

    print(f"\nFormation trading days: {len(form_dates)}")
    if failed or dropped:
        print("\nDropped tickers (incomplete formation-period history):")
        for t in failed:
            print(f"  {t}: no data returned")
        for t, (n_missing, first) in dropped.items():
            print(f"  {t}: {n_missing} missing days (first formation-period price: {first})")
    else:
        print("\nNo tickers dropped.")

    if len(keep) < 2:
        print("\nFewer than 2 tickers with full history; nothing to screen.")
        sys.exit(1)

    logpx = np.log(prices[keep])
    form_logpx = logpx.loc[FORMATION_START:FORMATION_END]
    test_logpx = logpx.loc[TEST_START:]

    # Hard guard: formation data must end on or before FORMATION_END
    assert form_logpx.index.max() <= pd.Timestamp(FORMATION_END)

    # --- Screen ---
    results = screen_pairs(form_logpx)
    n_pairs = len(results)
    bonferroni = ALPHA / n_pairs

    # --- Out-of-sample backtest for p < ALPHA ---
    oos_cols = ["oos_sharpe_gross", "oos_sharpe_net", "oos_days", "oos_frac_in_market", "oos_trades"]
    results[oos_cols] = np.nan
    sig = results["pvalue"] < ALPHA
    for i in results.index[sig]:
        results.loc[i, oos_cols] = oos_backtest(test_logpx, results.loc[i], args.cost_bps)
    results[["oos_days", "oos_trades"]] = results[["oos_days", "oos_trades"]].astype("Int64")

    results["bonferroni_sig"] = results["pvalue"] < bonferroni
    results = results.sort_values("pvalue").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results.to_csv(args.out, index=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tickers screened:                       {len(keep)}")
    print(f"Total pairs tested:                     {n_pairs}")
    print(f"Pairs with p < {ALPHA}:                    {int(sig.sum())}")
    print(f"Bonferroni threshold ({ALPHA}/{n_pairs}):      {bonferroni:.2e}")
    print(f"Pairs surviving Bonferroni:             {int(results['bonferroni_sig'].sum())}")
    print(f"Expected false positives at p < {ALPHA}:   {ALPHA * n_pairs:.1f}")

    # --- p-value == 1.0 flags ---
    ones = results[results["pvalue"] == 1.0]
    nans = results[results["pvalue"].isna()]
    if not ones.empty:
        print(f"\nWARNING: {len(ones)} pair(s) returned p-value == 1.0. Raw test statistics:")
        for _, r in ones.iterrows():
            print(f"  {r['y']}-{r['x']}: test_stat = {r['test_stat']!r}")
    if not nans.empty:
        print(f"\nWARNING: {len(nans)} pair(s) returned NaN p-values:")
        for _, r in nans.iterrows():
            print(f"  {r['y']}-{r['x']}: test_stat = {r['test_stat']!r}")

    # --- OOS results ---
    if sig.any():
        oos = results[results["pvalue"] < ALPHA]
        print(f"\nOut-of-sample ({TEST_START} -> {test_logpx.index.max().date()}), frozen formation params,"
              f" costs = {args.cost_bps:g} bps:")
        print(f"  {'pair':<14}{'p-value':>10}{'beta':>9}{'Sharpe gross':>14}{'Sharpe net':>12}"
              f"{'trades':>8}{'in mkt':>9}{'Bonf':>6}")
        for _, r in oos.iterrows():
            print(f"  {r['y'] + '-' + r['x']:<14}{r['pvalue']:>10.4f}{r['beta']:>9.3f}"
                  f"{r['oos_sharpe_gross']:>14.2f}{r['oos_sharpe_net']:>12.2f}{int(r['oos_trades']):>8}"
                  f"{r['oos_frac_in_market']:>9.1%}{'*' if r['bonferroni_sig'] else '':>6}")
    else:
        print(f"\nNo pairs with p < {ALPHA}; no out-of-sample backtests run.")

    print(f"\nFull results written to {args.out}\n")


if __name__ == "__main__":
    main()
