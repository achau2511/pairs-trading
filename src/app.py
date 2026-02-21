"""
UI Layer
Pairs Trading Dashboard (Panel)
Goals:
- User edits tickers -> updates equity curve + table
- User edits start/end dates -> updates equity curve + table
- User changes plot width/height -> updates plot
- Tabs to switch between Plot and Data
- Built with HoloViz Panel + Plotly
"""

import panel as pn
import pandas as pd
import numpy as np

from data import load_prices
from signals import hedge_ratio, compute_zscore, generate_signals, cointegration_test
from backtest import backtest
from plots import make_equity_curve

# DIMENSIONS
CARD_WIDTH = 320

# Simple cache so Table + Plot don't re-download/recompute twice every time
_CACHE = {
    "key": None,
    "out": None,
    "metrics_md": None,
}

def _compute_pair(t1, t2, start, end, use_log):
    """
    Computes everything once and returns:
      out: DataFrame with labeled columns
      metrics_md: Markdown string summary
    """

    prices = load_prices([t1, t2], start=str(start))
    prices = prices.loc[prices.index <= pd.Timestamp(end)]

    if prices.empty:
        raise ValueError("No data returned for that date range.")

    if t1 not in prices.columns or t2 not in prices.columns:
        raise ValueError(f"Missing ticker data. Got columns: {list(prices.columns)}")

    # Align by date
    px = prices[[t1, t2]].dropna()

    y_raw = px[t1]
    x_raw = px[t2]

    y = np.log(y_raw) if use_log else y_raw
    x = np.log(x_raw) if use_log else x_raw

    pval = float(cointegration_test(y, x))
    beta = float(hedge_ratio(y, x))

    spread = y - beta * x
    z = compute_zscore(spread)
    position = generate_signals(z)
    equity, sharpe = backtest(spread, position)

    out = pd.DataFrame({
        "date": px.index,
        f"{t1}_price": y_raw.values,
        f"{t2}_price": x_raw.values,
        f"{t1}_used": y.values,
        f"{t2}_used": x.values,
        "spread": spread.values,
        "z_score": z.values,
        "position": position.values,
        "equity": equity.values,
    }).dropna()

    metrics_md = f"""
### Results
- **Pair:** `{t1}` vs `{t2}`
- **Rows:** {len(out):,}
- **Date range:** {out["date"].min().date()} → {out["date"].max().date()}
- **Cointegration p-value:** `{pval:.6f}`
- **Beta:** `{beta:.6f}`
- **Sharpe (toy):** `{sharpe:.3f}`
"""

    return out, metrics_md


def _get_cached(t1, t2, start, end, use_log):
    """
    Ensures Plot + Table share the same computed result.
    """
    key = (t1, t2, str(start), str(end), bool(use_log))
    if _CACHE["key"] != key:
        out, metrics_md = _compute_pair(t1, t2, start, end, use_log)
        _CACHE["key"] = key
        _CACHE["out"] = out
        _CACHE["metrics_md"] = metrics_md
    return _CACHE["out"], _CACHE["metrics_md"]


# CALLBACK FUNCTIONS (pn.bind)

def get_table(ticker1, ticker2, start, end, use_log):
    t1 = str(ticker1).strip().upper()
    t2 = str(ticker2).strip().upper()

    out, _ = _get_cached(t1, t2, start, end, use_log)

    # show clearly labeled columns
    return pn.pane.DataFrame(out, sizing_mode="stretch_width", height=520)


def get_plot(ticker1, ticker2, start, end, use_log, width, height):
    t1 = str(ticker1).strip().upper()
    t2 = str(ticker2).strip().upper()

    out, metrics_md = _get_cached(t1, t2, start, end, use_log)

    # rebuild equity series for plotting
    equity = pd.Series(out["equity"].values, index=pd.to_datetime(out["date"]))

    metrics = pn.pane.Markdown(metrics_md)
    fig = make_equity_curve(equity, title=f"Equity Curve: {t1}–{t2}", width=width, height=height)

    return pn.Column(metrics, pn.pane.Plotly(fig), sizing_mode="stretch_width")


def main():
    pn.extension("plotly")

    # WIDGETS

    t1_in = pn.widgets.TextInput(name="Ticker 1", value="KO")
    t2_in = pn.widgets.TextInput(name="Ticker 2", value="PEP")

    start_picker = pn.widgets.DatePicker(name="Start date", value=pd.Timestamp("2018-01-01").date())
    end_picker = pn.widgets.DatePicker(name="End date", value=pd.Timestamp.today().date())

    use_log_chk = pn.widgets.Checkbox(name="Use log prices", value=True)

    width_sldr = pn.widgets.IntSlider(name="Width", start=700, end=1600, step=50, value=950)
    height_sldr = pn.widgets.IntSlider(name="Height", start=350, end=950, step=50, value=520)


    # CALLBACK BINDINGS (Connecting widgets to callback functions)

    table = pn.bind(get_table, t1_in, t2_in, start_picker, end_picker, use_log_chk)

    plot = pn.bind(
        get_plot,
        t1_in,
        t2_in,
        start_picker,
        end_picker,
        use_log_chk,
        width_sldr.param.value_throttled,
        height_sldr.param.value_throttled
    )

    # DASHBOARD WIDGET CONTAINERS ("CARDS")

    search_card = pn.Card(
        pn.Column(
            t1_in,
            t2_in,
            start_picker,
            end_picker,
            use_log_chk,
        ),
        title="Search",
        width=CARD_WIDTH,
        collapsed=False
    )

    plot_card = pn.Card(
        pn.Column(
            width_sldr,
            height_sldr
        ),
        title="Plot",
        width=CARD_WIDTH,
        collapsed=True
    )

    # LAYOUT 

    layout = pn.template.FastListTemplate(
        title="Pairs Trading Dashboard",
        sidebar=[search_card, plot_card],
        theme_toggle=False,
        main=[
            pn.Tabs(
                ("Plot", plot),
                ("Data", table),
                active=0
            )
        ],
        header_background="#1f2937",
        header_color="white"
    ).servable()

    layout.show()


if __name__ == "__main__":
    main()