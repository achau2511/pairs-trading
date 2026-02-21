import plotly.graph_objects as go


def make_equity_curve(equity, title, width=900, height=520):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (spread units)",
        width=int(width),
        height=int(height),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig