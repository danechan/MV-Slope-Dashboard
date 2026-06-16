"""
MA + Slope Dashboard — Streamlit
--------------------------------
Run with:
    streamlit run ma_slope_app.py

Features:
    - Sidebar config: ticker, date range (presets + custom), MA periods, slope window
    - Interactive Plotly charts (zoom/pan/hover)
    - Cached yfinance pulls — config changes don't re-download
    - 20-year history supported out of the box

Slope definition:
    slope_t = (SMA_t / SMA_{t-N} - 1) * 100 / N
    Units: % per day, comparable across MAs and tickers.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta, date

# ----------------------------- page setup -----------------------------
st.set_page_config(
    page_title="MA & Slope Dashboard",
    page_icon="📈",
    layout="wide",
)

# ----------------------------- sidebar --------------------------------
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Ticker", value="QQQ").strip().upper()

preset = st.sidebar.selectbox(
    "Date range",
    ["1Y", "3Y", "5Y", "10Y", "20Y", "Custom"],
    index=2,
)

today = date.today()
preset_years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "20Y": 20}

if preset == "Custom":
    col_a, col_b = st.sidebar.columns(2)
    start_date = col_a.date_input(
        "Start", value=today - timedelta(days=365 * 5),
        max_value=today - timedelta(days=30),
    )
    end_date = col_b.date_input("End", value=today, max_value=today)
else:
    years = preset_years[preset]
    start_date = today - timedelta(days=int(years * 365.25))
    end_date = today

ma_input = st.sidebar.text_input("MA periods (comma-separated days)", value="20, 50, 200")
try:
    ma_periods = sorted({int(x.strip()) for x in ma_input.split(",") if x.strip()})
    if not ma_periods or any(p <= 0 for p in ma_periods):
        raise ValueError
except ValueError:
    st.sidebar.error("Invalid MA periods. Use positive integers separated by commas.")
    st.stop()

slope_window = st.sidebar.slider(
    "Slope lookback (trading days)",
    min_value=1, max_value=30, value=5,
    help="Slope = (SMA_t / SMA_{t-N} − 1) × 100 / N, units % per day. "
         "Smaller = noisier/faster; larger = smoother/slower.",
)

show_volume = st.sidebar.checkbox("Show volume on price chart", value=False)
log_scale   = st.sidebar.checkbox("Log scale on price (useful for 10-20Y)", value=False)

# ----------------------------- data fetch -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol: str, start: date, end: date, buffer_days: int) -> pd.DataFrame:
    """Pull with buffer so the longest MA is fully formed at the display start."""
    fetch_start = start - timedelta(days=buffer_days)
    df = yf.download(
        symbol, start=fetch_start, end=end + timedelta(days=1),
        auto_adjust=False, progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

buffer_days = max(ma_periods) * 2 + 30

with st.spinner(f"Fetching {ticker}..."):
    try:
        df = fetch_data(ticker, start_date, end_date, buffer_days)
    except Exception as e:
        st.error(f"Failed to fetch {ticker}: {e}")
        st.stop()

if df.empty:
    st.error(f"No data returned for {ticker}. Check the symbol and date range.")
    st.stop()

# ----------------------------- compute --------------------------------
close  = df["Close"]
smas   = {p: close.rolling(p).mean() for p in ma_periods}
slopes = {p: (smas[p] / smas[p].shift(slope_window) - 1) * 100 / slope_window
          for p in ma_periods}

mask     = df.index >= pd.Timestamp(start_date)
df_d     = df.loc[mask]
smas_d   = {p: s.loc[mask] for p, s in smas.items()}
slopes_d = {p: s.loc[mask] for p, s in slopes.items()}

if df_d.empty:
    st.error("No data in selected display window. Try a wider date range.")
    st.stop()

# ----------------------------- header & metrics -----------------------
st.title(f"{ticker} — Moving Averages & Slope Dashboard")
st.caption(f"Range: {df_d.index[0].date()} → {df_d.index[-1].date()}   "
           f"({len(df_d)} trading days)")

last_row  = df_d.iloc[-1]
last_date = df_d.index[-1].date()

# Row 1: close + MA values
cols = st.columns(1 + len(ma_periods))
cols[0].metric("Close", f"${last_row['Close']:.2f}", f"as of {last_date}")
for i, p in enumerate(ma_periods):
    sma_val  = smas_d[p].iloc[-1]
    px_to_ma = (last_row["Close"] / sma_val - 1) * 100
    cols[i + 1].metric(
        f"{p}DMA", f"${sma_val:.2f}", f"{px_to_ma:+.2f}% vs close",
    )

# Row 2: slope values with percentile
slope_cols = st.columns(len(ma_periods))
for i, p in enumerate(ma_periods):
    s = slopes_d[p].dropna()
    cur = s.iloc[-1]
    pctile = (s < cur).mean() * 100
    slope_cols[i].metric(
        f"{p}DMA slope (%/day)",
        f"{cur:+.3f}",
        f"{pctile:.0f}th pctile of window",
        delta_color="off",
    )

# ----------------------------- palette --------------------------------
palette_base = px.colors.qualitative.Set1
ma_colors = {p: palette_base[i % len(palette_base)] for i, p in enumerate(ma_periods)}
canonical = {20: "#1f77b4", 50: "#ff7f0e", 200: "#d62728"}
for p in ma_periods:
    if p in canonical:
        ma_colors[p] = canonical[p]

# ============================ chart 1: price + MAs ====================
st.subheader("Price & Moving Averages")

if show_volume:
    fig1 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )
    fig1.add_trace(
        go.Scatter(x=df_d.index, y=df_d["Close"], name=f"{ticker} Close",
                   line=dict(color="black", width=1.3)),
        row=1, col=1,
    )
    for p in ma_periods:
        fig1.add_trace(
            go.Scatter(x=smas_d[p].index, y=smas_d[p], name=f"{p}DMA",
                       line=dict(color=ma_colors[p], width=1.5)),
            row=1, col=1,
        )
    fig1.add_trace(
        go.Bar(x=df_d.index, y=df_d["Volume"], name="Volume",
               marker_color="lightgray", showlegend=False),
        row=2, col=1,
    )
    fig1.update_yaxes(title_text="Price", type="log" if log_scale else "linear", row=1, col=1)
    fig1.update_yaxes(title_text="Volume", row=2, col=1)
else:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_d.index, y=df_d["Close"], name=f"{ticker} Close",
        line=dict(color="black", width=1.3),
    ))
    for p in ma_periods:
        fig1.add_trace(go.Scatter(
            x=smas_d[p].index, y=smas_d[p], name=f"{p}DMA",
            line=dict(color=ma_colors[p], width=1.5),
        ))
    fig1.update_yaxes(title_text="Price", type="log" if log_scale else "linear")

fig1.update_layout(
    height=560 if show_volume else 480,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig1, use_container_width=True)

# ============================ chart 2: slopes =========================
st.subheader(f"SMA Slopes (% per day, {slope_window}-day lookback)")

fig2 = go.Figure()
for p in ma_periods:
    fig2.add_trace(go.Scatter(
        x=slopes_d[p].index, y=slopes_d[p], name=f"{p}DMA slope",
        line=dict(color=ma_colors[p], width=1.5),
    ))
fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.5)

fig2.update_layout(
    height=460,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="Slope (% per day)",
)
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------- summary tables -------------------------
with st.expander("Slope statistics for selected window"):
    stats = []
    for p in ma_periods:
        s = slopes_d[p].dropna()
        stats.append({
            "MA": f"{p}DMA",
            "Current": round(s.iloc[-1], 3),
            "Mean": round(s.mean(), 3),
            "Std": round(s.std(), 3),
            "Min": round(s.min(), 3),
            "Min date": s.idxmin().date(),
            "Max": round(s.max(), 3),
            "% days negative": f"{(s < 0).mean() * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(stats).set_index("MA"), use_container_width=True)

with st.expander("Raw data (last 30 rows)"):
    out = df_d[["Open", "High", "Low", "Close", "Volume"]].copy()
    for p in ma_periods:
        out[f"SMA{p}"]   = smas_d[p].round(2)
        out[f"Slope{p}"] = slopes_d[p].round(3)
    st.dataframe(out.tail(30), use_container_width=True)
