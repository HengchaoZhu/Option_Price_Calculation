"""Option pricing path visualizer (European via Black–Scholes, American via binomial).

Simulates a geometric Brownian stock path and prices a call/put along the path.
Running as a script launches the interactive Streamlit UI; ``main()`` keeps the
standalone Plotly figure demo if you import and call it manually.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


OptionType = Literal["call", "put"]
OptionStyle = Literal["european", "american"]


@dataclass
class BSParams:
    """Parameters for the Black–Scholes model and path simulation."""


    spot: float = 100.0
    strike: float = 120.0  # OTM call by default
    maturity_years: float = 63.0 / 252.0  # about 3 months
    risk_free_rate: float = 0.025
    implied_vol: float = 0.30
    drift: float = 0.05  # expected stock drift for the simulated path
    steps: int = 252
    option_type: OptionType = "call"
    option_style: OptionStyle = "european"
    binomial_steps: int = 200
    seed: int | None = 42


def _norm_cdf(x: float) -> float:
    """Cumulative distribution for a standard normal."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType = "call",
) -> float:
    """Return the Black–Scholes price for a European call or put."""

    if time_to_maturity <= 0:
        intrinsic = max(spot - strike, 0.0)
        return intrinsic if option_type == "call" else max(strike - spot, 0.0)

    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    if option_type == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-risk_free_rate * time_to_maturity) * _norm_cdf(d2)

    return strike * math.exp(-risk_free_rate * time_to_maturity) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def american_option_price_binomial(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    steps: int,
    option_type: OptionType = "call",
) -> float:
    """Price an American option using a CRR binomial tree."""

    if time_to_maturity <= 0:
        intrinsic = max(spot - strike, 0.0)
        return intrinsic if option_type == "call" else max(strike - spot, 0.0)

    dt = time_to_maturity / steps
    u = math.exp(volatility * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-risk_free_rate * dt)
    p = (math.exp(risk_free_rate * dt) - d) / (u - d)

    # terminal payoffs
    payoffs = np.array(
        [
            max(spot * (u ** j) * (d ** (steps - j)) - strike, 0.0)
            if option_type == "call"
            else max(strike - spot * (u ** j) * (d ** (steps - j)), 0.0)
            for j in range(steps + 1)
        ]
    )

    # backward induction with early exercise
    for i in range(steps - 1, -1, -1):
        payoffs = disc * (p * payoffs[1 : i + 2] + (1 - p) * payoffs[0 : i + 1])
        stock_prices = spot * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
        intrinsic = (
            np.maximum(stock_prices - strike, 0.0)
            if option_type == "call"
            else np.maximum(strike - stock_prices, 0.0)
        )
        payoffs = np.maximum(payoffs, intrinsic)

    return float(payoffs[0])


def simulate_stock_path(params: BSParams) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a geometric Brownian motion stock path."""

    if params.seed is not None:
        np.random.seed(params.seed)

    dt = params.maturity_years / params.steps
    drift_term = (params.drift - 0.5 * params.implied_vol**2) * dt
    diffusion = params.implied_vol * math.sqrt(dt) * np.random.randn(params.steps)

    log_returns = drift_term + diffusion
    log_path = np.log(params.spot) + np.cumsum(log_returns)
    prices = np.concatenate(([params.spot], np.exp(log_path)))
    time_grid = np.linspace(0.0, params.maturity_years, params.steps + 1)
    return time_grid, prices


def option_path_from_stock(
    time_grid: np.ndarray, prices: np.ndarray, params: BSParams
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the option price and its time value along the stock path."""

    remaining_time = params.maturity_years - time_grid
    option_prices: list[float] = []
    extrinsic_values: list[float] = []
    for s, t in zip(prices, remaining_time):
        if params.option_style == "american":
            price = american_option_price_binomial(
                spot=float(s),
                strike=params.strike,
                time_to_maturity=float(t),
                risk_free_rate=params.risk_free_rate,
                volatility=params.implied_vol,
                steps=params.binomial_steps,
                option_type=params.option_type,
            )
        else:
            price = black_scholes_price(
                spot=float(s),
                strike=params.strike,
                time_to_maturity=float(t),
                risk_free_rate=params.risk_free_rate,
                volatility=params.implied_vol,
                option_type=params.option_type,
            )
        intrinsic = max(s - params.strike, 0.0) if params.option_type == "call" else max(
            params.strike - s, 0.0
        )
        option_prices.append(price)
        extrinsic_values.append(price - intrinsic)
    return np.array(option_prices), np.array(extrinsic_values)


def build_option_vs_time_fig(
    time_axis: np.ndarray,
    time_label: str,
    stock: np.ndarray,
    option_prices: np.ndarray,
    extrinsic: np.ndarray,
    params: BSParams,
) -> go.Figure:
    """Return an interactive chart with stock and option prices over time."""

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Simulated Stock Price vs Time",
            f"{params.option_type.title()} Price ({params.option_style.title()}) vs Time",
            "Time Value (Extrinsic) vs Time",
        ),
    )

    fig.add_trace(
        go.Scatter(x=time_axis, y=stock, name="Stock", line=dict(color="royalblue")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=option_prices,
            name="Option Price",
            line=dict(color="firebrick"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=extrinsic,
            name="Time Value",
            line=dict(color="seagreen"),
        ),
        row=3,
        col=1,
    )

    fig.update_xaxes(title_text=time_label, row=3, col=1)
    fig.update_yaxes(title_text="Stock Price", row=1, col=1)
    fig.update_yaxes(title_text="Option Price", row=2, col=1)
    fig.update_yaxes(title_text="Time Value", row=3, col=1)
    fig.update_layout(
        title="Black–Scholes Option Price Path",
        hovermode="x unified",
        height=950,
        template="plotly_white",
    )
    return fig


def main() -> None:
    """Run a quick demo of the Black–Scholes path visualization."""

    params = BSParams()
    time_grid, stock_path = simulate_stock_path(params)
    option_prices, extrinsic = option_path_from_stock(time_grid, stock_path, params)
    time_axis = time_grid * 252.0
    fig = build_option_vs_time_fig(
        time_axis, "Time (trading days to expiry)", stock_path, option_prices, extrinsic, params
    )
    fig.show()


def run_streamlit_app() -> None:
    """Render an interactive Streamlit app to play with option assumptions."""

    import streamlit as st

    st.set_page_config(page_title="Option Pricing Visualizer", layout="centered")
    st.title("Option Pricing Visualizer")
    st.caption("Switch between European (Black–Scholes) and American (binomial) pricing.")

    with st.sidebar:
        st.header("Inputs")
        spot = st.number_input("Spot", value=100.0, min_value=0.01, step=1.0)
        strike = st.number_input("Strike", value=120.0, min_value=0.01, step=1.0)
        time_mode = st.radio("Time mode", ["Trading days", "Minutes to expiry"], index=0)
        if time_mode == "Trading days":
            maturity_days = st.slider("Maturity (trading days)", min_value=1, max_value=504, value=63, step=1)
            maturity_minutes = None
        else:
            maturity_minutes = st.slider("Minutes to expiry", min_value=5, max_value=390, value=120, step=5)
            maturity_days = None
        risk_free = st.number_input("Risk-free rate", value=0.025, step=0.005, format="%.4f")
        vol = st.number_input("Implied volatility", value=0.30, min_value=0.01, step=0.01, format="%.4f")
        drift = st.number_input("Simulated stock drift", value=0.05, step=0.01, format="%.4f")
        steps_default = 252 if time_mode == "Trading days" else 300
        steps = st.slider("Path steps", min_value=20, max_value=2000, value=steps_default, step=10)
        option_type = st.selectbox("Option type", options=["call", "put"], index=0)
        option_style = st.selectbox("Style", options=["european", "american"], index=0)
        binom_steps = st.slider("Binomial steps (American)", min_value=10, max_value=400, value=200, step=10)
        seed = st.number_input("Random seed (optional)", value=42, step=1)

    trading_minutes_per_year = 252 * 390
    if maturity_days is not None:
        maturity_years = maturity_days / 252.0
        time_label = "Time (trading days to expiry)"
        time_scale = 252.0
    else:
        maturity_years = float(maturity_minutes) / trading_minutes_per_year  # type: ignore[arg-type]
        time_label = "Time (minutes to expiry)"
        time_scale = trading_minutes_per_year

    params = BSParams(
        spot=spot,
        strike=strike,
        maturity_years=maturity_years,
        risk_free_rate=risk_free,
        implied_vol=vol,
        drift=drift,
        steps=int(steps),
        option_type=option_type,  # type: ignore[arg-type]
        option_style=option_style,  # type: ignore[arg-type]
        binomial_steps=int(binom_steps),
        seed=int(seed),
    )

    time_grid, stock_path = simulate_stock_path(params)
    option_prices, extrinsic = option_path_from_stock(time_grid, stock_path, params)
    time_axis = time_grid * time_scale
    fig = build_option_vs_time_fig(time_axis, time_label, stock_path, option_prices, extrinsic, params)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # Default to the interactive Streamlit app when run as a script.
    run_streamlit_app()
