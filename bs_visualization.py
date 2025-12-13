"""Option pricing path visualizer (European via Black–Scholes, American via binomial).

Simulates a geometric Brownian stock path and prices a call/put along the path.
Running as a script launches the interactive Streamlit UI; ``main()`` keeps the
standalone Plotly figure demo if you import and call it manually.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dynamic_iv_sim import HestonParams, OUDriftParams, SimConfig, simulate_paths


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


@dataclass
class Regime:
    """Path regime with its own drift/vol and time fraction of the maturity."""

    fraction: float  # fraction of total maturity (will be normalized)
    drift: float
    implied_vol: float
    risk_free: float


def _norm_cdf(x: float) -> float:
    """Cumulative distribution for a standard normal."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Probability density for a standard normal."""

    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bucket_ohlc(
    time_axis: np.ndarray, series: np.ndarray, bucket: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a time series into OHLC buckets; bucket is in same units as time_axis."""

    if bucket <= 0:
        raise ValueError("Bucket size must be positive.")
    bucket_ids = np.floor(time_axis / bucket).astype(int)
    unique_ids = np.unique(bucket_ids)
    opens, highs, lows, closes, xs = [], [], [], [], []
    for bid in unique_ids:
        mask = bucket_ids == bid
        if not np.any(mask):
            continue
        seg = series[mask]
        opens.append(seg[0])
        closes.append(seg[-1])
        highs.append(seg.max())
        lows.append(seg.min())
        xs.append((bid + 1) * bucket)
    return np.array(xs), np.array(opens), np.array(highs), np.array(lows), np.array(closes)


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


def simulate_stock_path(
    params: BSParams, regimes: list[Regime] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a geometric Brownian motion stock path, optionally with regimes.

    Returns time_grid, prices, risk_free_path, vol_path (all aligned).
    """

    if params.seed is not None:
        np.random.seed(params.seed)

    if not regimes:
        dt = params.maturity_years / params.steps
        drift_term = (params.drift - 0.5 * params.implied_vol**2) * dt
        diffusion = params.implied_vol * math.sqrt(dt) * np.random.randn(params.steps)

        log_returns = drift_term + diffusion
        log_path = np.log(params.spot) + np.cumsum(log_returns)
        prices = np.concatenate(([params.spot], np.exp(log_path)))
        time_grid = np.linspace(0.0, params.maturity_years, params.steps + 1)
        risk_path = np.full_like(prices, params.risk_free_rate, dtype=float)
        vol_path = np.full_like(prices, params.implied_vol, dtype=float)
        return time_grid, prices, risk_path, vol_path

    # Normalize regime fractions
    total_fraction = sum(max(r.fraction, 0.0) for r in regimes)
    if total_fraction <= 0:
        norm_regimes = [
            Regime(fraction=1.0 / len(regimes), drift=r.drift, implied_vol=r.implied_vol, risk_free=r.risk_free)
            for r in regimes
        ]
    else:
        norm_regimes = [
            Regime(
                fraction=r.fraction / total_fraction,
                drift=r.drift,
                implied_vol=r.implied_vol,
                risk_free=r.risk_free,
            )
            for r in regimes
        ]

    prices: list[float] = [params.spot]
    times: list[float] = [0.0]
    risk_path: list[float] = [norm_regimes[0].risk_free if norm_regimes else params.risk_free_rate]
    vol_path: list[float] = [norm_regimes[0].implied_vol if norm_regimes else params.implied_vol]
    steps_used = 0
    for idx, r in enumerate(norm_regimes):
        remaining_steps = params.steps - steps_used
        if idx == len(norm_regimes) - 1:
            reg_steps = max(1, remaining_steps)
        else:
            reg_steps = max(1, int(round(params.steps * r.fraction)))
        steps_used += reg_steps

        reg_time = params.maturity_years * r.fraction
        dt = reg_time / reg_steps
        drift_term = (r.drift - 0.5 * r.implied_vol**2) * dt
        diffusion = r.implied_vol * math.sqrt(dt) * np.random.randn(reg_steps)

        log_returns = drift_term + diffusion
        log_path = math.log(prices[-1]) + np.cumsum(log_returns)
        segment_prices = np.exp(log_path)
        prices.extend(segment_prices.tolist())

        segment_times = times[-1] + np.linspace(dt, reg_time, reg_steps)
        times.extend(segment_times.tolist())
        risk_path.extend([r.risk_free] * reg_steps)
        vol_path.extend([r.implied_vol] * reg_steps)

    # Adjust last time to exactly maturity to avoid tiny drift due to rounding.
    times[-1] = params.maturity_years

    return np.array(times), np.array(prices), np.array(risk_path), np.array(vol_path)


def option_path_from_stock(
    time_grid: np.ndarray,
    prices: np.ndarray,
    params: BSParams,
    risk_free_path: np.ndarray | None = None,
    vol_path: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the option price and its time value along the stock path."""

    remaining_time = params.maturity_years - time_grid
    option_prices: list[float] = []
    extrinsic_values: list[float] = []
    for idx, (s, t) in enumerate(zip(prices, remaining_time)):
        rf = float(risk_free_path[idx]) if risk_free_path is not None else params.risk_free_rate
        vol = float(vol_path[idx]) if vol_path is not None else params.implied_vol
        if params.option_style == "american":
            price = american_option_price_binomial(
                spot=float(s),
                strike=params.strike,
                time_to_maturity=float(t),
                risk_free_rate=rf,
                volatility=vol,
                steps=params.binomial_steps,
                option_type=params.option_type,
            )
        else:
            price = black_scholes_price(
                spot=float(s),
                strike=params.strike,
                time_to_maturity=float(t),
                risk_free_rate=rf,
                volatility=vol,
                option_type=params.option_type,
            )
        intrinsic = max(s - params.strike, 0.0) if params.option_type == "call" else max(
            params.strike - s, 0.0
        )
        option_prices.append(price)
        extrinsic_values.append(price - intrinsic)
    return np.array(option_prices), np.array(extrinsic_values)


def option_price_for_params(
    params: BSParams,
    strike: float,
    option_type: OptionType,
    style: OptionStyle,
    risk_free_override: float | None = None,
    implied_vol_override: float | None = None,
) -> float:
    """Helper to price an option with the current parameter set."""

    risk_free = risk_free_override if risk_free_override is not None else params.risk_free_rate
    vol = implied_vol_override if implied_vol_override is not None else params.implied_vol

    if style == "american":
        return american_option_price_binomial(
            spot=params.spot,
            strike=strike,
            time_to_maturity=params.maturity_years,
            risk_free_rate=risk_free,
            volatility=vol,
            steps=params.binomial_steps,
            option_type=option_type,
        )
    return black_scholes_price(
        spot=params.spot,
        strike=strike,
        time_to_maturity=params.maturity_years,
        risk_free_rate=risk_free,
        volatility=vol,
        option_type=option_type,
    )


def intrinsic_value_at_expiry(prices: np.ndarray, strike: float, option_type: OptionType) -> np.ndarray:
    """Vectorized intrinsic value for an option at expiry across price grid."""

    if option_type == "call":
        return np.maximum(prices - strike, 0.0)
    return np.maximum(strike - prices, 0.0)


def option_price_with_time(
    spot: float,
    remaining_time: float,
    strike: float,
    option_type: OptionType,
    style: OptionStyle,
    params: BSParams,
    risk_free_override: float | None = None,
    implied_vol_override: float | None = None,
) -> float:
    """Price an option for a given remaining time; used to mark P/L along the path."""

    if remaining_time <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)

    risk_free = risk_free_override if risk_free_override is not None else params.risk_free_rate
    vol = implied_vol_override if implied_vol_override is not None else params.implied_vol

    if style == "american":
        # Scale binomial steps with remaining time to keep runtime modest.
        scaled_steps = params.binomial_steps
        if params.maturity_years > 0:
            scaled_steps = max(1, int(params.binomial_steps * remaining_time / params.maturity_years))
        return american_option_price_binomial(
            spot=spot,
            strike=strike,
            time_to_maturity=remaining_time,
            risk_free_rate=risk_free,
            volatility=vol,
            steps=scaled_steps,
            option_type=option_type,
        )

    return black_scholes_price(
        spot=spot,
        strike=strike,
        time_to_maturity=remaining_time,
        risk_free_rate=risk_free,
        volatility=vol,
        option_type=option_type,
    )


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
) -> tuple[float, float]:
    """Return delta and gamma for a European option."""

    if time_to_maturity <= 0 or volatility <= 0 or spot <= 0:
        if option_type == "call":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return delta, 0.0

    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
    ) / (volatility * sqrt_t)
    if option_type == "call":
        delta = _norm_cdf(d1)
    else:
        delta = _norm_cdf(d1) - 1.0
    gamma = _norm_pdf(d1) / (spot * volatility * sqrt_t)
    return delta, gamma


def compute_option_greeks(
    spot: float,
    remaining_time: float,
    strike: float,
    option_type: OptionType,
    style: OptionStyle,
    params: BSParams,
    risk_free_override: float | None = None,
    implied_vol_override: float | None = None,
) -> tuple[float, float]:
    """Return delta and gamma given style; for American use finite differences."""

    risk_free = risk_free_override if risk_free_override is not None else params.risk_free_rate
    vol = implied_vol_override if implied_vol_override is not None else params.implied_vol
    if style == "european":
        return black_scholes_greeks(
            spot=spot,
            strike=strike,
            time_to_maturity=remaining_time,
            risk_free_rate=risk_free,
            volatility=vol,
            option_type=option_type,
        )

    bump = max(0.01 * spot, 0.5)
    price_up = option_price_with_time(
        spot=spot + bump,
        remaining_time=remaining_time,
        strike=strike,
        option_type=option_type,
        style=style,
        params=params,
        risk_free_override=risk_free,
        implied_vol_override=vol,
    )
    price_down = option_price_with_time(
        spot=spot - bump,
        remaining_time=remaining_time,
        strike=strike,
        option_type=option_type,
        style=style,
        params=params,
        risk_free_override=risk_free,
        implied_vol_override=vol,
    )
    price_mid = option_price_with_time(
        spot=spot,
        remaining_time=remaining_time,
        strike=strike,
        option_type=option_type,
        style=style,
        params=params,
        risk_free_override=risk_free,
        implied_vol_override=vol,
    )
    delta = (price_up - price_down) / (2 * bump)
    gamma = (price_up - 2 * price_mid + price_down) / (bump**2)
    return delta, gamma


def build_option_chain(
    params: BSParams,
    strikes_each_side: int = 5,
    strike_step: float = 5.0,
    style_override: OptionStyle | None = None,
    risk_free_override: float | None = None,
    implied_vol_override: float | None = None,
) -> pd.DataFrame:
    """Generate a simple option chain around spot for quick strategy exploration."""

    style = style_override or params.option_style
    strikes = [
        max(0.01, params.spot + i * strike_step) for i in range(-strikes_each_side, strikes_each_side + 1)
    ]
    rows = []
    for strike in sorted(strikes):
        call_px = option_price_for_params(
            params, strike, "call", style, risk_free_override=risk_free_override, implied_vol_override=implied_vol_override
        )
        put_px = option_price_for_params(
            params, strike, "put", style, risk_free_override=risk_free_override, implied_vol_override=implied_vol_override
        )
        call_intrinsic = max(params.spot - strike, 0.0)
        put_intrinsic = max(strike - params.spot, 0.0)
        rows.append(
            {
                "Strike": strike,
                "Moneyness %": 100.0 * (strike / params.spot - 1.0),
                "Call": call_px,
                "Put": put_px,
                "Straddle": call_px + put_px,
                "Call TV": call_px - call_intrinsic,
                "Put TV": put_px - put_intrinsic,
            }
        )

    df = pd.DataFrame(rows)
    return df[
        ["Strike", "Moneyness %", "Call", "Put", "Straddle", "Call TV", "Put TV"]
    ].round({"Strike": 2, "Moneyness %": 2, "Call": 2, "Put": 2, "Straddle": 2, "Call TV": 2, "Put TV": 2})


def build_strategy_payoff(
    price_grid: np.ndarray,
    legs: list[dict[str, float | str]],
    params: BSParams,
    style: OptionStyle,
    risk_free_override: float | None = None,
    implied_vol_override: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute total payoff across a price grid for multi-leg option strategies."""

    total_payoff = np.zeros_like(price_grid, dtype=float)
    leg_rows: list[dict[str, float | str]] = []
    for idx, leg in enumerate(legs, start=1):
        opt_type = leg["option_type"]  # type: ignore[assignment]
        strike = float(leg["strike"])
        qty = float(leg["quantity"])
        premium = option_price_for_params(
            params,
            strike,
            opt_type,
            style,  # type: ignore[arg-type]
            risk_free_override=risk_free_override,
            implied_vol_override=implied_vol_override,
        )
        intrinsic = intrinsic_value_at_expiry(price_grid, strike, opt_type)  # type: ignore[arg-type]
        leg_payoff = qty * (intrinsic - premium)
        total_payoff += leg_payoff
        leg_rows.append(
            {
                "Leg": idx,
                "Type": opt_type,
                "Strike": strike,
                "Qty": qty,
                "Premium": premium,
                "Cost": qty * premium,
            }
        )

    legs_df = pd.DataFrame(leg_rows)
    return total_payoff, legs_df


def strategy_value_over_time(
    time_grid: np.ndarray,
    stock_path: np.ndarray,
    legs: list[dict[str, float | str]],
    params: BSParams,
    style: OptionStyle,
    risk_free_path: np.ndarray | None = None,
    vol_path: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return mark-to-market value and P/L along the simulated path."""

    initial_cost = 0.0
    for leg in legs:
        initial_cost += float(leg["quantity"]) * option_price_for_params(
            params,
            float(leg["strike"]),
            leg["option_type"],  # type: ignore[arg-type]
            style,
            risk_free_override=risk_free_path[0] if risk_free_path is not None else None,
            implied_vol_override=vol_path[0] if vol_path is not None else None,
        )

    values: list[float] = []
    for idx, (t, s) in enumerate(zip(time_grid, stock_path)):
        remaining = max(params.maturity_years - float(t), 0.0)
        total_val = 0.0
        for leg in legs:
            total_val += float(leg["quantity"]) * option_price_with_time(
                spot=float(s),
                remaining_time=remaining,
                strike=float(leg["strike"]),
                option_type=leg["option_type"],  # type: ignore[arg-type]
                style=style,
                params=params,
                risk_free_override=risk_free_path[idx] if risk_free_path is not None else None,
                implied_vol_override=vol_path[idx] if vol_path is not None else None,
            )
        values.append(total_val)

    values_arr = np.array(values)
    pnl_arr = values_arr - initial_cost
    return values_arr, pnl_arr, initial_cost


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
    time_grid, stock_path, risk_path, vol_path = simulate_stock_path(params)
    option_prices, extrinsic = option_path_from_stock(time_grid, stock_path, params, risk_path, vol_path)
    time_axis = time_grid * 252.0
    fig = build_option_vs_time_fig(
        time_axis, "Time (trading days to expiry)", stock_path, option_prices, extrinsic, params
    )
    fig.show()


def run_streamlit_app() -> None:
    """Render the Volatility Strategy Simulator Streamlit app."""

    import streamlit as st

    st.set_page_config(page_title="Volatility Strategy Simulator", layout="wide")
    st.title("Volatility Strategy Simulator")
    st.markdown(
        "Sketch and stress-test option strategies on a simulated stock path with regimes, "
        "option chain, Greeks, and path-aware P/L."
    )

    # -----------------------------------------
    # Part 1: Stock simulation
    # -----------------------------------------
    st.header("Part 1: Stock Simulation")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        spot = st.number_input("Spot", value=100.0, min_value=0.01, step=1.0)
        time_mode = st.radio("Time mode", ["Trading days", "Minutes"], index=0, horizontal=True)
        trading_minutes_per_year = 252 * 390
        if time_mode == "Trading days":
            maturity_units = st.number_input("Total horizon (trading days)", min_value=1, value=126, step=1)
            time_scale = 252.0
            time_label = "Time (trading days)"
        else:
            maturity_units = st.number_input("Total horizon (minutes)", min_value=5, value=390, step=5)
            time_scale = float(trading_minutes_per_year)
            time_label = "Time (minutes)"
        steps_default = 252 if time_mode == "Trading days" else 400
        steps = st.number_input("Path steps", min_value=20, max_value=10000, value=steps_default, step=10)
    with col_a2:
        base_drift = st.number_input("Initial drift μ0", value=0.05, step=0.01, format="%.4f")
        base_vol = st.number_input("Initial implied vol σ0", value=0.30, min_value=0.0001, step=0.01, format="%.4f")
        base_risk_free = st.number_input("Risk-free rate", value=0.025, step=0.005, format="%.4f")
        binom_steps = st.slider("Binomial steps (American pricing)", min_value=10, max_value=600, value=250, step=10)
    with col_a3:
        seed = st.number_input("Random seed (optional)", value=42, step=1)

    with st.expander("Advanced Heston / dynamic drift"):
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            h_kappa = st.number_input("Heston kappa", value=2.5, min_value=0.0001, step=0.1, format="%.4f")
            h_theta = st.number_input("Heston theta (var level)", value=base_vol * base_vol, min_value=0.0001, step=0.01, format="%.4f")
            h_xi = st.number_input("Heston vol-of-vol (xi)", value=0.45, min_value=0.0001, step=0.01, format="%.4f")
        with h_col2:
            h_rho = st.number_input("Heston rho (spot/var)", value=-0.7, min_value=-0.999, max_value=0.999, step=0.05, format="%.3f")
            mu_bar = st.number_input("OU long-run mu_bar", value=base_drift, step=0.01, format="%.4f")
            alpha = st.number_input("OU mean reversion (alpha)", value=1.5, min_value=0.0001, step=0.1, format="%.4f")
            sigma_mu = st.number_input("OU drift vol (sigma_mu)", value=0.05, min_value=0.0001, step=0.01, format="%.4f")
            rho_mu_s = st.number_input("Corr(dW, dB) rho_muS", value=-0.3, min_value=-0.999, max_value=0.999, step=0.05, format="%.3f")
    dynamic_params = {
        "kappa": h_kappa,
        "theta": h_theta,
        "xi": h_xi,
        "rho": h_rho,
        "mu0": base_drift,
        "mu_bar": mu_bar,
        "alpha": alpha,
        "sigma_mu": sigma_mu,
        "rho_mu_s": rho_mu_s,
    }
    maturity_years = maturity_units / time_scale
    cfg = SimConfig(
        S0=spot,
        r=base_risk_free,
        q=0.0,
        sigma0=base_vol,
        T_max=maturity_years,
        n_steps=int(steps),
        n_paths=1,
        heston=HestonParams(
            kappa=dynamic_params["kappa"],
            theta=dynamic_params["theta"],
            xi=dynamic_params["xi"],
            rho=dynamic_params["rho"],
        ),
        drift_model="ou",
        ou_params=OUDriftParams(
            mu0=dynamic_params["mu0"],
            alpha=dynamic_params["alpha"],
            mu_bar=dynamic_params["mu_bar"],
            sigma_mu=dynamic_params["sigma_mu"],
            rho_mu_s=dynamic_params["rho_mu_s"],
        ),
        seed=int(seed),
    )
    sim_res = simulate_paths(cfg, risk_neutral=False)
    time_grid = sim_res["time"]
    stock_path = sim_res["S"][:, 0]
    vol_path = np.sqrt(np.maximum(sim_res["v"][:, 0], 0.0))
    risk_path = np.full_like(time_grid, base_risk_free, dtype=float)
    mu_path = sim_res["mu"][:, 0]
    time_axis = time_grid * time_scale

    stock_view = st.radio("Stock price view", ["Line", "Candles"], index=0, horizontal=True, key="stock_view_mode")
    if stock_view == "Candles":
        interval_choice = st.selectbox(
            "Candle interval",
            ["Hour", "Day", "Week", "Month", "Quarter", "Year"],
            index=1 if time_mode == "Trading days" else 0,
            key="stock_candle_interval",
        )
        interval_map_days = {"Hour": 1 / 6.5, "Day": 1.0, "Week": 5.0, "Month": 21.0, "Quarter": 63.0, "Year": 252.0}
        interval_map_minutes = {
            "Hour": 60.0,
            "Day": 390.0,
            "Week": 390.0 * 5,
            "Month": 390.0 * 21,
            "Quarter": 390.0 * 63,
            "Year": 390.0 * 252,
        }
        bucket_len = interval_map_days[interval_choice] if time_mode == "Trading days" else interval_map_minutes[interval_choice]

    stock_fig = go.Figure()
    if stock_view == "Line":
        stock_fig.add_trace(go.Scatter(x=time_axis, y=stock_path, name="Stock", line=dict(color="royalblue")))
    else:
        xs, opens, highs, lows, closes = _bucket_ohlc(time_axis, stock_path, bucket_len)
        stock_fig = go.Figure(
            data=go.Candlestick(
                x=xs,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name="Stock",
            )
        )
    stock_fig.update_layout(
        title="Simulated Stock Price vs Time",
        xaxis_title=time_label,
        yaxis_title="Stock Price",
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(stock_fig, use_container_width=True)
    iv_mu_fig = go.Figure()
    iv_mu_fig.add_trace(go.Scatter(x=time_axis, y=vol_path, name="Implied vol (σ)", line=dict(color="orange")))
    iv_mu_fig.add_trace(go.Scatter(x=time_axis, y=mu_path, name="Drift (μ)", line=dict(color="green")))
    iv_mu_fig.update_layout(
        title="Expected IV and Drift vs Time",
        xaxis_title=time_label,
        yaxis_title="Level",
        hovermode="x unified",
        template="plotly_white",
        height=320,
    )
    st.plotly_chart(iv_mu_fig, use_container_width=True)

    # -----------------------------------------
    # Part 2: Option Price Path and Chain
    # -----------------------------------------
    st.header("Part 2: Option Price Path")
    idx_input = st.number_input(
        "Select a time point on the path (index)",
        min_value=0,
        max_value=len(time_axis) - 1,
        value=len(time_axis) // 3,
        step=1,
    )
    idx = int(idx_input)
    current_time = time_grid[idx]
    current_time_label = (
        f"{time_axis[idx]:.1f} trading days"
        if time_mode == "Trading days"
        else f"{time_axis[idx]:.0f} minutes"
    )
    spot_now = float(stock_path[idx])
    rf_now = float(risk_path[idx])
    vol_now = float(vol_path[idx])
    remaining_years = max(maturity_years - current_time, 0.0)
    remaining_units = remaining_years * time_scale

    option_type_path = st.selectbox(
        "Option type to track (for path plots and chain)",
        options=["call", "put"],
        index=0,
        key="option_type_path_select",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Spot at selected time", f"{spot_now:.2f}", help=f"Time = {current_time_label}")
    with c2:
        st.metric("Regime risk-free", f"{rf_now:.4f}")
    with c3:
        st.metric("Regime vol", f"{vol_now:.4f}")

    if remaining_years <= 0:
        st.warning("Selected time is at/after the horizon; no time left for option expiry.")
        return

    expiry_units = st.slider(
        "Expiry from this time",
        min_value=1,
        max_value=max(1, int(max(remaining_units, 1))),
        value=max(1, int(min(remaining_units, remaining_units * 0.5 or 1))),
        step=1,
        help="Set expiry relative to the selected time point.",
    )
    option_maturity_years = expiry_units / time_scale

    chain_depth = st.slider("Option chain depth (strikes each side)", min_value=1, max_value=15, value=6, step=1)
    chain_step = st.number_input("Strike spacing", value=5.0, min_value=0.5, step=0.5)
    pricing_style = st.selectbox("Pricing style", options=["european", "american"], index=0)
    tracked_strike = st.number_input("Tracked option strike", value=float(round(spot_now, 2)), step=1.0)

    option_params = BSParams(
        spot=spot_now,
        strike=tracked_strike,
        maturity_years=option_maturity_years,
        risk_free_rate=rf_now,
        implied_vol=vol_now,
        drift=mu_path[idx],
        steps=max(20, int(steps * option_maturity_years / maturity_years)) if maturity_years > 0 else 200,
        option_type=option_type_path,  # type: ignore[arg-type]
        option_style=pricing_style,  # type: ignore[arg-type]
        binomial_steps=int(binom_steps),
        seed=int(seed),
    )

    chain_wide = build_option_chain(
        option_params,
        strikes_each_side=int(chain_depth),
        strike_step=float(chain_step),
        style_override=pricing_style,  # type: ignore[arg-type]
        risk_free_override=rf_now,
        implied_vol_override=vol_now,
    )

    chain_rows: list[dict[str, float | str]] = []
    for _, row in chain_wide.iterrows():
        for opt_type, col_price, col_tv in [("call", "Call", "Call TV"), ("put", "Put", "Put TV")]:
            delta, gamma = compute_option_greeks(
                spot=spot_now,
                remaining_time=option_maturity_years,
                strike=float(row["Strike"]),
                option_type=opt_type,  # type: ignore[arg-type]
                style=pricing_style,  # type: ignore[arg-type]
                params=option_params,
                risk_free_override=rf_now,
                implied_vol_override=vol_now,
            )
            chain_rows.append(
                {
                    "Strike": float(row["Strike"]),
                    "Moneyness %": float(row["Moneyness %"]),
                    "Type": opt_type,
                    "Price": float(row[col_price]),
                    "Delta": delta,
                    "Gamma": gamma,
                    "Time Value": float(row[col_tv]),
                }
            )
    chain_long = pd.DataFrame(chain_rows)
    st.dataframe(
        chain_long.sort_values(["Strike", "Type"]).reset_index(drop=True).round(4),
        use_container_width=True,
    )

    mask_future = (time_grid >= current_time) & (time_grid <= current_time + option_maturity_years + 1e-12)
    future_times = time_grid[mask_future] - current_time
    future_stock = stock_path[mask_future]
    future_rf = risk_path[mask_future]
    future_vol = vol_path[mask_future]
    opt_prices, extrinsic = option_path_from_stock(
        future_times,
        future_stock,
        option_params,
        risk_free_path=future_rf,
        vol_path=future_vol,
    )
    option_time_axis = future_times * time_scale

    option_view = st.radio("Option/stock price view", ["Line", "Candles"], index=0, horizontal=True, key="option_view")
    if option_view == "Candles":
        interval_choice_opt = st.selectbox(
            "Candle interval (option view)",
            ["Hour", "Day", "Week", "Month", "Quarter", "Year"],
            index=1 if time_mode == "Trading days" else 0,
            key="option_candle_interval",
        )
        interval_map_days = {"Hour": 1 / 6.5, "Day": 1.0, "Week": 5.0, "Month": 21.0, "Quarter": 63.0, "Year": 252.0}
        interval_map_minutes = {
            "Hour": 60.0,
            "Day": 390.0,
            "Week": 390.0 * 5,
            "Month": 390.0 * 21,
            "Quarter": 390.0 * 63,
            "Year": 390.0 * 252,
        }
        bucket_len_opt = (
            interval_map_days[interval_choice_opt] if time_mode == "Trading days" else interval_map_minutes[interval_choice_opt]
        )
    opt_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Stock", "Option price"))
    if option_view == "Line":
        opt_fig.add_trace(go.Scatter(x=option_time_axis, y=future_stock, name="Stock", line=dict(color="royalblue")), row=1, col=1)
        opt_fig.add_trace(go.Scatter(x=option_time_axis, y=opt_prices, name="Option price", line=dict(color="firebrick")), row=2, col=1)
    else:
        xs_s, opens_s, highs_s, lows_s, closes_s = _bucket_ohlc(option_time_axis, future_stock, bucket_len_opt)
        xs_o, opens_o, highs_o, lows_o, closes_o = _bucket_ohlc(option_time_axis, opt_prices, bucket_len_opt)
        opt_fig.add_trace(
            go.Candlestick(
                x=xs_s,
                open=opens_s,
                high=highs_s,
                low=lows_s,
                close=closes_s,
                name="Stock",
            ),
            row=1,
            col=1,
        )
        opt_fig.add_trace(
            go.Candlestick(
                x=xs_o,
                open=opens_o,
                high=highs_o,
                low=lows_o,
                close=closes_o,
                name="Option price",
            ),
            row=2,
            col=1,
        )
    opt_fig.add_trace(
        go.Scatter(x=option_time_axis, y=extrinsic, name="Time value", line=dict(color="seagreen", dash="dot")),
        row=2,
        col=1,
    )
    opt_fig.update_layout(
        title="Price path from selected time to expiry",
        xaxis_title=time_label,
        xaxis2_title=time_label,
        hovermode="x unified",
        template="plotly_white",
        height=520,
    )
    st.plotly_chart(opt_fig, use_container_width=True)

    # -----------------------------------------
    # Part 3: Strategy Price Path
    # -----------------------------------------
    st.header("Part 3: Strategy Price Path")
    if chain_long.empty:
        st.info("Build an option chain above to select legs.")
        return

    chain_long["Label"] = chain_long.apply(
        lambda r: f"{r['Type'].title()} {r['Strike']:.2f} | Δ {r['Delta']:.2f} Γ {r['Gamma']:.3f} | {r['Price']:.2f}",
        axis=1,
    )
    options_map = {row["Label"]: row for _, row in chain_long.iterrows()}
    default_selection = list(options_map.keys())[:2]
    selected_labels = st.multiselect("Select legs from the option chain", options=list(options_map.keys()), default=default_selection)

    legs: list[dict[str, float | str]] = []
    for label in selected_labels:
        row = options_map[label]
        c1, c2 = st.columns(2)
        with c1:
            st.write(label)
        with c2:
            qty = st.number_input(
                "Qty (integer; negative for short)",
                value=1,
                step=1,
                min_value=-1000,
                max_value=1000,
                key=f"qty_{label}",
            )
        qty_int = int(qty)
        legs.append({"option_type": row["Type"], "strike": float(row["Strike"]), "quantity": qty_int})

    price_range_pct = st.slider("Payoff range around current spot (%)", min_value=20, max_value=200, value=80, step=10)
    price_low = max(0.01, spot_now * (1 - price_range_pct / 100.0))
    price_high = spot_now * (1 + price_range_pct / 100.0)
    price_grid = np.linspace(price_low, price_high, 260)

    strategy_style: OptionStyle = pricing_style  # reuse chain pricing style
    payoff, legs_df = build_strategy_payoff(
        price_grid,
        legs,
        option_params,
        strategy_style,
        risk_free_override=rf_now,
        implied_vol_override=vol_now,
    )

    net_premium = float(legs_df["Cost"].sum()) if not legs_df.empty else 0.0
    payoff_at_spot = float(np.interp(spot_now, price_grid, payoff))
    sim_end_price = float(future_stock[-1]) if len(future_stock) > 0 else spot_now
    payoff_at_sim = float(np.interp(sim_end_price, price_grid, payoff))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Net premium (+cost / -credit)", f"{net_premium:.2f}")
    with m2:
        st.metric("P/L at current spot", f"{payoff_at_spot:.2f}")
    with m3:
        st.metric("P/L at simulated expiry", f"{payoff_at_sim:.2f}")

    payoff_fig = go.Figure()
    payoff_fig.add_trace(
        go.Scatter(
            x=price_grid,
            y=payoff,
            name="Expiry P/L",
            line=dict(color="indianred"),
        )
    )
    payoff_fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
    payoff_fig.add_vline(x=spot_now, line_width=1, line_dash="dot", line_color="royalblue")
    payoff_fig.add_vline(x=sim_end_price, line_width=1, line_dash="dash", line_color="purple")
    payoff_fig.update_layout(
        title="Strategy payoff at expiry",
        xaxis_title="Underlying price at expiry",
        yaxis_title="Total P/L",
        hovermode="x unified",
        template="plotly_white",
        height=480,
    )
    st.plotly_chart(payoff_fig, use_container_width=True)
    st.dataframe(
        legs_df.round({"Strike": 2, "Qty": 2, "Premium": 2, "Cost": 2}),
        use_container_width=True,
    )

    values_path, pnl_path, initial_cost = strategy_value_over_time(
        future_times,
        future_stock,
        legs,
        option_params,
        strategy_style,
        risk_free_path=future_rf,
        vol_path=future_vol,
    )
    t_step = max(1, len(option_time_axis) // 300)
    t_idx = st.slider(
        "Inspect P/L before expiry",
        min_value=0,
        max_value=len(option_time_axis) - 1,
        value=min(len(option_time_axis) - 1, len(option_time_axis) // 2),
        step=t_step,
    )
    t_label = (
        f"{option_time_axis[t_idx]:.1f} trading days"
        if time_mode == "Trading days"
        else f"{option_time_axis[t_idx]:.0f} minutes"
    )
    st.metric("Mark-to-market P/L at selected t", f"{pnl_path[t_idx]:.2f}", help=f"t = {t_label}")

    mtm_fig = go.Figure()
    mtm_fig.add_trace(go.Scatter(x=option_time_axis, y=pnl_path, name="P/L", line=dict(color="teal")))
    mtm_fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
    mtm_fig.add_vline(x=option_time_axis[t_idx], line_width=1, line_dash="dash", line_color="black")
    mtm_fig.update_layout(
        title="Strategy P/L before expiry (mark-to-market)",
        xaxis_title=time_label,
        yaxis_title="P/L",
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(mtm_fig, use_container_width=True)


if __name__ == "__main__":
    # Default to the interactive Streamlit app when run as a script.
    # Uncomment the line below to run the standalone Plotly figure demo instead.
    # main()
    run_streamlit_app()
