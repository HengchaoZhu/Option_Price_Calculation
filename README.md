# Volatility Strategy Simulator

Streamlit app for simulating a stock path with dynamic drift/vol (Heston + OU drift) and exploring option chains and multi‑leg strategies with mark‑to‑market P/L.

## Features
- Heston variance with OU drift (dynamic μ and σ) driven by user inputs; single simulated path used throughout the app.
- Stock path chart with optional candlesticks plus overlay plots of simulated drift and implied vol.
- Option price path from any chosen time point and expiry; configurable pricing style (European/Black–Scholes, American/binomial), strike spacing/depth, and option type.
- Option chain with Delta/Gamma and time value; pick legs directly from the chain.
- Strategy builder with integer quantities (long/short), expiry payoff, and mark‑to‑market P/L before expiry.

## Run
From the project root:
```bash
streamlit run bs_visualization.py
```
Open the URL shown in the terminal (default http://localhost:8501).

## Inputs
- Spot, initial drift μ0, initial implied vol σ0, risk‑free rate
- Horizon (trading days or minutes), path steps, random seed
- Advanced (expander): Heston κ/θ/ξ/ρ and OU μ̄/α/σμ/ρμS
- Part 2: select time index on the path, option type, expiry length, pricing style, strike spacing/depth
- Part 3: select legs from the chain, set integer qty (negative for shorts)

## Notes
- The simulator uses the dynamic engine from `dynamic_iv_sim.py`; regimes/GBM are no longer used.
- Option pricing along the path uses the simulated drift/vol at the selected time; risk‑free is constant.
- Candlestick intervals can be toggled (hour/day/week/month/quarter/year) for stock and option views.
