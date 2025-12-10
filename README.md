# Option_value_simulation & Option Pricing Visualizer
Black–Scholes option pricing path visualizer.  This script simulates a stock path under geometric Brownian motion and computes the corresponding Black–Scholes option price as time decays. 
Interactive Black–Scholes / binomial option pricer with a simulated geometric Brownian stock path.

## What it does
- Simulates a stock path under GBM using user-defined drift/vol/steps and seed.
- Prices a call/put along the path: European via closed-form Black–Scholes, American via a CRR binomial tree with early exercise.
- Displays stock, option price, and time value (extrinsic) versus time (trading days or intraday minutes).

## Quick start
From the project root:
```bash
streamlit run bs_visualization.py
```
Open the local URL shown in the terminal (default http://localhost:8501).

## Controls (sidebar)
- Spot, Strike, Option type (call/put)
- Style: European (BS) or American (binomial)
- Time mode: trading days or minutes to expiry
- Risk-free rate, Implied volatility (used for both diffusion and pricing)
- Simulated stock drift
- Path steps (GBM path resolution)
- Binomial steps (American tree depth)
- Random seed

## Notes
- Time value is reported as option price minus intrinsic at each node.
- Minutes mode converts minutes to expiry into the correct year fraction using 252 trading days × 390 minutes/day.
- If you prefer a static plot without Streamlit, import and call `straddle_bet.bs_visualization.main()`.
