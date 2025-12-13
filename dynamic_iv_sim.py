"""Monte Carlo simulator with dynamic drift/vol and implied-vol surface construction.

Uses NumPy + SciPy + matplotlib only. Supports OU or regime-switching drift, Heston
variance with full truncation Euler, and Monte Carlo pricing with implied vols.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


DriftModel = Literal["ou", "regime"]


@dataclass
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float


@dataclass
class OUDriftParams:
    mu0: float
    alpha: float
    mu_bar: float
    sigma_mu: float
    rho_mu_s: float


@dataclass
class RegimeDriftParams:
    mu_regimes: np.ndarray  # shape (K,)
    transition_matrix: np.ndarray  # shape (K, K)
    vol_regimes: Optional[np.ndarray] = None  # optional per-regime vol level


@dataclass
class SimConfig:
    S0: float
    r: float
    q: float
    sigma0: float
    T_max: float
    n_steps: int
    n_paths: int
    heston: HestonParams
    drift_model: DriftModel
    ou_params: Optional[OUDriftParams] = None
    regime_params: Optional[RegimeDriftParams] = None
    seed: Optional[int] = None


def validate_config(cfg: SimConfig) -> None:
    if cfg.S0 <= 0 or cfg.sigma0 <= 0:
        raise ValueError("S0 and sigma0 must be positive.")
    if cfg.n_steps < 1 or cfg.n_paths < 1:
        raise ValueError("n_steps and n_paths must be positive.")
    if not (-0.999 < cfg.heston.rho < 0.999):
        raise ValueError("Heston rho must be in (-1, 1).")
    if cfg.heston.kappa <= 0 or cfg.heston.theta <= 0 or cfg.heston.xi <= 0:
        raise ValueError("Heston kappa, theta, xi must be positive.")
    if cfg.drift_model == "ou":
        if cfg.ou_params is None:
            raise ValueError("OU parameters required for OU drift model.")
        if not (-0.999 < cfg.ou_params.rho_mu_s < 0.999):
            raise ValueError("rho_mu_s must be in (-1, 1).")
    elif cfg.drift_model == "regime":
        if cfg.regime_params is None:
            raise ValueError("Regime parameters required for regime drift model.")
        mu_reg = cfg.regime_params.mu_regimes
        P = cfg.regime_params.transition_matrix
        if P.shape[0] != P.shape[1] or P.shape[0] != mu_reg.shape[0]:
            raise ValueError("Transition matrix must be square and match mu_regimes length.")
        if not np.allclose(P.sum(axis=1), 1.0):
            raise ValueError("Transition matrix rows must sum to 1.")
        if cfg.regime_params.vol_regimes is not None:
            if cfg.regime_params.vol_regimes.shape != mu_reg.shape:
                raise ValueError("vol_regimes must match mu_regimes length.")
    else:
        raise ValueError(f"Unsupported drift model: {cfg.drift_model}")


def black_scholes_call(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if vol <= 0:
        intrinsic = max(S - K, 0.0)
        return intrinsic
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return math.exp(-q * T) * S * norm.cdf(d1) - math.exp(-r * T) * K * norm.cdf(d2)


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float) -> float:
    if T <= 0 or price <= 0:
        return 0.0

    def f(sig: float) -> float:
        return black_scholes_call(S, K, T, r, q, sig) - price

    try:
        return brentq(f, 1e-6, 5.0, maxiter=200, rtol=1e-6)
    except ValueError:
        return 0.0


def simulate_paths(cfg: SimConfig, risk_neutral: bool = False) -> dict[str, np.ndarray]:
    """Simulate stock, variance, and drift/regime paths.

    Returns a dict with keys: ``S`` (prices), ``v`` (variance), ``mu`` (drift),
    ``Z`` (regime states, optional), and ``time`` (grid).
    """

    validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.T_max / cfg.n_steps
    sqrt_dt = math.sqrt(dt)
    S_paths = np.empty((cfg.n_steps + 1, cfg.n_paths))
    v_paths = np.empty_like(S_paths)
    mu_paths = np.empty_like(S_paths)
    Z_paths = None

    S_paths[0] = cfg.S0
    v0 = cfg.sigma0 * cfg.sigma0
    v_paths[0] = v0
    if cfg.drift_model == "ou" and cfg.ou_params:
        mu_paths[0] = cfg.ou_params.mu0
    else:
        mu_paths[0] = 0.0

    if cfg.drift_model == "regime":
        reg = cfg.regime_params  # type: ignore[assignment]
        states = rng.integers(0, reg.mu_regimes.shape[0], size=cfg.n_paths)
        Z_paths = np.empty_like(S_paths, dtype=int)
        Z_paths[0] = states
        mu_paths[0] = reg.mu_regimes[states]

    for t in range(cfg.n_steps):
        z1 = rng.standard_normal(cfg.n_paths)
        z2 = rng.standard_normal(cfg.n_paths)
        z_var = cfg.heston.rho * z1 + math.sqrt(1 - cfg.heston.rho**2) * z2

        if cfg.drift_model == "ou":
            ou = cfg.ou_params  # type: ignore[assignment]
            z_mu_indep = rng.standard_normal(cfg.n_paths)
            z_mu = ou.rho_mu_s * z1 + math.sqrt(1 - ou.rho_mu_s**2) * z_mu_indep
            mu_prev = mu_paths[t]
            mu_next = mu_prev + ou.alpha * (ou.mu_bar - mu_prev) * dt + ou.sigma_mu * sqrt_dt * z_mu
            mu_paths[t + 1] = mu_next
            drift_t = mu_prev if not risk_neutral else cfg.r - cfg.q
            drift_next = mu_next if not risk_neutral else cfg.r - cfg.q
        else:
            reg = cfg.regime_params  # type: ignore[assignment]
            states = Z_paths[t]
            mu_curr = reg.mu_regimes[states]
            drift_t = mu_curr if not risk_neutral else cfg.r - cfg.q

            U = rng.random(cfg.n_paths)
            cum_P = reg.transition_matrix.cumsum(axis=1)
            cum_rows = cum_P[states]
            cum_rows[:, -1] = 1.0  # guard against row sums slightly below 1
            next_states = np.sum(U[:, None] > cum_rows, axis=1)
            next_states = np.minimum(next_states, reg.mu_regimes.shape[0] - 1)
            Z_paths[t + 1] = next_states
            mu_next = reg.mu_regimes[next_states]
            mu_paths[t + 1] = mu_next
            drift_next = mu_next if not risk_neutral else cfg.r - cfg.q

        v_prev = v_paths[t]
        theta_t = cfg.heston.theta
        if cfg.drift_model == "regime" and cfg.regime_params and cfg.regime_params.vol_regimes is not None:
            theta_t = cfg.regime_params.vol_regimes[states] ** 2
        v_trunc = np.maximum(v_prev, 0.0)
        v_euler = v_prev + cfg.heston.kappa * (theta_t - v_trunc) * dt + cfg.heston.xi * np.sqrt(v_trunc) * sqrt_dt * z_var
        v_next = np.maximum(v_euler, 0.0)
        v_paths[t + 1] = v_next

        # Log-Euler for S for stability
        drift_use = drift_t if risk_neutral else (drift_t + drift_next) * 0.5
        diffusion = np.sqrt(v_trunc)
        S_paths[t + 1] = S_paths[t] * np.exp((drift_use - 0.5 * diffusion**2) * dt + diffusion * sqrt_dt * z1)

    return {"S": S_paths, "v": v_paths, "mu": mu_paths, "Z": Z_paths, "time": np.linspace(0.0, cfg.T_max, cfg.n_steps + 1)}


def mc_prices(S_paths: np.ndarray, time_grid: np.ndarray, K: Sequence[float], T: Sequence[float], r: float, q: float) -> np.ndarray:
    """Monte Carlo call prices for strike/maturity grid."""

    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    prices = np.zeros((T_arr.size, K_arr.size))
    for j, Tj in enumerate(T_arr):
        idx = np.searchsorted(time_grid, Tj)
        if idx >= len(time_grid):
            idx = -1
        ST = S_paths[idx]
        disc = math.exp(-r * Tj)
        for i, Ki in enumerate(K_arr):
            payoff = np.maximum(ST - Ki, 0.0)
            prices[j, i] = disc * payoff.mean()
    return prices


def implied_vol_surface(price_surf: np.ndarray, S0: float, K: Sequence[float], T: Sequence[float], r: float, q: float) -> np.ndarray:
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    iv = np.zeros_like(price_surf)
    for j, Tj in enumerate(T_arr):
        for i, Ki in enumerate(K_arr):
            iv[j, i] = implied_vol(price_surf[j, i], S0, Ki, Tj, r, q)
    return iv


def build_surface_demo() -> None:
    """Run a demo with equity-like defaults and plot outputs."""

    S0 = 100.0
    r = 0.02
    q = 0.0
    sigma0 = 0.25
    cfg = SimConfig(
        S0=S0,
        r=r,
        q=q,
        sigma0=sigma0,
        T_max=1.0,
        n_steps=504,
        n_paths=20000,
        heston=HestonParams(kappa=2.5, theta=0.20**2, xi=0.45, rho=-0.7),
        drift_model="ou",
        ou_params=OUDriftParams(mu0=0.06, alpha=1.5, mu_bar=0.04, sigma_mu=0.05, rho_mu_s=-0.3),
        seed=123,
    )

    sim_P = simulate_paths(cfg, risk_neutral=False)
    sim_Q = simulate_paths(cfg, risk_neutral=True)

    maturities = np.array([0.25, 0.5, 0.75, 1.0])
    strikes = np.linspace(60, 140, 17)

    price_surface = mc_prices(sim_Q["S"], sim_Q["time"], strikes, maturities, r, q)
    iv_surface_mat = implied_vol_surface(price_surface, S0, strikes, maturities, r, q)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Drift paths
    axes[0, 0].plot(sim_P["time"], sim_P["mu"][:, :20])
    axes[0, 0].set_title("Sample drift paths (mu_t)")
    axes[0, 0].set_xlabel("Time (years)")
    axes[0, 0].set_ylabel("Drift")

    # Vol paths
    axes[0, 1].plot(sim_P["time"], np.sqrt(sim_P["v"][:, :20]))
    axes[0, 1].set_title("Sample volatility paths (sqrt(v_t))")
    axes[0, 1].set_xlabel("Time (years)")
    axes[0, 1].set_ylabel("Volatility")

    # Smiles for select maturities
    for Tj, iv_row in zip(maturities, iv_surface_mat):
        axes[1, 0].plot(strikes, iv_row, label=f"T={Tj:.2f}y")
    axes[1, 0].set_title("Implied vol smiles")
    axes[1, 0].set_xlabel("Strike")
    axes[1, 0].set_ylabel("Implied vol")
    axes[1, 0].legend()

    # ATM term structure
    atm_idx = np.argmin(np.abs(strikes - S0))
    atm_iv = iv_surface_mat[:, atm_idx]
    axes[1, 1].plot(maturities, atm_iv, marker="o")
    axes[1, 1].set_title("ATM IV term structure")
    axes[1, 1].set_xlabel("Maturity (years)")
    axes[1, 1].set_ylabel("ATM implied vol")

    fig.suptitle("Dynamic Drift/Vol MC + Implied Vol Surface", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    build_surface_demo()
