import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.stats import norm
import matplotlib.pyplot as plt


try:
    import yfinance as yf 
except Exception as exc:  # pragma: no cover 
    yf = None 
    warnings.warn( 
        "yfinance is not installed. Install with: pip install yfinance", 
        RuntimeWarning, 
    ) 


# Reuse pricing from Step 1 without importing user as a package 
def _compute_d1_d2(spot_price: float, strike_price: float, time_to_maturity_years: float, risk_free_rate: float, volatility: float) -> Tuple[float, float]:
    numerator = np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity_years
    denominator = volatility * np.sqrt(time_to_maturity_years) 
    d1 = numerator / denominator
    d2 = d1 - denominator
    return d1, d2


def black_scholes_price(spot_price: float, strike_price: float, time_to_maturity_years: float, risk_free_rate: float, volatility: float, option_type: str = "call") -> float:
    d1, d2 = _compute_d1_d2(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility)
    if option_type == "call": 
        return spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(d2)
    elif option_type == "put": 
        return strike_price * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
    else: 
        raise ValueError("option_type must be 'call' or 'put'") 

# Explain 
def implied_volatility_brent(market_price: float, spot_price: float, strike_price: float, time_to_maturity_years: float, risk_free_rate: float, option_type: str = "call", sigma_lower: float = 1e-6, sigma_upper: float = 5.0, tol: float = 1e-8, maxiter: int = 200) -> Optional[float]:
    from scipy.optimize import brentq

    # finding IV 
    def objective(volatility: float) -> float:
        return black_scholes_price(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility, option_type) - market_price

    try:
        f_low = objective(sigma_lower)
        f_up = objective(sigma_upper)
        expansion_attempts = 0
        while f_low * f_up > 0 and expansion_attempts < 8:
            sigma_upper *= 2.0
            f_up = objective(sigma_upper)
            expansion_attempts += 1
        if f_low * f_up > 0:
            return None
        return float(brentq(objective, sigma_lower, sigma_upper, xtol=tol, maxiter=maxiter))
    except Exception:
        return None

# Explain 
@dataclass
class IVSurface:
    symbol: str
    spot: float
    risk_free_rate: float
    points: pd.DataFrame  # columns: [strike, T, iv, option_type]
    interpolator_linear: LinearNDInterpolator
    # nearest fallback uses griddata with method='nearest'

    def lookup_iv(self, strike_price: float, maturity_years: float) -> Optional[float]:
        iv = float(self.interpolator_linear(strike_price, maturity_years))
        if not np.isfinite(iv):
            # fallback to nearest
            pts = self.points[["strike", "T"]].to_numpy()
            vals = self.points["iv"].to_numpy()
            iv = griddata(pts, vals, (strike_price, maturity_years), method="nearest")
            if iv is None or not np.isfinite(iv):
                return None
            return float(iv)
        return iv

    def price(self, strike_price: float, maturity_years: float, option_type: str = "call") -> Optional[float]:
        iv = self.lookup_iv(strike_price, maturity_years)
        if iv is None:
            return None
        return black_scholes_price(self.spot, strike_price, maturity_years, self.risk_free_rate, iv, option_type)


def _now_utc_pd_timestamp() -> pd.Timestamp:
    # Use timezone-aware now to avoid localizing an already tz-aware Timestamp
    return pd.Timestamp.now(tz="UTC")


def _years_between(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float:
    return max(1e-9, (end_ts - start_ts).days / 365.0 + (end_ts - start_ts).seconds / (365.0 * 24 * 3600))

# Explain 
def fetch_option_market_data(symbol: str) -> Tuple[float, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("yfinance is required. Install with: pip install yfinance")

    ticker = yf.Ticker(symbol)
    # check if ticker is valid 
    if not ticker.info.get("regularMarketPrice"):
        raise RuntimeError("Invalid ticker.")
    spot = None
    try:
        spot = float(ticker.fast_info.get("last_price"))
    except Exception:
        try:
            spot = float(ticker.info.get("regularMarketPrice"))
        except Exception:
            hist = ticker.history(period="1d")
            if len(hist) == 0:
                raise RuntimeError("Could not determine spot price.")
            spot = float(hist["Close"].iloc[-1])

    expirations = ticker.options or []
    # print(expirations)
    rows = []
    for exp in expirations:
        try:
            # option chain for every expiration
            chain = ticker.option_chain(exp) 
            # print(chain) 
            # print("<-------------------------------->") 
        except Exception: 
            continue 
        for option_df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
            if option_df is None or option_df.empty:
                continue
            for _, row in option_df.iterrows():
                strike = float(row.get("strike"))
                bid = row.get("bid")
                ask = row.get("ask")
                last = row.get("lastPrice")
                bid = float(bid) if pd.notna(bid) else np.nan
                ask = float(ask) if pd.notna(ask) else np.nan
                last = float(last) if pd.notna(last) else np.nan
                if np.isfinite(bid) and np.isfinite(ask) and bid >= 0 and ask > 0:
                    price = (bid + ask) / 2.0
                elif np.isfinite(last) and last > 0:
                    price = last
                else:
                    continue
                rows.append({
                    "option_type": opt_type,
                    "strike": strike,
                    "expiration": pd.to_datetime(exp).tz_localize("UTC"),
                    "market_price": float(price),
                })
    df = pd.DataFrame(rows)
    # print(df)
    if df.empty:
        raise RuntimeError("No option quotes retrieved from Yahoo Finance.")
    return spot, df


def build_iv_surface(symbol: str, risk_free_rate: float = 0.02, min_price: float = 0.25, max_iv: float = 5.0) -> IVSurface:
    spot, quotes = fetch_option_market_data(symbol)
    asof = _now_utc_pd_timestamp()
    quotes = quotes.copy() # why do we need to copy? 
    quotes["T"] = quotes["expiration"].apply(lambda dt: _years_between(asof, dt)) 
    quotes = quotes[quotes["T"] > 1e-6] # why do we need to filter? 

    ivs = []
    for _, r in quotes.iterrows():
        market_price = float(r["market_price"])
        if market_price < min_price:
            continue
        iv = implied_volatility_brent(
            market_price=market_price,
            spot_price=spot,
            strike_price=float(r["strike"]),
            time_to_maturity_years=float(r["T"]),
            risk_free_rate=risk_free_rate,
            option_type=str(r["option_type"]),
        )
        if iv is None or not np.isfinite(iv) or iv <= 0 or iv > max_iv:
            continue
        ivs.append({
            "option_type": r["option_type"],
            "strike": float(r["strike"]),
            "T": float(r["T"]),
            "iv": float(iv),
        })

    iv_df = pd.DataFrame(ivs)
    if iv_df.empty:
        raise RuntimeError("No valid implied volatilities solved. Try adjusting filters.")

    pts = iv_df[["strike", "T"]].to_numpy() 
    vals = iv_df["iv"].to_numpy() # why do we need to convert to numpy? 
    # why do we need to use LinearNDInterpolator? 
    interpolator = LinearNDInterpolator(pts, vals, fill_value=np.nan)
    
    return IVSurface(
        symbol=symbol,
        spot=float(spot),
        risk_free_rate=float(risk_free_rate),
        points=iv_df,
        interpolator_linear=interpolator,
    )


def plot_smile(surface: IVSurface, maturity_years: float, ax: Optional[plt.Axes] = None) -> plt.Axes:
    df = surface.points.copy()
    # pick near maturity
    df["dist"] = np.abs(df["T"] - maturity_years)
    bucket = df.nsmallest(2000, "dist")  # take nearest sample if sparse
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(bucket["strike"], bucket["iv"], s=10, alpha=0.6)
    ax.axvline(surface.spot, color="gray", linestyle=":", label="Spot")
    ax.set_title(f"IV smile near T={maturity_years:.2f}y ({surface.symbol})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def plot_term_structure(surface: IVSurface, strike_price: float, ax: Optional[plt.Axes] = None) -> plt.Axes:
    df = surface.points.copy()
    df["dist"] = np.abs(df["strike"] - strike_price)
    bucket = df.nsmallest(2000, "dist")
    grouped = bucket.sort_values("T")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped["T"], grouped["iv"], marker="o", linestyle="-")
    ax.set_title(f"IV term structure near K={strike_price:.2f} ({surface.symbol})")
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Implied Volatility")
    ax.grid(True, alpha=0.25)
    return ax


def plot_surface_3d(surface: IVSurface) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    df = surface.points
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["strike"], df["T"], df["iv"], c=df["iv"], cmap="viridis", s=10)
    ax.set_title(f"Implied Volatility Surface ({surface.symbol})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("IV")
    plt.tight_layout()
    plt.show()

# Explain 
if __name__ == "__main__":
    import sys 

    symbol = None 
    if len(sys.argv) >= 2: 
        symbol = sys.argv[1]
    if not symbol:
       
        symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    try:
        surf = build_iv_surface(symbol, risk_free_rate=0.02)
        print(f"Built IV surface for {symbol}. Spot={surf.spot:.4f}, points={len(surf.points)}")
        # Quick demo lookups
        sample = surf.points.sample(min(5, len(surf.points)), random_state=0)
        for _, r in sample.iterrows():
            iv = surf.lookup_iv(r["strike"], r["T"])
            px_call = surf.price(r["strike"], r["T"], option_type="call")
            print(f"K={r['strike']:.2f}, T={r['T']:.3f}y -> IV~{iv:.4f}, Call~{px_call:.4f}")

        # Optional plots
        try:
            midT = float(np.median(surf.points["T"]))
            plot_smile(surf, maturity_years=midT)
            plot_term_structure(surf, strike_price=float(np.median(surf.points["strike"])))
            plt.show()
        except Exception:
            pass
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


