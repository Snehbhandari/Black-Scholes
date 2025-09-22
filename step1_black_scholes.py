from math import nan
from pickle import FALSE, TRUE
import numpy as np 
from pandas.core.arrays import period
from scipy.stats import norm 
import matplotlib.pyplot as plt 
import datetime as dt   
import yfinance as yf 
import pandas as pd 
from scipy.optimize import brentq
from typing import Optional, Union, Tuple


def _validate_inputs(spot_price: float, strike_price: float, time_to_maturity_years: float, risk_free_rate: float, volatility: float) -> None:
    # print(volatility) 
    # print(type(volatility)) 
    if spot_price <= 0 or strike_price <= 0: 
        raise ValueError("S and K must be positive.") 
    if time_to_maturity_years < 0: 
        raise ValueError("T must be non-negative (in years).") 
    if volatility <= 0: 
        # print(volatility)
        # print(type(volatility))
        raise ValueError("sigma must be positive (annualized decimal).")


def _compute_d1_d2(spot_price: float, strike_price: float, time_to_maturity_years: float, risk_free_rate: float, volatility: float) -> tuple[float, float]:
    numerator = np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity_years
    denominator = volatility * np.sqrt(time_to_maturity_years)
    d1 = numerator / denominator
    d2 = d1 - denominator
    return d1, d2


def black_scholes_price(spot_price: float, strike_price: float, time_to_maturity_years: float, risk_free_rate: float, volatility: float, option_type: str = "call") -> float:
    """
    Price a European call or put via the Black–Scholes formula (no dividends).

    Parameters are in natural units: r and sigma are decimals (e.g., 0.02, 0.20), T in years.
    Returns a price in the same currency as S and K.
    """
    _validate_inputs(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility) 
    # if time to maturity is equal to 0, then return intrinsic value = stock price 
    if time_to_maturity_years <= 0:
        # immediate payoff (European): no time value 
        return max(spot_price - strike_price, 0.0) if option_type.lower()=="call" else max(strike_price - spot_price, 0.0)
    d1, d2 = _compute_d1_d2(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility)
    option = option_type.lower()
    if option == "call":
        return spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(d2)
    elif option == "put": 
        return strike_price * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
    else: 
        raise ValueError("option_type must be 'call' or 'put'") 


def black_scholes_greeks(spot_price: float, strike_price: float, time_to_maturity_years: float, risk_free_rate: float, volatility: float, option_type: str = "call") -> dict:
    """
    Compute key Greeks for European options under Black–Scholes (annual units for Theta and Rho).

    Returns a dict with Delta, Gamma, Vega, Theta, Rho. Vega is per 1.00 change in sigma (not per 1%).
    """
    _validate_inputs(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility= volatility)
    if time_to_maturity_years <= 0:
        # immediate payoff (European): no time value 
        return {"Delta": 1.0 if spot_price > strike_price else 0.0,
            "Gamma": 0.0,
            "Vega": 0.0,
            "Theta": 0.0,
            "Rho": 0.0} 
    
    d1, d2 = _compute_d1_d2(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility)
    standard_normal_pdf_at_d1 = norm.pdf(d1) 
    option = option_type.lower() 

    if option == "call": 
        delta = norm.cdf(d1) 
        theta = -(
            spot_price * standard_normal_pdf_at_d1 * volatility / (2.0 * np.sqrt(time_to_maturity_years))
        ) - risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(d2)
        rho = strike_price * time_to_maturity_years * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(d2)
    elif option == "put":
        delta = norm.cdf(d1) - 1.0 
        theta = -(
            spot_price * standard_normal_pdf_at_d1 * volatility / (2.0 * np.sqrt(time_to_maturity_years))
        ) + risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(-d2)
        rho = -strike_price * time_to_maturity_years * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = standard_normal_pdf_at_d1 / (spot_price * volatility * np.sqrt(time_to_maturity_years))
    vega = spot_price * standard_normal_pdf_at_d1 * np.sqrt(time_to_maturity_years)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho,
    } 

# Get strikes and expirations given a ticker symbol - ToDo 
# def get_all_strikes_and_expirations(ticker: str) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Get all strikes and expirations for a given ticker from yFinance.
#     """
#     data = yf.Ticker(ticker).option_chain() 

#     return data.strikes, data.expirations  

def plot_price_vs_strike(spot_price: float, time_to_maturity_years: float, risk_free_rate: float, volatility: float, strikes: np.ndarray) -> None:
    call_prices = [black_scholes_price(spot_price, k, time_to_maturity_years, risk_free_rate, volatility, "call") for k in strikes]
    put_prices = [black_scholes_price(spot_price, k, time_to_maturity_years, risk_free_rate, volatility, "put") for k in strikes]

    plt.figure(figsize=(8, 5)) 
    plt.plot(strikes, call_prices, label="Call price")
    plt.plot(strikes, put_prices, label="Put price")
    plt.axvline(x=spot_price, color="gray", linestyle=":", label="S (spot)")
    plt.title("Black–Scholes: Price vs Strike")
    plt.xlabel("Strike K")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_greeks_vs_strike(spot_price: float, time_to_maturity_years: float, risk_free_rate: float, volatility: float, strikes: np.ndarray, option_type: str = "call") -> None:
    deltas, gammas, vegas = [], [], []
    for k in strikes:
        greeks = black_scholes_greeks(spot_price, k, time_to_maturity_years, risk_free_rate, volatility, option_type)
        deltas.append(greeks["Delta"])
        gammas.append(greeks["Gamma"])
        vegas.append(greeks["Vega"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(strikes, deltas)
    axes[0].set_ylabel("Delta")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(f"{option_type.capitalize()} Delta vs Strike")

    axes[1].plot(strikes, gammas)
    axes[1].set_ylabel("Gamma")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title(f"{option_type.capitalize()} Gamma vs Strike")

    axes[2].plot(strikes, vegas)
    axes[2].set_ylabel("Vega")
    axes[2].set_xlabel("Strike K")
    axes[2].grid(True, alpha=0.25)
    axes[2].set_title(f"{option_type.capitalize()} Vega vs Strike")

    plt.tight_layout()
    plt.show()


def plot_greeks_vs_maturity(spot_price: float, strike_price: float, risk_free_rate: float, volatility: float, maturities: np.ndarray, option_type: str = "call") -> None:
    deltas, gammas, vegas = [], [], []
    for t in maturities:
        greeks = black_scholes_greeks(spot_price, strike_price, t, risk_free_rate, volatility, option_type)
        deltas.append(greeks["Delta"])
        gammas.append(greeks["Gamma"])
        vegas.append(greeks["Vega"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(maturities, deltas)
    axes[0].set_ylabel("Delta")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(f"{option_type.capitalize()} Delta vs Maturity")

    axes[1].plot(maturities, gammas)
    axes[1].set_ylabel("Gamma")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title(f"{option_type.capitalize()} Gamma vs Maturity")

    axes[2].plot(maturities, vegas)
    axes[2].set_ylabel("Vega")
    axes[2].set_xlabel("Maturity T (years)")
    axes[2].grid(True, alpha=0.25)
    axes[2].set_title(f"{option_type.capitalize()} Vega vs Maturity")

    plt.tight_layout()

    plt.show() 

def check_available(asset: str) -> bool:
    """
    Checks if an asset is available via the Yahoo Finance API.
        """
    try:
        info = yf.Ticker(asset).history(
            period='10d',
            interval='1d')
        return not info.empty
    except Exception: 
        return FALSE 

# IV Calculation 
def implied_vol(price, S, K, T, r, option_type="call") -> float:
    """Back out IV using Brent's method. Returns np.nan on failure."""
    def f(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, option_type=option_type) - price
    try:
        lo, hi = 1e-6, 5.0
        flo, fhi = f(lo), f(hi)
        tries = 0
        while flo * fhi > 0 and tries < 6:
            hi *= 2.0
            fhi = f(hi)
            tries += 1
        if flo * fhi > 0:
            return np.nan
        return float(brentq(f, lo, hi, xtol=1e-8, maxiter=200))
    except Exception:
        return np.nan
 

# Validating market price - no-arbitrage bounds
def is_valid_price(price, S, K, T, r, option_type="call") -> bool:
    if option_type == "call":
        min_price = max(0.0, S - K * np.exp(-r * T))
        max_price = S
    else:
        min_price = max(0.0, K * np.exp(-r * T) - S)
        max_price = K
    return (min_price <= price <= max_price)

def plot_smile(df, target_expiry, option_type="call", spot=None):
    target = pd.to_datetime(target_expiry, utc=True)
    # pick nearest expiry
    df_ = df[df["type"] == option_type].copy()
    df_["dist"] = (df_["expiry"] - target).abs()
    bucket = df_.nsmallest(2000, "dist")
    plt.figure(figsize=(7,4))
    plt.scatter(bucket["strike"], bucket["IV"], s=12, alpha=0.7)
    if spot is not None:
        plt.axvline(spot, color="gray", linestyle=":", label="Spot")
        plt.legend()
    plt.title(f"IV smile near {target.date()} ({option_type})")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

# plots 

# 1. Smiles/ Skews - 
    

def build_iv_dataframe(all_data: list, asof: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, pd.Timestamp]:
    df = pd.DataFrame(all_data)
    if df.empty:
        raise ValueError("No IV data available to plot.")
    if asof is None:
        asof = pd.Timestamp.now(tz="UTC")
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    df["T"] = (df["expiry"] - asof).dt.total_seconds() / (365.0 * 24 * 3600.0)
    df = df[np.isfinite(df["IV"]) & np.isfinite(df["strike"]) & (df["T"] > 0)]
    return df, asof


def plot_iv_vs_strike(df_iv: pd.DataFrame, target_expiry: Optional[Union[str, pd.Timestamp]] = None, option_type: str = "call", spot: Optional[float] = None, ax=None):
    if target_expiry is None:
        # default: first expiry
        chosen_expiry = df_iv.sort_values("expiry")["expiry"].iloc[0]
    else:
        target_ts = pd.to_datetime(target_expiry, utc=True)
        # choose closest expiry present in data
        expiries = df_iv["expiry"].dropna().unique()
        diffs = np.abs(pd.to_datetime(expiries) - target_ts)
        chosen_expiry = pd.to_datetime(expiries[np.argmin(diffs)])

    bucket = df_iv[(df_iv["type"] == option_type) & (df_iv["expiry"] == chosen_expiry)].copy()
    if bucket.empty:
        raise ValueError("No IV points for the selected expiry/option type.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(bucket["strike"], bucket["IV"], s=12, alpha=0.7)
    if spot is not None and np.isfinite(spot):
        ax.axvline(float(spot), color="gray", linestyle=":", label="Spot")
        ax.legend()
    ax.set_title(f"IV vs Strike @ {pd.to_datetime(chosen_expiry).date()} ({option_type})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.grid(True, alpha=0.25)
    return ax


def plot_iv_vs_expiry(df_iv: pd.DataFrame, target_strike: Optional[float] = None, option_type: str = "call", spot: Optional[float] = None, ax=None):
    df_type = df_iv[df_iv["type"] == option_type].copy()
    if df_type.empty:
        raise ValueError("No IV points for requested option type.")

    if target_strike is None:
        if spot is not None and np.isfinite(spot):
            # choose strike closest to spot
            strikes = df_type["strike"].to_numpy(dtype=float)
            target_strike = float(strikes[np.argmin(np.abs(strikes - float(spot)))])
        else:
            target_strike = float(df_type["strike"].median())

    df_type["dist"] = np.abs(df_type["strike"].astype(float) - float(target_strike))
    bucket = df_type.nsmallest(2000, "dist").sort_values("T")
    if bucket.empty:
        raise ValueError("No IV points near the selected strike.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bucket["T"], bucket["IV"], marker="o")
    ax.set_title(f"IV vs Expiry near K={float(target_strike):.2f} ({option_type})")
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Implied Volatility")
    ax.grid(True, alpha=0.25)
    return ax


if __name__ == "__main__":
    # Take user input 

    # Take ticker as an input - done 
    while TRUE: 
        ticker = input("Enter ticker (or 'q' to quit): ").strip().upper() 
        if ticker == 'Q':
            raise SystemExit(0)
        if check_available(ticker):
            break
        print("Invalid or unavailable ticker. Please try again.") 

    # Calculate IV for all 

    # Get all chains for the ticker 
    tk = yf.Ticker(ticker) 
    # get all expirations: 
    expirations = tk.options 
    print(expirations) 
    # print(tk.option_chain(expirations[0]).calls) 

    # spot prices 
    S = tk.history(period = "1d")["Close"].iloc[-1] 

    # risk free rate 
    R = 0.05 

    all_data = [] 
    asof = pd.Timestamp.now(tz="UTC")
    for exp in expirations[:2]: 
        chain = tk.option_chain(exp) 
        exp_ts = pd.to_datetime(exp, utc=True)
        T = max(1e-6, (exp_ts - asof).total_seconds() / (365.0 * 24 * 3600.0))
        print(T) 
        for option_type, df in zip(["call", "put"], [chain.calls, chain.puts]): 
            # robust mid price with fallback
            df = df.copy()
            bid = pd.to_numeric(df.get("bid"), errors="coerce")
            ask = pd.to_numeric(df.get("ask"), errors="coerce")
            last = pd.to_numeric(df.get("lastPrice"), errors="coerce")
            df["mid"] = np.where(np.isfinite(bid) & np.isfinite(ask) & (ask > 0), (bid + ask) / 2.0, last)
            df = df[np.isfinite(df["mid"]) & (df["mid"] > 0)]
            print("****")
            for _, row in df.iterrows(): 
                price_mid = float(row["mid"])
                K = float(row["strike"]) 
                if is_valid_price(price_mid, S, K, T, R, option_type): 
                    iv = implied_vol(price_mid, S, K, T, R, option_type) 
                else: 
                    iv = np.nan 
                if np.isfinite(iv):
                    print(f"IV: {iv:.6f}")
                    all_data.append({ 
                        "expiry": exp,
                        "type": option_type,
                        "strike": K,
                        "mid": price_mid,
                        "IV": iv
                    }) 
    print(all_data) 

    # Build DataFrame and plot with helper functions
    if all_data:
        df_iv, asof = build_iv_dataframe(all_data, asof)
        plot_iv_vs_strike(df_iv, target_expiry=None, option_type="call", spot=S)
        plot_iv_vs_expiry(df_iv, target_strike=None, option_type="call", spot=S)
        plt.tight_layout()
        plt.show()

    all_data = pd.DataFrame(all_data)

    # Plotting the data - IV vs Strike 
    plt.figure(figsize=(10, 6))
    plt.scatter(all_data["strike"], all_data["IV"], alpha=0.5)
    plt.xlabel("Strike")
    plt.ylabel("IV")
    plt.title("IV vs Strike")
    plt.show() 

    # Plotting the data 
    plt.figure(figsize=(10, 6))
    plt.scatter(all_data["expiry"], all_data["IV"], alpha=0.5)
    plt.xlabel("Strike")
    plt.ylabel("IV")
    plt.title("IV vs Expiry")
    plt.show() 

    plot_smile(all_data, expirations[0], option_type="call", spot=S)

    plot_smile(all_data, expirations[0], option_type="put", spot=S)

    # 1. User inputs Strike or not 
    # strike_input = float(input("Enter strike price or 'all' to get all strikes: ").strip())

    # # 2. User inputs expiration or not 
    # expiration_input = input("Enter expiration date (YYYY-MM-DD) or 'all' to get all expirations: ").strip() 
    # # check validity of expiration date 

    # # 3. call/ puts 
    # option_type = input("Enter option type (call/put): ").strip().lower() 
    # check validity of option type 

