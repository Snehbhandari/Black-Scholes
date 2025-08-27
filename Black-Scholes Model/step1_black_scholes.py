import numpy as np 
from scipy.stats import norm 
import matplotlib.pyplot as plt 


def _validate_inputs(spot_price: float, strike_price: float, time_to_maturity_years: float, risk_free_rate: float, volatility: float) -> None:
    if spot_price <= 0 or strike_price <= 0: 
        raise ValueError("S and K must be positive.") 
    if time_to_maturity_years < 0: 
        raise ValueError("T must be non-negative (in years).") 
    if volatility <= 0: 
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
    _validate_inputs(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility)
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


if __name__ == "__main__":
    # Take user input 
    S = float(input("Enter spot price: "))
    K = float(input("Enter strike price: "))
    T = float(input("Enter time to maturity: "))
    r = float(input("Enter risk-free rate: "))
    sigma = float(input("Enter volatility: "))
    opt = input("Enter option type (call/put): ").lower()


    price = black_scholes_price(S, K, T, r, sigma, opt) 
    greeks = black_scholes_greeks(S, K, T, r, sigma, opt) 

    print(f"Black–Scholes {opt} price: {price:.4f}") 
    for name, value in greeks.items(): 
        print(f"{name}: {value:.6f}") 

    # Plots
    strikes = np.linspace(S-100, S+100, 101) 
    plot_price_vs_strike(S, T, r, sigma, strikes) 
    plot_greeks_vs_strike(S, T, r, sigma, strikes, option_type=opt)  

    maturities = np.linspace(0.05, 2.0, 80)
    plot_greeks_vs_maturity(S, K, r, sigma, maturities, option_type=opt)

