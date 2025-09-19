# Black–Scholes Model (V1) 

This has a standalone, offline implementation of the base Black–Scholes model for European options. 

## What this file does 
- Computes European call/put prices using the Black–Scholes formula. 
- Calculates key Greeks: Delta, Gamma, Theta, Vega, Rho 
- Provides quick visualizations:
  - Price vs strike
  - Greeks vs strike 
  - Greeks vs maturity 

## Purpose and use
- **Purpose**: Serve as a clean, testable baseline for European option pricing and risk sensitivities. 
- **Use cases**: 
  - Compute option price and Greeks from given inputs 


## File 
- How you can use the file: `step1_black_scholes.py` 
  - `black_scholes_price(S, K, T, r, sigma, option_type='call')` 
  - `black_scholes_greeks(S, K, T, r, sigma, option_type='call')` 
  - `plot_price_vs_strike(...)`, `plot_greeks_vs_strike(...)`, `plot_greeks_vs_maturity(...)` 
  Where
  - **S**: current spot price of the underlying asset
  - **K**: option strike price 
  - **T**: time to maturity in years 
  - **r**: risk‑free rate (annualized, decimal) 
  - **σ (sigma)**: volatility (annualized, decimal) 
  - **option_type**: `'call'` or `'put'` 


## Notes
- All functions use natural units: `r` and `σ` are decimals (e.g., 0.02 and 0.20), and `T` is in years. 
- Theta and Rho are returned in annual units; rescale if you need per‑day or per‑bp.
  

### Step 2: Implied Volatility Surface (Online)
- `step2_iv_surface.py`
  - Fetches option chains from Yahoo Finance
  - Solves implied vols per strike/maturity (Brent method)
  - Builds a 2D IV surface interpolator and exposes lookup and pricing 
  - Optional plots: smile, term structure, 3D surface 

Run example:
```bash
python3 "Black-Scholes Model/step2_iv_surface.py" AAPL
```
Dependencies for Step 2:
```bash
pip install yfinance pandas matplotlib scipy
```


## Quick start
From the project root:
```bash
python3 -c "from Black_Scholes_Model.step1_black_scholes import black_scholes_price; print(black_scholes_price(225, 215, 0.92, 0.0426, 0.3))"
```
Or run the built‑in demo (opens plots):
```bash
python3 "Black-Scholes Model/step1_black_scholes.py"
```


## Example
```python
from step1_black_scholes import black_scholes_price, black_scholes_greeks

S, K, T, r, sigma = 225, 215, 0.92, 0.0426, 0.30
print(black_scholes_price(S, K, T, r, sigma, 'put'))
print(black_scholes_greeks(S, K, T, r, sigma, 'put'))
```

## Dependencies
```bash
pip install numpy scipy matplotlib
```

 
