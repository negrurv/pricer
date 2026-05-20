import pricer_core
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import time
import warnings

# Silence the SciPy deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Global counter to track Powell's progress through the noise
eval_count = 0 

# ============================================================================
# 1. Market Data Fetcher (Surgical ATM Extraction)
# ============================================================================
def get_spy_options_data():
    """Fetches real SPY options data and isolates the 5 At-The-Money strikes."""
    print("Fetching live SPY options data from Yahoo Finance...")
    spy = yf.Ticker("SPY")
    S0 = spy.history(period="1d")['Close'].iloc[-1]
    
    # Get the nearest expiration date that is roughly 30 days out
    expirations = spy.options
    target_exp = expirations[2] 
    
    chain = spy.option_chain(target_exp)
    calls = chain.calls
    
    # Filter for baseline liquidity
    calls = calls[calls['volume'] > 100] 
    
    # THE PIVOT: Sort by absolute distance to Spot and take only the 5 closest strikes
    calls['dist'] = abs(calls['strike'] - S0)
    calls = calls.sort_values('dist').head(5)
    
    # Time to maturity in years (approximate for the script)
    T = 30.0 / 365.0 
    
    market_data = []
    for _, row in calls.iterrows():
        # Use the mid-price of the bid-ask spread as the "fair market price"
        mid_price = (row['bid'] + row['ask']) / 2.0
        market_data.append({'K': row['strike'], 'Price': mid_price, 'T': T})
        
    return S0, market_data

# ============================================================================
# 2. The Loss Function (Mean Squared Error)
# ============================================================================
def heston_loss(params_array, S0, market_data, cpp_params):
    """
    Cost function for Powell's Method.
    params_array = [v0, kappa, theta, sigma, rho]
    """
    global eval_count
    
    # Unpack the optimizer's current guess
    cpp_params.v0 = params_array[0]
    cpp_params.kappa = params_array[1]
    cpp_params.theta = params_array[2]
    cpp_params.sigma = params_array[3]
    cpp_params.rho = params_array[4]
    
    total_error = 0.0
    
    # Batch price the 5 strikes using the C++ multithreaded engine
    for data in market_data:
        cpp_params.K = data['K']
        cpp_params.T = data['T']
        
        # 10,000 paths per strike is fast enough for a 5-strike surface
        results = pricer_core.batch_price_heston(cpp_params, 10000) 
        model_price = np.mean(results)
        
        # Mean Squared Error
        total_error += (model_price - data['Price']) ** 2
        
    mse = total_error / len(market_data)
    
    eval_count += 1
    # Print progress every 10 evaluations so we know the engine is breathing
    if eval_count % 10 == 0:
        print(f"Eval {eval_count:3d} | Current MSE: {mse:.4f} | Guess: v0={params_array[0]:.4f}, rho={params_array[4]:.4f}")
        
    return mse

# ============================================================================
# 3. The Powell Optimizer
# ============================================================================
def calibrate():
    S0, market_data = get_spy_options_data()
    print(f"Calibrating to {len(market_data)} SPY strikes. Spot Price: ${S0:.2f}")
    
    # Setup the C++ parameter bridge
    cpp_params = pricer_core.HestonParams()
    cpp_params.S0 = S0
    cpp_params.r = 0.05 # 5% risk-free rate assumption
    cpp_params.steps = 50 # Discretization steps
    
    # Initial Guess: [v0, kappa, theta, sigma, rho]
    initial_guess = [0.04, 2.0, 0.04, 0.2, -0.5]
    
    # Box Constraints
    bounds = (
        (0.001, 1.0),   # v0: Must be positive
        (0.01, 10.0),   # kappa: Mean reversion speed
        (0.001, 1.0),   # theta: Long-term variance
        (0.01, 2.0),    # sigma: Vol of vol
        (-0.99, 0.99)   # rho: Correlation 
    )
    
    print("\nBeginning Powell Optimization (Derivative-Free)...")
    start_time = time.perf_counter()
    
    # Run the optimizer
    result = minimize(
        fun=heston_loss,
        x0=initial_guess,
        args=(S0, market_data, cpp_params),
        method='Powell', # The noise-resilient algorithm
        bounds=bounds,
        options={'maxiter': 50, 'disp': True}
    )
    
    end_time = time.perf_counter()
    
    print("\n" + "="*50)
    print("CALIBRATION COMPLETE")
    print("="*50)
    print(f"Time Elapsed: {(end_time - start_time):.2f} seconds")
    print(f"Final Loss (MSE): {result.fun:.4f}")
    print("\nCalibrated Heston Parameters:")
    print(f"v0 (Initial Variance):  {result.x[0]:.4f}")
    print(f"kappa (Mean Reversion): {result.x[1]:.4f}")
    print(f"theta (Long-Term Var):  {result.x[2]:.4f}")
    print(f"sigma (Vol of Vol):     {result.x[3]:.4f}")
    print(f"rho (Correlation):      {result.x[4]:.4f}")

if __name__ == "__main__":
    calibrate()
