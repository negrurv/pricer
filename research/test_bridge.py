import pricer_core
import time
import numpy as np 

params = pricer_core.HestonParams()
params.S0 = 100.0
params.v0 = 0.04
params.r = 0.05
params.kappa = 2.0
params.theta = 0.04
params.sigma = 0.1
params.rho = -0.7
params.T = 1.0
params.K = 100.0
params.steps = 365

print("Pushing 100,000 contracts to the C++ HPC Engine...")

start = time.perf_counter()
results = pricer_core.batch_price_heston(params, 100_000)
end = time.perf_counter()

# The Mathematical Fix
fair_value = np.mean(results)
std_error = np.std(results) / np.sqrt(len(results))

print(f"Batch completed in {(end - start) * 1000:.2f} ms")
print(f"Heston Option Fair Value: ${fair_value:.4f} ± {std_error:.4f}")
