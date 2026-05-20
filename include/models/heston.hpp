#pragma once

#include <cmath>
#include <algorithm>
#include "../math/random.hpp"

#ifdef __ARM_NEON
#include <arm_neon.h> // The Apple Silicon / ARM64 SIMD header
#endif

namespace pricer::models {

// 1. The CRTP Base Class
template <typename Derived>
struct BasePricer {
    inline double price(math::Xoshiro256& prng) const {
        return static_cast<const Derived*>(this)->price_impl(prng);
    }
};

// 2. Data Structure
// Removed alignas(64) to rule out alignment-related crashes on ARM64
struct HestonParams {
    double S0;
    double v0;
    double r;
    double kappa;
    double theta;
    double sigma;
    double rho;
    double T;
    double K;
    uint32_t steps;
};

// 3. The Heston Implementation
struct HestonCallPricer : public BasePricer<HestonCallPricer> {
    const HestonParams params;

    const double dt;
    const double sqrt_dt;
    const double rho_complement;

    explicit HestonCallPricer(const HestonParams& p) 
        : params(p), 
          dt(p.T / static_cast<double>(p.steps)),
          sqrt_dt(std::sqrt(dt)),
          rho_complement(std::sqrt(1.0 - p.rho * p.rho)) {}

    inline double price_impl(math::Xoshiro256& prng) const {
        double X_t = std::log(params.S0);
        double v_t = params.v0;

        for (uint32_t i = 0; i < params.steps; ++i) {
            const double v_trunc = v_t > 0.0 ? v_t : 0.0;
            const double sqrt_v_trunc = std::sqrt(v_trunc);
            const auto [Z1, Z2] = prng.next_normal_pair();

            const double dW_S = Z1;
            const double dW_v = params.rho * Z1 + rho_complement * Z2;

            X_t += (params.r - 0.5 * v_trunc) * dt + sqrt_v_trunc * sqrt_dt * dW_S;
            v_t += params.kappa * (params.theta - v_trunc) * dt + params.sigma * sqrt_v_trunc * sqrt_dt * dW_v;
        }

        const double S_T = std::exp(X_t);
        const double payoff = std::max(S_T - params.K, 0.0);
        return payoff * std::exp(-params.r * params.T);
    }

#ifdef __ARM_NEON
    // Monte Carlo vectorized pricing using ARM NEON intrinsics
    inline double price_monte_carlo_vectorized(math::Xoshiro256& prng, uint32_t num_simulations) const {
        double total_undiscounted_payoff = 0.0;
        
        // Pre-load constants into 128-bit vector registers (Duplicate to both lanes)
        const float64x2_t v_dt = vdupq_n_f64(dt);
        const float64x2_t v_sqrt_dt = vdupq_n_f64(sqrt_dt);
        const float64x2_t v_r = vdupq_n_f64(params.r);
        const float64x2_t v_kappa = vdupq_n_f64(params.kappa);
        const float64x2_t v_theta = vdupq_n_f64(params.theta);
        const float64x2_t v_sigma = vdupq_n_f64(params.sigma);
        const float64x2_t v_rho = vdupq_n_f64(params.rho);
        const float64x2_t v_rho_comp = vdupq_n_f64(rho_complement);
        const float64x2_t v_zero = vdupq_n_f64(0.0);
        const float64x2_t v_half = vdupq_n_f64(0.5);

        // Process TWO paths per loop iteration
        uint32_t vectorized_simulations = (num_simulations / 2) * 2;
        uint32_t remaining_simulations = num_simulations % 2;

        for (uint32_t p = 0; p < vectorized_simulations; p += 2) {
            float64x2_t v_X = vdupq_n_f64(std::log(params.S0));
            float64x2_t v_v = vdupq_n_f64(params.v0);

            for (uint32_t i = 0; i < params.steps; ++i) {
                // 1. Branchless Full Truncation: max(v, 0.0) for two paths at once
                float64x2_t v_trunc = vmaxnmq_f64(v_v, v_zero);
                float64x2_t v_sqrt_v = vsqrtq_f64(v_trunc); // Hardware-accelerated vector square root

                // 2. Fetch FOUR normal variables (2 for Path A, 2 for Path B)
                auto [Z1_A, Z2_A] = prng.next_normal_pair();
                auto [Z1_B, Z2_B] = prng.next_normal_pair();
                
                // Load them into the 128-bit lanes
                float64x2_t v_Z1 = { Z1_A, Z1_B }; 
                float64x2_t v_Z2 = { Z2_A, Z2_B };

                // 3. Cholesky Decomposition
                float64x2_t v_dW_S = v_Z1;
                // Fused Multiply-Add (FMA): (rho * Z1) + (rho_comp * Z2) in one cycle!
                float64x2_t v_dW_v = vfmaq_f64(vmulq_f64(v_rho, v_Z1), v_rho_comp, v_Z2);

                // 4. Update Log-Price: X += (r - 0.5 * v) * dt + sqrt_v * sqrt_dt * dW_S
                float64x2_t v_drift_X = vmulq_f64(vsubq_f64(v_r, vmulq_f64(v_half, v_trunc)), v_dt);
                float64x2_t v_diff_X = vmulq_f64(vmulq_f64(v_sqrt_v, v_sqrt_dt), v_dW_S);
                v_X = vaddq_f64(v_X, vaddq_f64(v_drift_X, v_diff_X));

                // 5. Update Variance: v += kappa * (theta - v) * dt + sigma * sqrt_v * sqrt_dt * dW_v
                float64x2_t v_drift_v = vmulq_f64(vmulq_f64(v_kappa, vsubq_f64(v_theta, v_trunc)), v_dt);
                float64x2_t v_diff_v = vmulq_f64(vmulq_f64(v_sigma, vmulq_f64(v_sqrt_v, v_sqrt_dt)), v_dW_v);
                v_v = vaddq_f64(v_v, vaddq_f64(v_drift_v, v_diff_v));
            }

            // Accumulate undiscounted payoffs
            total_undiscounted_payoff += std::max(std::exp(vgetq_lane_f64(v_X, 0)) - params.K, 0.0);
            total_undiscounted_payoff += std::max(std::exp(vgetq_lane_f64(v_X, 1)) - params.K, 0.0);
        }

        // Handle any remaining single path if num_simulations was odd
        // This will reuse the existing scalar price_impl logic (which returns discounted payoff)
        for (uint32_t p = 0; p < remaining_simulations; ++p) {
            double X_t = std::log(params.S0);
            double v_t = params.v0;

            for (uint32_t i = 0; i < params.steps; ++i) {
                const double v_trunc = v_t > 0.0 ? v_t : 0.0;
                const double sqrt_v_trunc = std::sqrt(v_trunc);
                const auto [Z1, Z2] = prng.next_normal_pair();

                const double dW_S = Z1;
                const double dW_v = params.rho * Z1 + rho_complement * Z2;

                X_t += (params.r - 0.5 * v_trunc) * dt + sqrt_v_trunc * sqrt_dt * dW_S;
                v_t += params.kappa * (params.theta - v_trunc) * dt + params.sigma * sqrt_v_trunc * sqrt_dt * dW_v;
            }
            total_undiscounted_payoff += std::max(std::exp(X_t) - params.K, 0.0);
        }

        // Apply final discount factor once
        return (total_undiscounted_payoff / num_simulations) * std::exp(-params.r * params.T);
    }
#endif // __ARM_NEON
};

} // namespace pricer::models
