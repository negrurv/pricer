#pragma once

#include <cstdint>
#include <cmath>
#include <array>

namespace pricer::math {

/**
 * @brief Xoshiro256++ 1.0 PRNG
 * 
 * Highly optimized, cache-friendly random number generator.
 * State fits in 32 bytes (4x uint64_t).
 */
struct Xoshiro256 {
    alignas(32) uint64_t s[4];

    static constexpr double PI = 3.14159265358979323846;

    struct NormalPair {
        double z1;
        double z2;
    };

    /**
     * @brief Seeds the generator using a single 64-bit value via SplitMix64.
     */
    void seed(uint64_t seed_val) {
        auto splitmix64 = [&](uint64_t& x) -> uint64_t {
            uint64_t z = (x += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        };

        s[0] = splitmix64(seed_val);
        s[1] = splitmix64(seed_val);
        s[2] = splitmix64(seed_val);
        s[3] = splitmix64(seed_val);
    }

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    /**
     * @brief Generates the next 64-bit random integer.
     * Aggressively inlined for performance.
     */
    inline uint64_t next() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        return result;
    }

    /**
     * @brief Generates a double in the range [0, 1).
     */
    inline double next_double() {
        return (next() >> 11) * (1.0 / (1ULL << 53));
    }

    /**
     * @brief Generates TWO standard normal random variables simultaneously.
     * Perfect for Heston model Cholesky decomposition. Zero branching.
     */
    inline NormalPair next_normal_pair() {
        double u1 = next_double();
        double u2 = next_double();

        // Minor optimization: u1 is almost never 0.0 with this generator, 
        // but safety first.
        while (u1 == 0.0) u1 = next_double();

        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * PI * u2;

        return { radius * std::cos(theta), radius * std::sin(theta) };
    }
};

} // namespace pricer::math
