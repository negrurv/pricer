#include <gtest/gtest.h>
#include "../include/models/heston.hpp"
#include "../include/math/random.hpp"

using namespace pricer;

TEST(HestonPricerTest, BasicConvergence) {
    models::HestonParams params = {
        100.0,  // S0
        0.04,   // v0
        0.05,   // r
        2.0,    // kappa
        0.04,   // theta
        0.1,    // sigma
        -0.7,   // rho
        1.0,    // T
        100.0,  // K
        365     // steps
    };

    models::HestonCallPricer pricer(params);
    math::Xoshiro256 prng;
    prng.seed(42);

    double sum = 0.0;
    int simulations = 1000;
    for (int i = 0; i < simulations; ++i) {
        sum += pricer.price(prng);
    }
    double average_price = sum / simulations;

    // A very loose check just to ensure it's not zero or exploding
    EXPECT_GT(average_price, 0.0);
    EXPECT_LT(average_price, 100.0);
}
