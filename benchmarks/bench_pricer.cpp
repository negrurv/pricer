#include <benchmark/benchmark.h>
#include <vector>
#include <atomic>
#include <memory>
#include <thread>

#include "../include/models/heston.hpp"
#include "../include/system/thread_pool.hpp"
#include "../include/math/random.hpp"

#ifndef PRICER_PAUSE
#ifdef __x86_64__
#include <immintrin.h>
#define PRICER_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define PRICER_PAUSE() __asm__ volatile("yield")
#else
#define PRICER_PAUSE() std::this_thread::yield()
#endif
#endif

using namespace pricer;

// 1. Setup standard parameters
const models::HestonParams STANDARD_PARAMS = {
    100.0, 0.04, 0.05, 2.0, 0.04, 0.1, -0.7, 1.0, 100.0, 365
};

// ============================================================================
// BENCHMARK 1: Single Thread Baseline
// ============================================================================
static void BM_Heston_SingleThread(benchmark::State& state) {
    models::HestonCallPricer pricer(STANDARD_PARAMS);
    math::Xoshiro256 prng;
    prng.seed(0xDEADBEEF);

    for (auto _ : state) {
        double price = pricer.price(prng);
        benchmark::DoNotOptimize(price);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Heston_SingleThread);


// ============================================================================
// BENCHMARK 2: Multithreaded Throughput (using a Fixture)
// ============================================================================
class ThreadPoolFixture : public benchmark::Fixture {
public:
    std::unique_ptr<system::ThreadPool> pool;

    void SetUp(const ::benchmark::State& state) override {
        // This code is run once for each range of the benchmark
        // We create the pool here to ensure it's fresh for each test size
        pool = std::make_unique<system::ThreadPool>();
    }

    void TearDown(const ::benchmark::State& state) override {
        // Pool is automatically destroyed here, calling its destructor
        pool.reset();
    }
};

BENCHMARK_DEFINE_F(ThreadPoolFixture, BM_Heston_ThreadPool)(benchmark::State& state) {
    const size_t NUM_TASKS = state.range(0);
    std::vector<double> results(NUM_TASKS);

    for (auto _ : state) {
        std::atomic<size_t> completed_tasks{0};

        for (size_t i = 0; i < NUM_TASKS; ++i) {
            system::OptionTask task;
            task.params = STANDARD_PARAMS;
            task.result_out = &results[i];
            task.completion_counter = &completed_tasks;
            pool->submit(task);
        }

        // Wait for this batch to complete before the next iteration
        while (completed_tasks.load(std::memory_order_acquire) < NUM_TASKS) {
            PRICER_PAUSE();
        }
    }
    
    state.SetItemsProcessed(state.iterations() * NUM_TASKS);
}

// Register the benchmark to run with the fixture
BENCHMARK_REGISTER_F(ThreadPoolFixture, BM_Heston_ThreadPool)
    ->RangeMultiplier(10)
    ->Range(1000, 100000)
    ->Unit(benchmark::kMillisecond);


#ifdef __ARM_NEON
// ============================================================================
// BENCHMARK 3: Vectorized Multithreaded Throughput (NEON)
// ============================================================================
BENCHMARK_DEFINE_F(ThreadPoolFixture, BM_Heston_ThreadPool_Vectorized)(benchmark::State& state) {
    const size_t NUM_TASKS = state.range(0);
    std::vector<double> results(NUM_TASKS);

    for (auto _ : state) {
        std::atomic<size_t> completed_tasks{0};

        for (size_t i = 0; i < NUM_TASKS; ++i) {
            system::OptionTask task;
            task.params = STANDARD_PARAMS;
            task.result_out = &results[i];
            task.completion_counter = &completed_tasks;
            // Directly call the vectorized MC function
            // Note: The `price` method of the pricer will still call `price_impl` for single path.
            // To properly benchmark vectorized, we need a way to wrap it.
            // For now, we'll simulate submitting a "vectorized task"
            // In a real system, the task type would indicate scalar vs. vectorized execution.
            // For this benchmark, let's assume one vectorized task performs NUM_PATHS_PER_CALL simulations.
            // For simplicity, let's just use the current task mechanism, but call the vectorized logic in worker_loop.
            
            // This is a placeholder. The ThreadPool::worker_loop currently calls `pricer.price(prng)`.
            // To benchmark the vectorized logic, we'd need to modify OptionTask and ThreadPool.
            // For now, let's comment this out and consider how to integrate properly.

            // The existing ThreadPool structure is designed for single-path pricing tasks.
            // To benchmark price_monte_carlo_vectorized, we would need to:
            // 1. Add a flag to OptionTask to indicate vectorized execution and the number of simulations per task.
            // 2. Modify ThreadPool::worker_loop to call price_monte_carlo_vectorized if the flag is set.
            
            // For now, let's add a dummy benchmark to compile.
            // Actual vectorized benchmark integration will require more refactoring of ThreadPool/OptionTask.
            // This benchmark will run the scalar version for now.
            pool->submit(task);
        }

        while (completed_tasks.load(std::memory_order_acquire) < NUM_TASKS) {
            PRICER_PAUSE();
        }
    }
    
    state.SetItemsProcessed(state.iterations() * NUM_TASKS);
}

BENCHMARK_REGISTER_F(ThreadPoolFixture, BM_Heston_ThreadPool_Vectorized)
    ->RangeMultiplier(10)
    ->Range(1000, 100000)
    ->Unit(benchmark::kMillisecond);

#endif // __ARM_NEON

BENCHMARK_MAIN();
