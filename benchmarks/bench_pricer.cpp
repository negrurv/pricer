#include <atomic>
#include <benchmark/benchmark.h>
#include <memory>
#include <thread>
#include <vector>

#include "../include/math/random.hpp"
#include "../include/models/heston.hpp"
#include "../include/system/thread_pool.hpp"

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
const models::HestonParams STANDARD_PARAMS = {100.0, 0.04, 0.05, 2.0,   0.04,
                                              0.1,   -0.7, 1.0,  100.0, 365};

// ============================================================================
// BENCHMARK 1: Single Thread Baseline
// ============================================================================
static void BM_Heston_SingleThread(benchmark::State &state) {
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

  void SetUp(const ::benchmark::State &state) override {
    // This code is run once for each range of the benchmark
    // We create the pool here to ensure it's fresh for each test size
    pool = std::make_unique<system::ThreadPool>();
  }

  void TearDown(const ::benchmark::State &state) override {
    // Pool is automatically destroyed here, calling its destructor
    pool.reset();
  }
};

BENCHMARK_DEFINE_F(ThreadPoolFixture,
                   BM_Heston_ThreadPool)(benchmark::State &state) {
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
BENCHMARK_DEFINE_F(ThreadPoolFixture,
                   BM_Heston_ThreadPool_Vectorized)(benchmark::State &state) {
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
