#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include <memory>

#ifdef __x86_64__
#include <immintrin.h>
#define PRICER_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define PRICER_PAUSE() __asm__ volatile("yield")
#else
#define PRICER_PAUSE() std::this_thread::yield()
#endif

#include "spsc_queue.hpp"
#include "../models/heston.hpp"

namespace pricer::system {

// Add alignas(64) back to ensure the struct fits perfectly in a cache line
struct alignas(64) OptionTask {
    models::HestonParams params;
    double* result_out;
    std::atomic<size_t>* completion_counter = nullptr;
    uint64_t path_seed; // <-- CRN Seed Injection
};

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) 
        : running_(true), next_worker_(0) {
        
        size_t actual_threads = (num_threads > 0) ? num_threads : 1;

        for (size_t i = 0; i < actual_threads; ++i) {
            queues_.emplace_back(std::make_unique<SPSCQueue<OptionTask, 2048>>());
        }
        
        for (size_t i = 0; i < actual_threads; ++i) {
            threads_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~ThreadPool() {
        running_.store(false, std::memory_order_release);
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    inline void submit(const OptionTask& task) {
        size_t worker_idx = next_worker_.fetch_add(1, std::memory_order_relaxed) % queues_.size();

        while (!queues_[worker_idx]->push(task)) {
            PRICER_PAUSE(); 
        }
    }

private:
    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<SPSCQueue<OptionTask, 2048>>> queues_;
    std::atomic<bool> running_;
    std::atomic<size_t> next_worker_;

    void worker_loop(size_t queue_index) {
        OptionTask task;

        while (true) {
            if (queues_[queue_index]->pop(task)) {
                
                // Deterministic re-seed for this specific task
                math::Xoshiro256 prng; 
                prng.seed(task.path_seed);

                models::HestonCallPricer pricer(task.params);
                *(task.result_out) = pricer.price(prng);
                if (task.completion_counter) {
                    task.completion_counter->fetch_add(1, std::memory_order_release);
                }
            } else if (!running_.load(std::memory_order_acquire)) {
                // Final drain
                while (queues_[queue_index]->pop(task)) {
                    math::Xoshiro256 prng; 
                    prng.seed(task.path_seed);

                    models::HestonCallPricer pricer(task.params);
                    *(task.result_out) = pricer.price(prng);
                    if (task.completion_counter) {
                        task.completion_counter->fetch_add(1, std::memory_order_release);
                    }
                }
                break;
            } else {
                PRICER_PAUSE(); 
            }
        }
    }
};

} // namespace pricer::system
