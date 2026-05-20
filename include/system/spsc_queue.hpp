#pragma once

#include <atomic>
#include <cstddef>
#include <vector>
#include <memory>

namespace pricer::system {

/**
 * @brief Lock-free Single-Producer Single-Consumer (SPSC) Ring Buffer.
 */
template <typename T, size_t Size>
class SPSCQueue {
    static_assert((Size != 0) && ((Size & (Size - 1)) == 0), "Queue size must be a power of 2");
    static constexpr size_t MASK = Size - 1;

public:
    SPSCQueue() : head_(0), tail_(0), buffer_(Size) {}

    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    bool push(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head - current_tail == Size) {
            return false;
        }

        buffer_[current_head & MASK] = item;
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false;
        }

        item = buffer_[current_tail & MASK];
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

private:
    // Using atomic with padding to avoid false sharing, but keeping it simple
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    
    // Use vector for heap allocation to avoid stack overflow or alignment issues
    std::vector<T> buffer_;
};

} // namespace pricer::system
