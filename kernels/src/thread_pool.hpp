#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace forge {

inline unsigned detect_num_threads() {
  if (const char* env = std::getenv("FORGE_NUM_THREADS")) {
    int v = std::atoi(env);
    if (v > 0) {
      return static_cast<unsigned>(v);
    }
  }
  unsigned hc = std::thread::hardware_concurrency();
  return hc == 0 ? 1u : hc;
}

/**
 * Persistent pool. `parallel_for(n, fn)` runs fn(begin,end) over contiguous
 * chunks. The calling thread executes chunk 0; workers do the rest.
 */
class ThreadPool {
 public:
  static ThreadPool& instance() {
    static ThreadPool pool;
    return pool;
  }

  unsigned workers() const { return static_cast<unsigned>(threads_.size()); }

  unsigned total_threads() const { return workers() + 1; /* +caller */ }

  template <typename Fn>
  void parallel_for(size_t n, Fn&& fn) {
    if (n == 0) {
      return;
    }
    unsigned nt = total_threads();
    if (nt <= 1 || n == 1 || threads_.empty()) {
      fn(0, n);
      return;
    }
    nt = std::min(nt, static_cast<unsigned>(n));
    const size_t chunk = (n + nt - 1) / nt;

    std::atomic<int> remaining{static_cast<int>(nt - 1)};
    for (unsigned t = 1; t < nt; ++t) {
      size_t begin = static_cast<size_t>(t) * chunk;
      if (begin >= n) {
        remaining.fetch_sub(1);
        continue;
      }
      size_t end = std::min(n, begin + chunk);
      enqueue([&, begin, end]() {
        fn(begin, end);
        remaining.fetch_sub(1);
      });
    }

    // Caller does first chunk.
    fn(0, std::min(n, chunk));

    while (remaining.load(std::memory_order_acquire) > 0) {
      std::this_thread::yield();
    }
  }

 private:
  ThreadPool() {
    unsigned n = detect_num_threads();
    if (n <= 1) {
      return;
    }
    stop_ = false;
    threads_.reserve(n - 1);
    for (unsigned i = 0; i + 1 < n; ++i) {
      threads_.emplace_back([this]() { loop(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  void enqueue(std::function<void()> job) {
    {
      std::lock_guard<std::mutex> g(mu_);
      jobs_.push_back(std::move(job));
    }
    cv_.notify_one();
  }

  void loop() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]() { return stop_ || !jobs_.empty(); });
        if (stop_ && jobs_.empty()) {
          return;
        }
        job = std::move(jobs_.back());
        jobs_.pop_back();
      }
      job();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::function<void()>> jobs_;
  std::vector<std::thread> threads_;
  bool stop_{true};
};

}  // namespace forge
