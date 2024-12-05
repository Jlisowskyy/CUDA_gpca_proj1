//
// Created by Jlisowskyy on 03/12/24.
//

#ifndef SRC_THREADPOOL_CUH
#define SRC_THREADPOOL_CUH

#include <thread>
#include <vector>

class ThreadPool {
public:
    // ------------------------------
    // Internal types
    // ------------------------------

    static constexpr __uint32_t INVALID_THREAD_NUM = 0;

    // ------------------------------
    // Class creation
    // ------------------------------

    explicit ThreadPool(const __uint32_t numThreads) : m_numThreadsToSpawn(numThreads) {
        assert(m_numThreadsToSpawn != INVALID_THREAD_NUM && "ThreadPool: numThreads cannot be 0");
    }

    ~ThreadPool() {
        for (std::thread *pThread: m_threads) {
            if (pThread) {
                std::abort();
            }
        }
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    template<class FuncT, class... Args>
    void RunThreads(FuncT&& func, Args&&... args) {
        assert(m_numThreadsToSpawn != INVALID_THREAD_NUM && "Detected second run usage on the thread pool");

        m_threads.reserve(m_numThreadsToSpawn);
        for (__uint32_t idx = 0; idx < m_numThreadsToSpawn; ++idx) {
            m_threads.push_back(new std::thread(func, idx, std::forward<Args>(args)...));
        }

        m_numThreadsToSpawn = INVALID_THREAD_NUM;
    }

    void Wait() {
        assert(m_numThreadsToSpawn == INVALID_THREAD_NUM && "Detected early wait on thread pool");

        while (!m_threads.empty()) {
            std::thread *pThread = m_threads.back();
            m_threads.pop_back();

            pThread->join();
            delete pThread;
        }
    }

    void Reset(const __uint32_t numThreads) {
        m_numThreadsToSpawn = numThreads;
    }

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:

    // ------------------------------
    // Class fields
    // ------------------------------

    __uint32_t m_numThreadsToSpawn{};
    std::vector<std::thread *> m_threads{};
};

#endif //SRC_THREADPOOL_CUH
