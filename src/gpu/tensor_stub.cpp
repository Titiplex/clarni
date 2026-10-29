#include "tensor_backend.h"
#include <stdexcept>

namespace ml::gpu::resident
{
    namespace
    {
        [[noreturn]] void unavailable() { throw std::runtime_error("Resident CUDA backend not compiled"); }
    }

    double* allocate(std::size_t, int) { unavailable(); }

    void release(double*, int) noexcept
    {
    }

    void fill(double*, std::size_t, double, int) { unavailable(); }
    void upload(double*, const double*, std::size_t, int) { unavailable(); }
    void download(double*, const double*, std::size_t, int) { unavailable(); }

    void operation(Operation, const double*, const double*, const double*, double*, int, int, int, int, int, int,
                   double, double, int) { unavailable(); }

    void loss(const double*, const double*, double*, double*, int, int, int, int) { unavailable(); }
    double sum(const double*, std::size_t, int) { unavailable(); }
    bool all_finite(const double*, std::size_t, int) { unavailable(); }
    Statistics statistics() noexcept { return {}; }

    void reset_statistics() noexcept
    {
    }

    void synchronize(int) { unavailable(); }
    const char* matmul_implementation() noexcept { return "CPU-only build"; }
}
