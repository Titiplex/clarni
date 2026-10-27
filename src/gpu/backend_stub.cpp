#include "gpu_backend.h"
#include <stdexcept>

namespace ml::gpu
{
    bool compiled() noexcept { return false; }

    ProbeResult probe()
    {
        return {false, -1, {}, "Binary compiled without CUDA"};
    }

    void matmul(const double*, const double*, double*, int, int, int, int)
    {
        throw std::logic_error("CUDA backend not compiled");
    }

    void elementwise(const double*, const double*, double*, std::size_t,
                     Elementwise, double, int)
    {
        throw std::logic_error("CUDA backend not compiled");
    }
}
