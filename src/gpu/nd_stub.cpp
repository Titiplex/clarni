#include "nd_backend.h"
#include <stdexcept>

namespace ml::gpu::resident
{
    void nd_operation(NDOperation, const NDDescriptor&, const double*, const double*,
                      const double*, double*, std::size_t, int)
    {
        throw std::runtime_error("N-D CUDA kernels are not compiled");
    }
} // namespace ml::gpu::resident
