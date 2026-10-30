#include "nd_backend.h"
#include "nd_kernels.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ml::gpu::resident
{
    namespace
    {
        void check(const cudaError_t status, const char* operation)
        {
            if (status != cudaSuccess)
                throw std::runtime_error(std::string(operation) + ": " +
                    cudaGetErrorString(status));
        }

        __global__ void kernel(const NDOperation op, const NDDescriptor& descriptor, const double* a,
                               const double* b, const double* c, double* out, const std::size_t count)
        {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
                 i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
                out[i] = detail::nd_value(op, descriptor, a, b, c, i);
        }
    } // namespace
    void nd_operation(const NDOperation op, const NDDescriptor& d, const double* a, const double* b,
                      const double* c, double* out, const std::size_t count, const int device)
    {
        if (!count)
            return;
        check(cudaSetDevice(device), "N-D cudaSetDevice");
        auto blocks = static_cast<unsigned>((std::min)((count - 1) / 256 + 1, static_cast<std::size_t>(65535)));
        kernel<<<blocks, 256>>>(op, d, a, b, c, out, count);
        check(cudaGetLastError(), "N-D kernel launch");
    }
} // namespace ml::gpu::resident
