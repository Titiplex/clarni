#include "gpu_backend.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace
{
    void check(const cudaError_t error, const char* operation)
    {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }

    class DeviceBuffer
    {
    public:
        double* data = nullptr;

        explicit DeviceBuffer(const std::size_t count)
        {
            if (count != 0)
                check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(double)),
                      "cudaMalloc");
        }

        ~DeviceBuffer() { if (data != nullptr) cudaFree(data); }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    };

    __global__ void probe_kernel()
    {
    }

    __global__ void matmul_kernel(
        const double* a,
        const double* b,
        double* c,
        const std::size_t count,
        const int inner,
        const int cols
    )
    {
        const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t index = first; index < count; index += stride)
        {
            const std::size_t row = index / cols;
            const std::size_t col = index % cols;
            double sum = 0.0;
            for (int k = 0; k < inner; ++k)
                sum += a[row * inner + k]
                    * b[static_cast<std::size_t>(k) * cols + col];
            c[index] = sum;
        }
    }

    __global__ void elementwise_kernel(
        const double* a,
        const double* b,
        double* c,
        const std::size_t count,
        const ml::gpu::Elementwise op,
        const double scalar
    )
    {
        const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t i = first; i < count; i += stride)
        {
            switch (op)
            {
            case ml::gpu::Elementwise::Add: c[i] = a[i] + b[i];
                break;
            case ml::gpu::Elementwise::Subtract: c[i] = a[i] - b[i];
                break;
            case ml::gpu::Elementwise::Scale: c[i] = a[i] * scalar;
                break;
            }
        }
    }

    unsigned int blocks_for(const std::size_t count)
    {
        return static_cast<unsigned int>((std::min)((count - 1) / 256 + 1,
                                                    std::size_t{65535}));
    }

    void to_device(double* dst, const double* src, const std::size_t count)
    {
        check(cudaMemcpy(dst, src, count * sizeof(double), cudaMemcpyHostToDevice),
              "Copy CPU -> GPU");
    }

    void to_host(double* dst, const double* src, const std::size_t count)
    {
        check(cudaMemcpy(dst, src, count * sizeof(double), cudaMemcpyDeviceToHost),
              "Copy GPU -> CPU");
    }
}

namespace ml::gpu
{
    bool compiled() noexcept { return true; }

    ProbeResult probe()
    {
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess)
        {
            const std::string detail = cudaGetErrorString(status);
            cudaGetLastError();
            return {false, -1, {}, "CUDA detection : " + detail};
        }
        if (count == 0) return {false, -1, {}, "No GPU CUDA visible"};

        std::string failures;
        for (int device = 0; device < count; ++device)
        {
            cudaDeviceProp properties{};
            cudaError_t error = cudaGetDeviceProperties(&properties, device);
            if (error == cudaSuccess) error = cudaSetDevice(device);
            if (error == cudaSuccess)
            {
                probe_kernel<<<1, 1>>>();
                error = cudaGetLastError();
                if (error == cudaSuccess) error = cudaDeviceSynchronize();
            }
            if (error == cudaSuccess)
                return {true, device, properties.name, "CUDA initialises and test's kernel executes"};
            if (!failures.empty()) failures += "; ";
            failures += "GPU " + std::to_string(device) + ": " + cudaGetErrorString(error);
            cudaGetLastError();
        }
        return {false, -1, {}, failures};
    }

    void matmul(
        const double* a,
        const double* b,
        double* c,
        const int rows,
        const int inner,
        const int cols,
        const int device
    )
    {
        if (rows <= 0 || inner <= 0 || cols <= 0)
            throw std::invalid_argument("CUDA dimensions invalid");

        check(cudaSetDevice(device), "cudaSetDevice");
        const auto count_a = static_cast<std::size_t>(rows) * inner;
        const auto count_b = static_cast<std::size_t>(inner) * cols;
        const auto count_c = static_cast<std::size_t>(rows) * cols;
        const DeviceBuffer da(count_a);
        const DeviceBuffer db(count_b);
        const DeviceBuffer dc(count_c);
        to_device(da.data, a, count_a);
        to_device(db.data, b, count_b);
        matmul_kernel<<<blocks_for(count_c), 256>>>(da.data, db.data, dc.data,
                                                    count_c, inner, cols);
        check(cudaGetLastError(), "matmul_kernel launch");
        check(cudaDeviceSynchronize(), "matmul_kernel execution");
        to_host(c, dc.data, count_c);
    }

    void elementwise(
        const double* a,
        const double* b,
        double* c,
        const std::size_t count,
        const Elementwise operation,
        const double scalar,
        const int device
    )
    {
        if (count == 0) return;
        check(cudaSetDevice(device), "cudaSetDevice");
        const bool binary = operation != Elementwise::Scale;
        const DeviceBuffer da(count);
        const DeviceBuffer db(binary ? count : 0);
        const DeviceBuffer dc(count);
        to_device(da.data, a, count);
        if (binary) to_device(db.data, b, count);
        elementwise_kernel<<<blocks_for(count), 256>>>(da.data, db.data, dc.data,
                                                       count, operation, scalar);
        check(cudaGetLastError(), "elementwise_kernel launch ");
        check(cudaDeviceSynchronize(), "elementwise_kernel execution");
        to_host(c, dc.data, count);
    }
}
