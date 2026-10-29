#include "tensor_backend.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#if defined(ML_USE_CUBLAS)
#include <cublas_v2.h>
#endif

namespace ml::gpu::resident
{
    namespace
    {
        std::atomic<std::size_t> allocation_count{0}, upload_count{0}, download_count{0}, upload_bytes{0},
                                 download_bytes{0};

        void check(const cudaError_t error, const char* what)
        {
            if (error != cudaSuccess)throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
        }

        void select(const int device) { check(cudaSetDevice(device), "cudaSetDevice"); }
#if defined(ML_USE_CUBLAS)
        void check_blas(cublasStatus_t status, const char* what)
        {
            if (status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error(
                    std::string(what) + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
        }
        struct BlasHandle
        {
            cublasHandle_t handle = nullptr;
            int device = -1;

            ~BlasHandle()
            {
                if (handle)
                {
                    cudaSetDevice(device);
                    cublasDestroy(handle);
                }
            }

            cublasHandle_t get(int requested)
            {
                if (handle && device != requested)
                {
                    cudaSetDevice(device);
                    check_blas(cublasDestroy(handle), "cublasDestroy");
                    handle = nullptr;
                }
                select(requested);
                if (!handle)
                {
                    check_blas(cublasCreate(&handle), "cublasCreate");
                    device = requested;
                    check_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
                }
                return handle;
            }
        };
#endif
        unsigned blocks(const std::size_t n)
        {
            return static_cast<unsigned>((std::min)((n - 1) / 256 + 1, std::size_t{65535}));
        }

        __device__ double sigmoid(const double x)
        {
            if (x >= 0)return 1 / (1 + exp(-x));
            const double e = exp(x);
            return e / (1 + e);
        }

        __global__ void fill_kernel(double* p, const std::size_t n, const double value)
        {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i +=
                 static_cast<std::size_t>(gridDim.x) * blockDim.x)
                p[i] = value;
        }

        __global__ void operation_kernel(const Operation op, const double* a, const double* b, const double* c,
                                         double* out,
                                         const int rows, const int cols, const int output_rows, const int output_cols,
                                         const int other, const int auxiliary,
                                         const double scale, const double offset)
        {
            const std::size_t count = static_cast<std::size_t>(output_rows) * output_cols;
            for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
                 index += static_cast<std::size_t>(gridDim.x) * blockDim.x)
            {
                double value = 0;
                switch (op)
                {
                case Operation::Add: value = a[index] + b[index];
                    break;
                case Operation::Subtract: value = a[index] - b[index];
                    break;
                case Operation::Multiply: value = a[index] * b[index];
                    break;
                case Operation::Divide: value = a[index] / b[index];
                    break;
                case Operation::Affine: value = scale * a[index] + offset;
                    break;
                case Operation::Unary:
                    {
                        const double x = a[index];
                        switch (auxiliary)
                        {
                        case 0: value = exp(x);
                            break;
                        case 1: value = log(x);
                            break;
                        case 2: value = x * x;
                            break;
                        case 3: value = sqrt(x);
                            break;
                        case 4: value = 1 / x;
                            break;
                        default: ;
                        }
                        break;
                    }
                case Operation::Transpose: value = a[(index % rows) * static_cast<std::size_t>(cols) + index / rows];
                    break;
                case Operation::Matmul:
                    {
                        const std::size_t row = index / output_cols;
                        const std::size_t col = index % output_cols;
                        for (int k = 0; k < cols; ++k)
                            value += a[row * cols + k] * b[static_cast<std::size_t>(k) *
                                output_cols + col];
                        break;
                    }
                case Operation::SumColumns: for (int j = 0; j < cols; ++j)value += a[index * cols + j];
                    break;
                case Operation::SumRows: for (int i = 0; i < rows; ++i)
                        value += a[static_cast<std::size_t>(i) * cols +
                            index];
                    break;
                case Operation::BroadcastColumns: value = a[index / other];
                    break;
                case Operation::BroadcastRows: value = a[index % cols];
                    break;
                case Operation::BiasRow: value = index < static_cast<std::size_t>(cols) ? 1 : a[index - cols];
                    break;
                case Operation::SliceRows: value = a[static_cast<std::size_t>(auxiliary) * cols + index];
                    break;
                case Operation::ZeroFirstColumn: value = index % cols == 0 ? 0 : a[index];
                    break;
                case Operation::Activate:
                    {
                        const double x = a[index];
                        switch (auxiliary)
                        {
                        case 0: value = x;
                            break;
                        case 1: value = sigmoid(x);
                            break;
                        case 2: value = tanh(x);
                            break;
                        case 3: value = x > 0 ? x : 0;
                            break;
                        case 4: value = x > 0 ? x : 0.01 * x;
                            break;
                        default: ;
                        }
                        break;
                    }
                case Operation::ActivationBackward:
                    {
                        const double x = a[index];
                        const double y = b[index];
                        double d = 1;
                        switch (auxiliary)
                        {
                        case 1: d = y * (1 - y);
                            break;
                        case 2: d = 1 - y * y;
                            break;
                        case 3: d = x > 0 ? 1 : 0;
                            break;
                        case 4: d = x > 0 ? 1 : 0.01;
                            break;
                        default: ;
                        }
                        value = c[index] * d;
                        break;
                    }
                }
                out[index] = value;
            }
        }

        // One thread per sample handles the coupled class dimension.
        __global__ void softmax_kernel(const double* z, const double* output, const double* gradient, double* result,
                                       const int rows, const int cols, const bool backward)
        {
            for (std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; col < static_cast<
                     std::size_t>(cols); col += static_cast<std::size_t>(gridDim.x) * blockDim.x)
            {
                double maximum = -INFINITY, total = 0;
                if (backward)
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        const auto i = static_cast<std::size_t>(row) * cols + col;
                        total += output[i] * gradient[i];
                    }
                    for (int row = 0; row < rows; ++row)
                    {
                        const auto i = static_cast<std::size_t>(row) * cols + col;
                        result[i] = output[i] * (gradient[i] - total);
                    }
                }
                else
                {
                    for (int row = 0; row < rows; ++row)
                        maximum = fmax(
                            maximum, z[static_cast<std::size_t>(row) * cols + col]);
                    for (int row = 0; row < rows; ++row)
                    {
                        const auto i = static_cast<std::size_t>(row) * cols + col;
                        result[i] = exp(z[i] - maximum);
                        total += result[i];
                    }
                    for (int row = 0; row < rows; ++row)result[static_cast<std::size_t>(row) * cols + col] /= total;
                }
            }
        }

        __global__ void loss_kernel(const double* z, const double* target, double* values, double* gradient,
                                    const int rows,
                                    const int cols, const int kind)
        {
            for (std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; col < static_cast<
                     std::size_t>(cols); col += static_cast<std::size_t>(gridDim.x) * blockDim.x)
            {
                double maximum = -INFINITY, total = 0, target_sum = 0, value = 0;
                bool valid = true;
                for (int row = 0; row < rows; ++row)
                {
                    const auto i = static_cast<std::size_t>(row) * cols + col;
                    maximum = fmax(maximum, z[i]);
                    target_sum += target[i];
                    valid = valid && isfinite(z[i]) && isfinite(target[i]) && (kind == 0 || (target[i] >= 0 && target[i]
                        <= 1));
                }
                if (kind == 2)
                {
                    valid = valid && fabs(target_sum - 1) <= 1e-8;
                    for (int row = 0; row < rows; ++row)
                        total += exp(
                            z[static_cast<std::size_t>(row) * cols + col] - maximum);
                }
                for (int row = 0; row < rows; ++row)
                {
                    const auto i = static_cast<std::size_t>(row) * cols + col;
                    const double x = z[i];
                    const double y = target[i];
                    if (kind == 0)
                    {
                        value += 0.5 * (x - y) * (x - y);
                        gradient[i] = x - y;
                    }
                    else if (kind == 1)
                    {
                        value += fmax(x, 0) - x * y + log1p(exp(-fabs(x)));
                        gradient[i] = sigmoid(x) - y;
                    }
                    else
                    {
                        value += y * ((maximum - x) + log(total));
                        gradient[i] = exp(x - maximum) / total * target_sum - y;
                    }
                    if (!valid)gradient[i] = NAN;
                }
                values[col] = valid ? value : NAN;
            }
        }

        __global__ void sum_kernel(const double* values, const std::size_t count, double* result)
        {
            __shared__ double partial[256];
            double sum = 0;
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i +=
                 static_cast<std::size_t>(gridDim.x) * blockDim.x)
                sum += values[i];
            partial[threadIdx.x] = sum;
            __syncthreads();
            for (unsigned step = 128; step; step /= 2)
            {
                if (threadIdx.x < step)partial[threadIdx.x] += partial[threadIdx.x + step];
                __syncthreads();
            }
            if (threadIdx.x == 0)atomicAdd(result, partial[0]);
        }

        __global__ void finite_kernel(const double* values, const std::size_t count, int* invalid)
        {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i +=
                 static_cast<std::size_t>(gridDim.x) * blockDim.x)
                if (!isfinite(values[i]))atomicExch(invalid, 1);
        }
    }

    double* allocate(const std::size_t count, const int device)
    {
        if (!count)return nullptr;
        if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(double))
            throw std::length_error(
                "CUDA allocation overflow");
        select(device);
        double* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(double)), "resident cudaMalloc");
        ++allocation_count;
        return pointer;
    }

    void release(double* pointer, const int device) noexcept
    {
        if (pointer)
        {
            cudaSetDevice(device);
            cudaFree(pointer);
        }
    }

    void fill(double* pointer, const std::size_t count, const double value, const int device)
    {
        if (!count)return;
        select(device);
        fill_kernel<<<blocks(count),256>>>(pointer, count, value);
        check(cudaGetLastError(), "resident fill");
    }

    void upload(double* destination, const double* source, const std::size_t count, const int device)
    {
        if (!count)return;
        select(device);
        check(cudaMemcpy(destination, source, count * sizeof(double), cudaMemcpyHostToDevice), "resident upload");
        ++upload_count;
        upload_bytes += count * sizeof(double);
    }

    void download(double* destination, const double* source, const std::size_t count, const int device)
    {
        if (!count)return;
        select(device);
        check(cudaMemcpy(destination, source, count * sizeof(double), cudaMemcpyDeviceToHost), "resident download");
        ++download_count;
        download_bytes += count * sizeof(double);
    }

    void operation(const Operation op, const double* a, const double* b, const double* c, double* out, int rows,
                   int cols,
                   const int output_rows, int output_cols, const int other, const int auxiliary, const double scale,
                   const double offset, int device)
    {
        const auto count = static_cast<std::size_t>(output_rows) * output_cols;
        if (!count)return;
        select(device);
#if defined(ML_USE_CUBLAS)
        if (op == Operation::Matmul)
        {
            if (cols == 0)return; // Output has already been initialized to zero.
            thread_local BlasHandle context;
            const double alpha = 1, beta = 0;
            // Row-major C=A*B is column-major C^T=B^T*A^T: operands reversed.
            check_blas(cublasDgemm(context.get(device), CUBLAS_OP_N, CUBLAS_OP_N,
                                   output_cols, rows, cols, &alpha, b, output_cols, a, cols, &beta, out, output_cols),
                       "cublasDgemm");
            return;
        }
#endif
        if ((op == Operation::Activate || op == Operation::ActivationBackward) && auxiliary == 5)
            softmax_kernel<<<
                blocks(cols),256>>>(a, b, c, out, rows, cols, op == Operation::ActivationBackward);
        else
            operation_kernel<<<blocks(count),256>>>(op, a, b, c, out, rows, cols, output_rows, output_cols, other,
                                                    auxiliary, scale, offset);
        check(cudaGetLastError(), "resident operation");
    }

    void loss(const double* logits, const double* target, double* values, double* gradient, const int rows,
              const int cols,
              const int kind, const int device)
    {
        select(device);
        loss_kernel<<<blocks(cols),256>>>(logits, target, values, gradient, rows, cols, kind);
        check(cudaGetLastError(), "resident loss");
    }

    double sum(const double* pointer, const std::size_t count, const int device)
    {
        if (!count)return 0;
        select(device);
        double* result = allocate(1, device);
        double value = 0;
        try
        {
            check(cudaMemset(result, 0, sizeof(double)), "resident sum initialize");
            sum_kernel<<<blocks(count),256>>>(pointer, count, result);
            check(cudaGetLastError(), "resident sum");
            download(&value, result, 1, device);
        }
        catch (...)
        {
            release(result, device);
            throw;
        }
        release(result, device);
        return value;
    }

    bool all_finite(const double* pointer, const std::size_t count, const int device)
    {
        if (!count)return true;
        select(device);
        int* invalid = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&invalid), sizeof(int)), "finite flag");
        int result = 0;
        try
        {
            check(cudaMemset(invalid, 0, sizeof(int)), "finite initialize");
            finite_kernel<<<blocks(count),256>>>(pointer, count, invalid);
            check(cudaGetLastError(), "finite check");
            check(cudaMemcpy(&result, invalid, sizeof(int), cudaMemcpyDeviceToHost), "finite download");
            ++download_count;
            download_bytes += sizeof(int);
        }
        catch (...)
        {
            cudaFree(invalid);
            throw;
        }
        cudaFree(invalid);
        return result == 0;
    }

    Statistics statistics() noexcept
    {
        return {
            allocation_count.load(), upload_count.load(), download_count.load(), upload_bytes.load(),
            download_bytes.load()
        };
    }

    void reset_statistics() noexcept
    {
        allocation_count = 0;
        upload_count = 0;
        download_count = 0;
        upload_bytes = 0;
        download_bytes = 0;
    }

    void synchronize(const int device)
    {
        select(device);
        check(cudaDeviceSynchronize(), "resident synchronize");
    }

    const char* matmul_implementation() noexcept
    {
#if defined(ML_USE_CUBLAS)
        return "cuBLAS Dgemm";
#else
        return "reference CUDA kernel";
#endif
    }
}
