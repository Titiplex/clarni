#include "tensor.h"
#include "tensor_backend.h"
#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>

void measure(const ml::math::Matrix& a, const ml::math::Matrix& b, const ml::Backend backend, const bool resident,
             const double reference)
{
    const auto left = ml::Tensor::from_matrix(a, backend), right = ml::Tensor::from_matrix(b, backend);
    for (int i = 0; i < 3; ++i)left.matmul(right).synchronize();
    ml::gpu::resident::reset_statistics();
    ml::Tensor result;
    constexpr int iterations = 20;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        auto x = resident ? left : ml::Tensor::from_matrix(a, backend);
        auto y = resident ? right : ml::Tensor::from_matrix(b, backend);
        result = x.matmul(y);
        result.synchronize();
    }
    const auto end = std::chrono::steady_clock::now();
    const double checksum = result.sum();
    if (std::abs(checksum - reference) > 1e-8 * (1 + std::abs(reference)))
        throw std::runtime_error(
            "Benchmark parity failed");
    const auto transfers = ml::gpu::resident::statistics();
    std::cout << (backend == ml::Backend::CUDA ? ml::gpu::resident::matmul_implementation() : "CPU") << ", " << (
            resident ? "resident inputs" : "recreate/upload inputs")
        << ": " << std::chrono::duration<double, std::milli>(end - start).count() / iterations << " ms/iteration"
        << ", uploads=" << transfers.uploads << ", uploaded_bytes=" << transfers.uploaded_bytes << '\n';
}

int main()
{
    try
    {
        ml::math::Matrix a(128, 128), b(128, 64);
        for (int i = 0; i < a.get_x(); ++i)for (int j = 0; j < a.get_y(); ++j)a(i, j) = 0.01 * (i % 5 + j % 7);
        for (int i = 0; i < b.get_x(); ++i)for (int j = 0; j < b.get_y(); ++j)b(i, j) = 0.02 * (i % 3 + j % 4);
        const auto reference = ml::Tensor::from_matrix(a).matmul(ml::Tensor::from_matrix(b)).sum();
        measure(a, b, ml::Backend::CPU, true, reference);
        measure(a, b, ml::Backend::CPU, false, reference);
        if (ml::initialize_compute().cuda_usable)
        {
            measure(a, b, ml::Backend::CUDA, true, reference);
            measure(a, b, ml::Backend::CUDA, false, reference);
        }
        else std::cout << "CUDA benchmark skipped: " << ml::compute_device().diagnostic << '\n';
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
