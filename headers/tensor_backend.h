#pragma once
#include <cstddef>

namespace ml::gpu::resident
{
    enum class Operation
    {
        Add, Subtract, Multiply, Divide, Affine, Unary, Transpose, Matmul,
        SumColumns, SumRows, BroadcastColumns, BroadcastRows, BiasRow, SliceRows, ZeroFirstColumn,
        Activate, ActivationBackward
    };

    struct Statistics
    {
        std::size_t allocations = 0, uploads = 0, downloads = 0, uploaded_bytes = 0, downloaded_bytes = 0;
    };

    double* allocate(std::size_t count, int device);
    void release(double* pointer, int device) noexcept;
    void fill(double* pointer, std::size_t count, double value, int device);
    void upload(double* destination, const double* source, std::size_t count, int device);
    void download(double* destination, const double* source, std::size_t count, int device);
    void operation(Operation op, const double* a, const double* b, const double* c, double* out,
                   int rows, int cols, int output_rows, int output_cols, int other, int auxiliary,
                   double scale, double offset, int device);
    void loss(const double* logits, const double* target, double* values, double* gradient,
              int rows, int cols, int kind, int device);
    double sum(const double* pointer, std::size_t count, int device);
    bool all_finite(const double* pointer, std::size_t count, int device);
    Statistics statistics() noexcept;
    void reset_statistics() noexcept;
    void synchronize(int device);
    const char* matmul_implementation() noexcept;
}
