#pragma once
#include "compute_device.h"
#include "matrix.h"
#include "nd_backend.h"
#include "tensor_backend.h"
#include <cstddef>
#include <memory>
#include <span>

namespace ml
{
    enum class Activation
    {
        Identity,
        Sigmoid,
        Tanh,
        Relu,
        LeakyRelu,
        Softmax
    };

    enum class LossKind
    {
        Squared,
        BinaryCrossEntropy,
        CrossEntropy
    };

    enum class Unary
    {
        Exp,
        Log,
        Square,
        Sqrt,
        Reciprocal
    };

    struct LossResult;

    // Immutable value semantics; copies share storage, operations allocate outputs.
    // Samples occupy columns. This first Tensor supports rank two, not arbitrary ND.
    class Tensor
    {
        struct Buffer;
        std::shared_ptr<Buffer> buffer_;
        int rows_, cols_;
        Backend backend_;
        [[nodiscard]] const double *read() const;
        [[nodiscard]] double *write() const;
        void compatible(const Tensor &other, bool shape = true) const;
        Tensor device_operation(gpu::resident::Operation operation, int output_rows,
                                int output_cols, const Tensor *b = nullptr,
                                const Tensor *c = nullptr, int other = 0, int auxiliary = 0,
                                double scale = 0, double offset = 0) const;

      public:
        explicit Tensor(int rows = 0, int cols = 0, double value = 0.0,
                        Backend backend = Backend::CPU);
        // Immutable CPU view; valid while this tensor's shared storage is alive.
        [[nodiscard]] std::span<const double> cpu_values() const &;
        std::span<const double> cpu_values() const && = delete;
        // Adopts CPU storage; CUDA remains an explicit upload.
        static Tensor from_values(int rows, int cols, std::vector<double> values,
                                  Backend backend = Backend::CPU);
        static Tensor from_matrix(const math::Matrix &matrix, Backend backend = Backend::CPU);
        [[nodiscard]] math::Matrix to_matrix() const;
        [[nodiscard]] Tensor to(Backend backend) const;
        [[nodiscard]] int rows() const
        {
            return rows_;
        }
        [[nodiscard]] int cols() const
        {
            return cols_;
        }
        [[nodiscard]] std::size_t size() const
        {
            return static_cast<std::size_t>(rows_) * cols_;
        }
        [[nodiscard]] Backend backend() const
        {
            return backend_;
        }
        Tensor operator+(const Tensor &other) const;
        Tensor operator-(const Tensor &other) const;
        [[nodiscard]] Tensor hadamard(const Tensor &other) const;
        [[nodiscard]] Tensor divide(const Tensor &other) const;
        [[nodiscard]] Tensor affine(double scale, double offset = 0.0) const;
        [[nodiscard]] Tensor unary(Unary operation) const;
        // Internal bridge: callers supply validated N-D descriptors.
        [[nodiscard]] Tensor nd_operation(gpu::resident::NDOperation, int output_rows,
                                          int output_cols, const gpu::resident::NDDescriptor &,
                                          const Tensor *b = nullptr,
                                          const Tensor *c = nullptr) const;
        [[nodiscard]] double l2_norm() const;
        [[nodiscard]] Tensor reshape(int rows, int cols) const;
        [[nodiscard]] Tensor transpose() const;
        [[nodiscard]] Tensor matmul(const Tensor &other) const;
        [[nodiscard]] Tensor sum_columns() const;
        [[nodiscard]] Tensor sum_rows() const;
        [[nodiscard]] Tensor broadcast_columns(int count) const;
        [[nodiscard]] Tensor broadcast_rows(int count) const;
        [[nodiscard]] Tensor with_bias_row() const;
        [[nodiscard]] Tensor slice_rows(int first, int count) const;
        [[nodiscard]] Tensor zero_first_column() const;
        [[nodiscard]] Tensor activate(Activation activation) const;
        [[nodiscard]] Tensor activation_backward(const Tensor &output, const Tensor &upstream,
                                                 Activation activation) const;
        [[nodiscard]] LossResult loss(const Tensor &target, LossKind kind) const;
        [[nodiscard]] double sum() const;
        [[nodiscard]] double item() const;
        [[nodiscard]] bool all_finite() const;
        void synchronize() const;
    };

    struct LossResult
    {
        Tensor values;   // 1 x B, sum over outputs for each sample.
        Tensor gradient; // Unaveraged per-sample gradient, with respect to input/logits.
        [[nodiscard]] double mean() const;
    };
} // namespace ml
