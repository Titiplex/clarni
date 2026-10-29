#pragma once
#include "matrix.h"
#include "compute_device.h"
#include <memory>

namespace ml
{
    enum class Activation { Identity, Sigmoid, Tanh, Relu, LeakyRelu, Softmax };

    enum class LossKind { Squared, BinaryCrossEntropy, CrossEntropy };

    enum class Unary { Exp, Log, Square, Sqrt, Reciprocal };

    struct LossResult;

    // Immutable value semantics; copies share storage, operations allocate outputs.
    // Samples occupy columns. This first Tensor supports rank two, not arbitrary ND.
    class Tensor
    {
        struct Buffer;
        std::shared_ptr<Buffer> buffer_;
        int rows_, cols_;
        Backend backend_;
        const double* read() const;
        double* write();
        void compatible(const Tensor& other, bool shape = true) const;

    public:
        Tensor(int rows = 0, int cols = 0, double value = 0.0, Backend backend = Backend::CPU);
        static Tensor from_matrix(const math::Matrix& matrix, Backend backend = Backend::CPU);
        math::Matrix to_matrix() const;
        Tensor to(Backend backend) const;
        int rows() const { return rows_; }
        int cols() const { return cols_; }
        std::size_t size() const { return static_cast<std::size_t>(rows_) * cols_; }
        Backend backend() const { return backend_; }
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor hadamard(const Tensor& other) const;
        Tensor divide(const Tensor& other) const;
        Tensor affine(double scale, double offset = 0.0) const;
        Tensor unary(Unary operation) const;
        Tensor transpose() const;
        Tensor matmul(const Tensor& other) const;
        Tensor sum_columns() const;
        Tensor sum_rows() const;
        Tensor broadcast_columns(int count) const;
        Tensor broadcast_rows(int count) const;
        Tensor with_bias_row() const;
        Tensor slice_rows(int first, int count) const;
        Tensor zero_first_column() const;
        Tensor activate(Activation activation) const;
        Tensor activation_backward(const Tensor& output, const Tensor& upstream, Activation activation) const;
        LossResult loss(const Tensor& target, LossKind kind) const;
        double sum() const;
        double item() const;
        bool all_finite() const;
    };

    struct LossResult
    {
        Tensor values; // 1 x B, sum over outputs for each sample.
        Tensor gradient; // Unaveraged per-sample gradient, with respect to input/logits.
        double mean() const;
    };
}
