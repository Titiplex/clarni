#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ml
{
    struct Tensor::Buffer
    {
        std::vector<double> values;
    };

    namespace
    {
        double sigmoid(double x)
        {
            if (x >= 0)return 1.0 / (1.0 + std::exp(-x));
            const double e = std::exp(x);
            return e / (1.0 + e);
        }

        std::size_t checked_size(int rows, int cols)
        {
            if (rows < 0 || cols < 0)throw std::invalid_argument("Tensor: negative shape");
            const auto r = static_cast<std::size_t>(rows), c = static_cast<std::size_t>(cols);
            if (c && r > std::vector<double>().max_size() / c)throw std::length_error("Tensor too large");
            return r * c;
        }
    }

    Tensor::Tensor(int rows, int cols, double value, Backend backend)
        : buffer_(std::make_shared<Buffer>()), rows_(rows), cols_(cols),
          backend_(backend == Backend::Auto ? compute_device().selected : backend)
    {
        if (backend_ != Backend::CPU)throw std::runtime_error("Resident CUDA tensor backend not built yet");
        buffer_->values.assign(checked_size(rows, cols), value);
    }

    const double* Tensor::read() const { return buffer_->values.data(); }
    double* Tensor::write() { return buffer_->values.data(); }

    void Tensor::compatible(const Tensor& other, bool shape) const
    {
        if (backend_ != other.backend_ || (shape && (rows_ != other.rows_ || cols_ != other.cols_)))
            throw std::invalid_argument("Tensor: incompatible shape/backend");
    }

    Tensor Tensor::from_matrix(const math::Matrix& matrix, Backend backend)
    {
        Tensor result(matrix.get_x(), matrix.get_y(), 0, backend);
        for (int i = 0; i < result.rows_; ++i)
            for (int j = 0; j < result.cols_; ++j)
                result.write()[static_cast<std::size_t>(i) * result.cols_ + j] = matrix(i, j);
        return result;
    }

    math::Matrix Tensor::to_matrix() const
    {
        math::Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i)
            for (int j = 0; j < cols_; ++j)
                result(i, j) = read()[static_cast<std::size_t>(i)
                    * cols_ + j];
        return result;
    }

    Tensor Tensor::to(Backend backend) const
    {
        if (backend == Backend::Auto)backend = compute_device().selected;
        if (backend == backend_)return *this;
        return from_matrix(to_matrix(), backend);
    }

    Tensor Tensor::operator+(const Tensor& other) const
    {
        compatible(other);
        Tensor out(rows_, cols_, 0, backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = read()[i] + other.read()[i];
        return out;
    }

    Tensor Tensor::operator-(const Tensor& other) const
    {
        compatible(other);
        Tensor out(rows_, cols_, 0, backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = read()[i] - other.read()[i];
        return out;
    }

    Tensor Tensor::hadamard(const Tensor& other) const
    {
        compatible(other);
        Tensor out(rows_, cols_, 0, backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = read()[i] * other.read()[i];
        return out;
    }

    Tensor Tensor::divide(const Tensor& other) const
    {
        compatible(other);
        Tensor out(rows_, cols_, 0, backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = read()[i] / other.read()[i];
        return out;
    }

    Tensor Tensor::affine(double scale, double offset) const
    {
        Tensor out(rows_, cols_, 0, backend_);
        for (std::size_t i = 0; i < size(); ++i)out.write()[i] = scale * read()[i] + offset;
        return out;
    }

    Tensor Tensor::unary(Unary operation) const
    {
        Tensor out(rows_, cols_, 0, backend_);
        for (std::size_t i = 0; i < size(); ++i)
        {
            const double x = read()[i];
            double y = 0;
            switch (operation)
            {
            case Unary::Exp: y = std::exp(x);
                break;
            case Unary::Log: y = std::log(x);
                break;
            case Unary::Square: y = x * x;
                break;
            case Unary::Sqrt: y = std::sqrt(x);
                break;
            case Unary::Reciprocal: y = 1 / x;
                break;
            }
            out.write()[i] = y;
        }
        return out;
    }

    Tensor Tensor::transpose() const
    {
        Tensor out(cols_, rows_, 0, backend_);
        for (int i = 0; i < rows_; ++i)
            for (int j = 0; j < cols_; ++j)
                out.write()[static_cast<std::size_t>(j) * rows_ +
                    i] = read()[static_cast<std::size_t>(i) * cols_ + j];
        return out;
    }

    Tensor Tensor::matmul(const Tensor& other) const
    {
        compatible(other, false);
        if (cols_ != other.rows_)throw std::invalid_argument("Tensor matmul dimensions");
        Tensor out(rows_, other.cols_, 0, backend_);
        for (int i = 0; i < rows_; ++i)
            for (int j = 0; j < other.cols_; ++j)
            {
                double sum = 0;
                for (int k = 0; k < cols_; ++k)
                    sum += read()[static_cast<std::size_t>(i) * cols_ + k] * other.read()[
                        static_cast<std::size_t>(k) * other.cols_ + j];
                out.write()[static_cast<std::size_t>(i) * other.cols_ + j] = sum;
            }
        return out;
    }

    Tensor Tensor::sum_columns() const
    {
        Tensor out(rows_, 1, 0, backend_);
        for (int i = 0; i < rows_; ++i)
            for (int j = 0; j < cols_; ++j)
                out.write()[i] += read()[static_cast<std::size_t>(
                    i) * cols_ + j];
        return out;
    }

    Tensor Tensor::sum_rows() const
    {
        Tensor out(1, cols_, 0, backend_);
        for (int i = 0; i < rows_; ++i)
            for (int j = 0; j < cols_; ++j)
                out.write()[j] += read()[static_cast<std::size_t>(
                    i) * cols_ + j];
        return out;
    }

    Tensor Tensor::broadcast_columns(int count) const
    {
        if (cols_ != 1 || count < 0)throw std::invalid_argument("Tensor column broadcast");
        Tensor out(rows_, count, 0, backend_);
        for (int i = 0; i < rows_; ++i)
            for (int j = 0; j < count; ++j)
                out.write()[static_cast<std::size_t>(i) * count +
                    j] = read()[i];
        return out;
    }

    Tensor Tensor::broadcast_rows(int count) const
    {
        if (rows_ != 1 || count < 0)throw std::invalid_argument("Tensor row broadcast");
        Tensor out(count, cols_, 0, backend_);
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < cols_; ++j)
                out.write()[static_cast<std::size_t>(i) * cols_ +
                    j] = read()[j];
        return out;
    }

    Tensor Tensor::with_bias_row() const
    {
        if (rows_ == (std::numeric_limits<int>::max)())throw std::length_error("Bias row overflow");
        Tensor out(rows_ + 1, cols_, 1, backend_);
        std::copy(read(), read() + size(), out.write() + cols_);
        return out;
    }

    Tensor Tensor::slice_rows(int first, int count) const
    {
        if (first < 0 || count < 0 || first > rows_ || count > rows_ - first)
            throw std::invalid_argument(
                "Tensor row slice");
        Tensor out(count, cols_, 0, backend_);
        if (out.size())
            std::copy(read() + static_cast<std::size_t>(first) * cols_,
                      read() + static_cast<std::size_t>(first + count) * cols_, out.write());
        return out;
    }

    Tensor Tensor::zero_first_column() const
    {
        Tensor out(rows_, cols_, 0, backend_);
        if (size())std::copy(read(), read() + size(), out.write());
        if (cols_)for (int i = 0; i < rows_; ++i)out.write()[static_cast<std::size_t>(i) * cols_] = 0;
        return out;
    }

    Tensor Tensor::activate(Activation activation) const
    {
        Tensor out(rows_, cols_, 0, backend_);
        if (activation == Activation::Softmax)
        {
            if (rows_ == 0)throw std::invalid_argument("Softmax requires classes");
            for (int j = 0; j < cols_; ++j)
            {
                double m = -std::numeric_limits<double>::infinity(), sum = 0;
                for (int i = 0; i < rows_; ++i)m = std::max(m, read()[static_cast<std::size_t>(i) * cols_ + j]);
                for (int i = 0; i < rows_; ++i)
                {
                    const auto k = static_cast<std::size_t>(i) * cols_ + j;
                    out.write()[k] = std::exp(read()[k] - m);
                    sum += out.read()[k];
                }
                for (int i = 0; i < rows_; ++i)out.write()[static_cast<std::size_t>(i) * cols_ + j] /= sum;
            }
            return out;
        }
        for (std::size_t i = 0; i < size(); ++i)
        {
            const double x = read()[i];
            double y = x;
            switch (activation)
            {
            case Activation::Identity: break;
            case Activation::Sigmoid: y = sigmoid(x);
                break;
            case Activation::Tanh: y = std::tanh(x);
                break;
            case Activation::Relu: y = std::max(0.0, x);
                break;
            case Activation::LeakyRelu: y = x > 0 ? x : 0.01 * x;
                break;
            default: throw std::invalid_argument("Activation");
            }
            out.write()[i] = y;
        }
        return out;
    }

    Tensor Tensor::activation_backward(const Tensor& output, const Tensor& upstream, Activation activation) const
    {
        compatible(output);
        compatible(upstream);
        Tensor out(rows_, cols_, 0, backend_);
        if (activation == Activation::Softmax)
        {
            for (int j = 0; j < cols_; ++j)
            {
                double dot = 0;
                for (int i = 0; i < rows_; ++i)
                {
                    auto k = static_cast<std::size_t>(i) * cols_ + j;
                    dot += output.read()[k] * upstream.read()[k];
                }
                for (int i = 0; i < rows_; ++i)
                {
                    auto k = static_cast<std::size_t>(i) * cols_ + j;
                    out.write()[k] = output.read()[k] * (upstream.read()[k] - dot);
                }
            }
            return out;
        }
        for (std::size_t i = 0; i < size(); ++i)
        {
            const double x = read()[i], a = output.read()[i];
            double d = 1;
            switch (activation)
            {
            case Activation::Identity: break;
            case Activation::Sigmoid: d = a * (1 - a);
                break;
            case Activation::Tanh: d = 1 - a * a;
                break;
            case Activation::Relu: d = x > 0 ? 1 : 0;
                break;
            case Activation::LeakyRelu: d = x > 0 ? 1 : 0.01;
                break;
            default: throw std::invalid_argument("Activation");
            }
            out.write()[i] = upstream.read()[i] * d;
        }
        return out;
    }

    LossResult Tensor::loss(const Tensor& target, LossKind kind) const
    {
        compatible(target);
        if (rows_ == 0 || cols_ == 0)throw std::invalid_argument("Empty loss input");
        Tensor values(1, cols_, 0, backend_), gradient(rows_, cols_, 0, backend_);
        for (int j = 0; j < cols_; ++j)
        {
            double m = 0, normalizer = 0, target_sum = 0;
            if (kind == LossKind::CrossEntropy)
            {
                m = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < rows_; ++i)m = std::max(m, read()[static_cast<std::size_t>(i) * cols_ + j]);
                for (int i = 0; i < rows_; ++i)
                    normalizer += std::exp(
                        read()[static_cast<std::size_t>(i) * cols_ + j] - m);
            }
            for (int i = 0; i < rows_; ++i)
            {
                const auto index = static_cast<std::size_t>(i) * cols_ + j;
                const double x = read()[index], y = target.read()[index];
                if (!std::isfinite(x) || !std::isfinite(y))throw std::invalid_argument("Non-finite loss input");
                if (kind != LossKind::Squared && (y < 0 || y > 1))
                    throw std::invalid_argument(
                        "Invalid probability target");
                switch (kind)
                {
                case LossKind::Squared: values.write()[j] += 0.5 * (x - y) * (x - y);
                    gradient.write()[index] = x - y;
                    break;
                case LossKind::BinaryCrossEntropy: values.write()[j] += std::max(x, 0.0) - x * y + std::log1p(
                        std::exp(-std::abs(x)));
                    gradient.write()[index] = sigmoid(x) - y;
                    break;
                case LossKind::CrossEntropy: values.write()[j] += y * ((m - x) + std::log(normalizer));
                    gradient.write()[index] = std::exp(x - m) / normalizer - y;
                    target_sum += y;
                    break;
                default: throw std::invalid_argument("Unknown loss");
                }
            }
            if (kind == LossKind::CrossEntropy && std::abs(target_sum - 1) > 1e-8)
                throw std::invalid_argument(
                    "Class targets must sum to one");
        }
        return {values, gradient};
    }

    double Tensor::sum() const
    {
        double out = 0;
        for (std::size_t i = 0; i < size(); ++i)out += read()[i];
        return out;
    }

    double Tensor::item() const
    {
        if (size() != 1)throw std::invalid_argument("item requires one element");
        return read()[0];
    }

    bool Tensor::all_finite() const
    {
        for (std::size_t i = 0; i < size(); ++i)if (!std::isfinite(read()[i]))return false;
        return true;
    }

    double LossResult::mean() const { return values.sum() / values.cols(); }
}
