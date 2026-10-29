#pragma once
#include "autograd.h"
#include <vector>

// CPU reference N-D frontend over the existing reverse-mode engine.
// Row-major, batch first, feature last for Linear/RNN/attention.
namespace ml::nd
{
    using Shape = std::vector<int>;
    std::size_t numel(const Shape& shape);

    class Array
    {
        Shape shape_;
        ad::Variable data_;

    public:
        Array(Shape shape, ad::Variable data);
        explicit Array(const Shape& shape, double fill = 0, bool requires_grad = false);
        static Array from(Shape shape, const std::vector<double>& values,
                          bool requires_grad = false);

        [[nodiscard]] const Shape& shape() const
        {
            return shape_;
        }

        [[nodiscard]] std::size_t size() const
        {
            return numel(shape_);
        }

        [[nodiscard]] const ad::Variable& variable() const
        {
            return data_;
        }

        [[nodiscard]] std::vector<double> values() const;
        [[nodiscard]] std::vector<double> grad() const;

        [[nodiscard]] bool has_grad() const
        {
            return data_.has_grad();
        }

        void backward() const
        {
            auto root = data_;
            root.backward();
        }

        [[nodiscard]] Array detach() const
        {
            return {shape_, data_.detach()};
        }

        [[nodiscard]] double item() const
        {
            return data_.value().item();
        }
    };

    Array reshape(const Array&, Shape);
    Array permute(const Array&, const std::vector<int>& axes);
    Array slice(const Array&, int axis, int first, int count);
    Array concat(const std::vector<Array>&, int axis);
    Array broadcast_to(const Array&, Shape);
    Array add(const Array&, const Array&); // NumPy-style trailing-axis broadcasting
    Array multiply(const Array&, const Array&);
    Array scale(const Array&, double factor, double offset = 0);
    Array activate(const Array&, Activation);
    Array sum(const Array&);
    Array mean(const Array&);
    Array reduce_sum(const Array&, int axis, bool keepdim = false);
    Array matmul(const Array&, const Array&); // rank 2
    // Batched matrix product with trailing-axis batch broadcasting.
    Array batched_matmul(const Array&, const Array&);
    Array softmax(const Array&, int axis = -1);
    Array mse(const Array& prediction, const Array& target);
    // Logits [..., classes], integer labels [...]; ignores labels == ignore_index.
    Array cross_entropy(const Array&, const std::vector<int>& labels, int ignore_index = -1);
    // Reusable CPU operation hook for layers; VJP arguments/returns are flat row-major.
    using Vector = std::vector<double>;
    using VJP = std::function<Vector(const Vector&)>;
    Array operation(Shape, Vector values, const std::vector<std::pair<Array, VJP>>& parents);
} // namespace ml::nd
