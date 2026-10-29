#pragma once
#include "tensor.h"
#include <memory>
#include <functional>
#include <vector>
#include <utility>

namespace ml::ad
{
    struct Node;
    class Variable;
    using Pullback = std::function<Tensor(const Tensor&)>;
    Variable operation(Tensor value, const std::vector<std::pair<Variable, Pullback>>& parents);

    class Variable
    {
        std::shared_ptr<Node> node_;
        explicit Variable(std::shared_ptr<Node> node);
        friend Variable operation(Tensor, const std::vector<std::pair<Variable, Pullback>>&);
        friend Variable add(const Variable&, const Variable&);
        friend Variable matmul(const Variable&, const Variable&);
        friend Variable hadamard(const Variable&, const Variable&);
        friend Variable affine(const Variable&, double, double);
        friend Variable transpose(const Variable&);
        friend Variable broadcast_columns(const Variable&, int);
        friend Variable sum(const Variable&);
        friend Variable activate(const Variable&, Activation);
        friend Variable loss(const Variable&, const Tensor&, LossKind);

    public:
        explicit Variable(Tensor value, bool requires_gradient = true);
        const Tensor& value() const;
        const Tensor& grad() const;
        bool requires_grad() const;
        bool has_grad() const;
        Variable detach() const;
        void backward();
        void backward(const Tensor& seed);
    };

    Variable add(const Variable&, const Variable&);
    Variable matmul(const Variable&, const Variable&);
    Variable hadamard(const Variable&, const Variable&);
    Variable affine(const Variable&, double scale, double offset = 0);
    Variable transpose(const Variable&);
    Variable broadcast_columns(const Variable&, int count);
    Variable sum(const Variable&);
    Variable mean(const Variable&);
    Variable square(const Variable&);
    Variable activate(const Variable&, Activation);
    Variable loss(const Variable&, const Tensor& target, LossKind);
}
