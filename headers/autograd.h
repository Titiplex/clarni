#pragma once
#include "tensor.h"
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace ml::ad
{
    // Thread-local recording state. eval() remains independent of autograd.
    [[nodiscard]] bool grad_enabled() noexcept;

    class NoGradGuard
    {
        bool previous_;

    public:
        NoGradGuard() noexcept;
        ~NoGradGuard();
        NoGradGuard(const NoGradGuard&) = delete;
        NoGradGuard& operator=(const NoGradGuard&) = delete;
        NoGradGuard(NoGradGuard&&) = delete;
    };

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
        [[nodiscard]] const Tensor& value() const;
        [[nodiscard]] const Tensor& grad() const;
        [[nodiscard]] bool requires_grad() const;
        [[nodiscard]] bool has_grad() const;
        [[nodiscard]] Variable detach() const;
        void backward();
        void backward(const Tensor& seed) const;
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
} // namespace ml::ad
