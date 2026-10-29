#include "autograd.h"
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ml::ad
{
    struct Node
    {
        Tensor value;
        std::optional<Tensor> gradient;
        Node *pending_delete = nullptr;
        bool requires_gradient = false;
        using Edge = std::pair<std::shared_ptr<Node>, std::function<Tensor(const Tensor &)>>;
        std::vector<Edge> parents;

        Node(Tensor v, const bool needs_gradient)
            : value(std::move(v)), requires_gradient(needs_gradient)
        {
        }
    };

    namespace
    {
        thread_local bool recording = true;
        // Deferred deletion also makes destruction of long chains iterative.
        void delete_node(Node *root) noexcept
        {
            thread_local Node *pending = nullptr;
            thread_local bool draining = false;
            root->pending_delete = pending;
            pending = root;
            if (draining)
                return;
            draining = true;
            while (pending)
            {
                Node *current = pending;
                pending = current->pending_delete;
                delete current;
            }
            draining = false;
        }
        std::shared_ptr<Node> make_node(Tensor value, bool needs_gradient)
        {
            return std::shared_ptr<Node>(new Node(std::move(value), needs_gradient), delete_node);
        }

        std::shared_ptr<Node> node(Tensor value, std::vector<Node::Edge> edges)
        {
            auto result = make_node(std::move(value), false);
            if (!recording)
                return result;
            for (auto &edge : edges)
                if (edge.first->requires_gradient)
                {
                    result->parents.push_back(std::move(edge));
                    result->requires_gradient = true;
                }
            return result;
        }
    } // namespace

    Variable::Variable(Tensor value, bool requires_gradient)
        : node_(make_node(std::move(value), requires_gradient))
    {
    }

    Variable::Variable(std::shared_ptr<Node> node) : node_(std::move(node)) {}

    const Tensor &Variable::value() const
    {
        return node_->value;
    }

    const Tensor &Variable::grad() const
    {
        if (!node_->gradient)
            throw std::logic_error("backward has not reached variable");
        return *node_->gradient;
    }

    void Variable::backward()
    {
        if (value().size() != 1)
            throw std::invalid_argument("Non-scalar backward requires seed");
        backward(Tensor(value().rows(), value().cols(), 1, value().backend()));
    }

    void Variable::backward(const Tensor &seed)
    {
        if (!node_->requires_gradient || seed.rows() != value().rows() ||
            seed.cols() != value().cols() || seed.backend() != value().backend() ||
            !seed.all_finite())
            throw std::invalid_argument("Invalid backward seed/root");
        std::vector<std::shared_ptr<Node>> order;
        std::unordered_set<Node *> visited;
        struct Frame
        {
            std::shared_ptr<Node> node;
            std::size_t next = 0;
        };
        std::vector<Frame> stack{{node_}};
        visited.insert(node_.get());
        while (!stack.empty())
        {
            auto &frame = stack.back();
            if (frame.next < frame.node->parents.size())
            {
                auto parent = frame.node->parents[frame.next++].first;
                if (visited.insert(parent.get()).second)
                    stack.push_back({std::move(parent)});
            }
            else
            {
                order.push_back(frame.node);
                stack.pop_back();
            }
        }
        for (const auto &current : order)
            current->gradient.reset();
        node_->gradient = seed;
        for (auto it = order.rbegin(); it != order.rend(); ++it)
            for (const auto &[parent, pullback] : (*it)->parents)
            {
                auto contribution = pullback(*(*it)->gradient);
                if (parent->gradient)
                    parent->gradient = *parent->gradient + contribution;
                else
                    parent->gradient = std::move(contribution);
            }
    }

    bool grad_enabled() noexcept
    {
        return recording;
    }
    NoGradGuard::NoGradGuard() noexcept : previous_(recording)
    {
        recording = false;
    }
    NoGradGuard::~NoGradGuard()
    {
        recording = previous_;
    }

    bool Variable::requires_grad() const
    {
        return node_->requires_gradient;
    }
    bool Variable::has_grad() const
    {
        return node_->gradient.has_value();
    }
    Variable Variable::detach() const
    {
        return Variable(value(), false);
    }

    Variable operation(Tensor value, const std::vector<std::pair<Variable, Pullback>> &parents)
    {
        if (!recording)
            return Variable(std::move(value), false);
        std::vector<Node::Edge> edges;
        for (const auto &[parent, pullback] : parents)
        {
            if (!pullback)
                throw std::invalid_argument("Missing pullback");
            if (parent.value().backend() != value.backend())
                throw std::invalid_argument("Mixed backends in autograd operation");
            edges.emplace_back(parent.node_,
                               [pullback, expected = parent.value()](const Tensor &gradient)
                               {
                                   auto result = pullback(gradient);
                                   if (result.rows() != expected.rows() ||
                                       result.cols() != expected.cols() ||
                                       result.backend() != expected.backend())
                                       throw std::logic_error("Invalid pullback shape/backend");
                                   return result;
                               });
        }
        return Variable(node(std::move(value), std::move(edges)));
    }

    Variable add(const Variable &a, const Variable &b)
    {
        return Variable(
            node(a.value() + b.value(), {{a.node_, [](const Tensor &g) { return g; }},
                                         {b.node_, [](const Tensor &g) { return g; }}}));
    }

    Variable affine(const Variable &a, double scale, const double offset)
    {
        return Variable(node(a.value().affine(scale, offset),
                             {{a.node_, [scale](const Tensor &g) { return g.affine(scale); }}}));
    }

    Variable transpose(const Variable &a)
    {
        return Variable(node(a.value().transpose(),
                             {{a.node_, [](const Tensor &g) { return g.transpose(); }}}));
    }

    Variable matmul(const Variable &a, const Variable &b)
    {
        return Variable(node(
            a.value().matmul(b.value()),
            {{a.node_, [v = b.value()](const Tensor &g) { return g.matmul(v.transpose()); }},
             {b.node_, [v = a.value()](const Tensor &g) { return v.transpose().matmul(g); }}}));
    }

    Variable hadamard(const Variable &a, const Variable &b)
    {
        return Variable(
            node(a.value().hadamard(b.value()),
                 {{a.node_, [v = b.value()](const Tensor &g) { return g.hadamard(v); }},
                  {b.node_, [v = a.value()](const Tensor &g) { return g.hadamard(v); }}}));
    }

    Variable broadcast_columns(const Variable &a, const int count)
    {
        return Variable(node(a.value().broadcast_columns(count),
                             {{a.node_, [](const Tensor &g) { return g.sum_columns(); }}}));
    }

    Variable sum(const Variable &a)
    {
        return Variable(
            node(a.value().sum_columns().sum_rows(),
                 {{a.node_, [rows = a.value().rows(), cols = a.value().cols()](const Tensor &g)
                   { return g.broadcast_rows(rows).broadcast_columns(cols); }}}));
    }

    Variable mean(const Variable &a)
    {
        if (a.value().size() == 0)
            throw std::invalid_argument("Empty mean");
        return affine(sum(a), 1.0 / static_cast<double>(a.value().size()));
    }

    Variable square(const Variable &a)
    {
        return hadamard(a, a);
    }

    Variable activate(const Variable &a, Activation activation)
    {
        auto output = a.value().activate(activation);
        return Variable(
            node(output, {{a.node_, [input = a.value(), output, activation](const Tensor &g)
                           { return input.activation_backward(output, g, activation); }}}));
    }

    Variable loss(const Variable &a, const Tensor &target, const LossKind kind)
    {
        auto result = a.value().loss(target, kind);
        const double factor = 1.0 / result.values.cols();
        return Variable(
            node(result.values.sum_columns().affine(factor),
                 {{a.node_, [gradient = result.gradient, factor](const Tensor &g)
                   {
                       return gradient.affine(factor).hadamard(
                           g.broadcast_rows(gradient.rows()).broadcast_columns(gradient.cols()));
                   }}}));
    }
} // namespace ml::ad
