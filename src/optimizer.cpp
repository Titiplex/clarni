#include "optimizer.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ml
{
    Optimizer::Optimizer(OptimizerOptions options) : options_(options)
    {
        if (options.kind != OptimizerKind::SGD && options.kind != OptimizerKind::Momentum &&
            options.kind != OptimizerKind::Adam)
            throw std::invalid_argument("Unknown optimizer");
        for (const double value : {options.momentum, options.beta1, options.beta2})
            if (!std::isfinite(value) || value < 0 || value >= 1)
                throw std::invalid_argument("Invalid optimizer decay");
        if (!std::isfinite(options.epsilon) || options.epsilon <= 0)
            throw std::invalid_argument("Invalid optimizer epsilon");
    }

    Tensor Optimizer::update(const std::size_t key, const Tensor& parameters,
                             const Tensor& gradient, const double rate)
    {
        if (!parameters.size() || !std::isfinite(rate) || rate <= 0 ||
            parameters.rows() != gradient.rows() || parameters.cols() != gradient.cols() ||
            parameters.backend() != gradient.backend() || !parameters.all_finite() ||
            !gradient.all_finite())
            throw std::invalid_argument("Invalid optimizer input");
        if (key >= states_.max_size())
            throw std::length_error("Optimizer parameter key too large");
        if (states_.size() <= key)
            states_.resize(key + 1);
        State next = states_[key];
        if (next.steps && (next.rows != parameters.rows() || next.cols != parameters.cols()))
            throw std::invalid_argument("Parameter shape changed: reset optimizer state");
        if (next.steps == (std::numeric_limits<std::size_t>::max)())
            throw std::overflow_error("Optimizer step overflow");
        next.rows = parameters.rows();
        next.cols = parameters.cols();
        Tensor direction = gradient;
        if (options_.kind != OptimizerKind::SGD)
        {
            if (!next.steps)
                next.first = Tensor(next.rows, next.cols, 0, parameters.backend());
            else
                next.first = next.first.to(parameters.backend());
            if (options_.kind == OptimizerKind::Momentum)
                next.first = next.first.affine(options_.momentum) + gradient;
            else
            {
                if (!next.steps)
                    next.second = Tensor(next.rows, next.cols, 0, parameters.backend());
                else
                    next.second = next.second.to(parameters.backend());
                next.first =
                    next.first.affine(options_.beta1) + gradient.affine(1 - options_.beta1);
                next.second = next.second.affine(options_.beta2) +
                    gradient.unary(Unary::Square).affine(1 - options_.beta2);
            }
            direction = next.first;
        }
        ++next.steps;
        if (options_.kind == OptimizerKind::Adam)
        {
            const auto t = static_cast<double>(next.steps);
            const auto m = next.first.affine(1.0 / (1 - std::pow(options_.beta1, t)));
            const auto v = next.second.affine(1.0 / (1 - std::pow(options_.beta2, t)));
            direction = m.divide(v.unary(Unary::Sqrt).affine(1, options_.epsilon));
        }
        auto result = parameters - direction.affine(rate);
        if (!result.all_finite())
            throw std::overflow_error("Non-finite optimizer result");
        states_[key] = std::move(next);
        return result;
    }

    void Optimizer::reset(const std::size_t key)
    {
        if (key < states_.size())
            states_[key] = State();
    }

    void Optimizer::reset()
    {
        states_.clear();
    }

    void Optimizer::restore_states(const std::vector<State>& states)
    {
        for (const auto& state : states)
            if (state.steps)
            {
                if (state.rows <= 0 || state.cols <= 0)
                    throw std::invalid_argument("Invalid optimizer state shape");
                if (options_.kind != OptimizerKind::SGD &&
                    (state.first.rows() != state.rows || state.first.cols() != state.cols ||
                        !state.first.all_finite()))
                    throw std::invalid_argument("Invalid first moment");
                if (options_.kind == OptimizerKind::Adam &&
                    (state.second.rows() != state.rows || state.second.cols() != state.cols ||
                        !state.second.all_finite()))
                    throw std::invalid_argument("Invalid second moment");
                if (options_.kind == OptimizerKind::Adam && !state.second.all_nonnegative())
                    throw std::invalid_argument("Negative or non-finite second moment");
            }
        states_ = states;
    }
} // namespace ml
