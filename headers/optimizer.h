#pragma once
#include "tensor.h"
#include <cstddef>
#include <vector>

namespace ml
{
    enum class OptimizerKind { SGD, Momentum, Adam };

    struct OptimizerOptions
    {
        OptimizerKind kind = OptimizerKind::SGD;
        double momentum = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    };

    class Optimizer
    {
    public:
        struct State
        {
            std::size_t steps = 0;
            int rows = 0, cols = 0;
            Tensor first, second;
        };

    private:
        OptimizerOptions options_;
        std::vector<State> states_;

    public:
        explicit Optimizer(OptimizerOptions options = {});
        // Gradient is already averaged over the actual batch size.
        Tensor update(std::size_t key, const Tensor& parameters, const Tensor& gradient, double rate);
        void reset(std::size_t key);
        void reset();
        const OptimizerOptions& options() const { return options_; }
        const std::vector<State>& states() const { return states_; }
        void restore_states(const std::vector<State>& states);
    };
}
