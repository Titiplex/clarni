#pragma once
#include <cstddef>
#include "recurrent.h"

namespace ml::nn
{
    // Gate order [reset,update,new], reset-after formulation:
    // n=tanh(x_n + r*(h@W_hn+b_hn)); h'=(1-z)*n+z*h.
    class GRUCell final : public Module
    {
        int input_, hidden_;

    public:
        Param weight_x, weight_h, bias_x, bias_h;
        GRUCell(int input, int hidden, std::uint32_t seed = 42);
        nd::Array step(const nd::Array&, const nd::Array& hidden) const;

        [[nodiscard]] int input_size() const
        {
            return input_;
        }

        [[nodiscard]] int hidden_size() const
        {
            return hidden_;
        }

        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override;
    };

    class GRU final : public Module
    {
    public:
        GRUCell cell;

        GRU(const int input, const int hidden, const std::uint32_t seed = 42) : cell(input, hidden, seed)
        {
        }

        RNNResult run(const nd::Array&, const std::vector<int>& lengths = {},
                      const std::optional<nd::Array>& initial = std::nullopt,
                      std::size_t detach_every = 0) const;

        nd::Array forward(const nd::Array& a) override
        {
            return run(a).sequence;
        }

        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override
        {
            return detail::prefix_parameters(cell, "cell.");
        }

        void train(const bool e = true) override
        {
            Module::train(e);
            cell.train(e);
        }
    };

    struct LSTMState
    {
        nd::Array hidden, cell;
    };

    struct LSTMResult
    {
        nd::Array sequence, hidden, cell;
    };

    // Gate order [input,forget,candidate,output]; forget bias initialized to 1.
    class LSTMCell final : public Module
    {
        int input_, hidden_;

    public:
        Param weight_x, weight_h, bias;
        LSTMCell(int input, int hidden, std::uint32_t seed = 42);
        LSTMState step(const nd::Array&, const LSTMState&) const;

        [[nodiscard]] int input_size() const
        {
            return input_;
        }

        [[nodiscard]] int hidden_size() const
        {
            return hidden_;
        }

        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override;
    };

    class LSTM final : public Module
    {
    public:
        LSTMCell cell;

        LSTM(const int input, const int hidden, const std::uint32_t seed = 42) : cell(input, hidden, seed)
        {
        }

        LSTMResult run(const nd::Array&, const std::vector<int>& lengths = {},
                       std::optional<LSTMState> initial = std::nullopt,
                       std::size_t detach_every = 0) const;

        nd::Array forward(const nd::Array& a) override
        {
            return run(a).sequence;
        }

        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override
        {
            return detail::prefix_parameters(cell, "cell.");
        }

        void train(const bool e = true) override
        {
            Module::train(e);
            cell.train(e);
        }
    };
} // namespace ml::nn
