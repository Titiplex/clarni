#pragma once
#include "module.h"
#include <optional>

namespace ml::nn
{
    class RNNCell final : public Module
    {
        int input_, hidden_;

    public:
        Param weight_x, weight_h, bias;
        RNNCell(int input, int hidden, std::uint32_t seed = 42);
        nd::Array step(const nd::Array& x, const nd::Array& hidden);

        [[nodiscard]] int hidden_size() const
        {
            return hidden_;
        }

        [[nodiscard]] int input_size() const
        {
            return input_;
        }

        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override;
    };

    struct RNNResult
    {
        nd::Array sequence, hidden;
    };

    class RNN final : public Module
    {
    public:
        RNNCell cell;

        RNN(const int input, const int hidden, const std::uint32_t seed = 42) : cell(input, hidden, seed)
        {
        }

        // [B,T,D] -> [B,T,H]. Padding outputs zero; state freezes after lengths[b].
        RNNResult run(const nd::Array&, const std::vector<int>& lengths = {},
                      const std::optional<nd::Array>& initial = std::nullopt,
                      std::size_t detach_every = 0);

        nd::Array forward(const nd::Array& a) override
        {
            return run(a).sequence;
        }

        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override;

        void train(const bool e = true) override
        {
            Module::train(e);
            cell.train(e);
        }
    };

    // Shared sequence helpers also used by GRU/LSTM. Lengths can contain zero.
    namespace detail
    {
        std::vector<int> sequence_lengths(const nd::Array&, int features,
                                          const std::vector<int>&);
        nd::Array timestep(const nd::Array&, int time);
        nd::Array timestep_mask(int batch, int time, const std::vector<int>& lengths);
        nd::Array choose(const nd::Array& mask, const nd::Array& yes, const nd::Array& no);
        std::vector<NamedParameter> prefix_parameters(const Module&, const std::string& prefix);
    } // namespace detail
} // namespace ml::nn
