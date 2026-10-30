#include "recurrent.h"
#include <cmath>
#include <stdexcept>

namespace ml::nn
{
    namespace detail
    {
        std::vector<int> sequence_lengths(const nd::Array& a, const int features,
                                          const std::vector<int>& lengths)
        {
            if (a.shape().size() != 3 || a.shape()[2] != features)
                throw std::invalid_argument("Sequence expects [B,T,input_size]");
            auto result = lengths.empty() ? std::vector(a.shape()[0], a.shape()[1]) : lengths;
            if (result.size() != static_cast<std::size_t>(a.shape()[0]))
                throw std::invalid_argument("Length count mismatch");
            for (const int l : result)
                if (l < 0 || l > a.shape()[1])
                    throw std::invalid_argument("Invalid sequence length");
            return result;
        }

        nd::Array timestep(const nd::Array& a, const int time)
        {
            return nd::reshape(nd::slice(a, 1, time, 1), {a.shape()[0], a.shape()[2]});
        }

        nd::Array timestep_mask(int batch, const int time, const std::vector<int>& lengths)
        {
            nd::Vector v(batch);
            for (int b = 0; b < batch; ++b)
                v[b] = time < lengths[b] ? 1 : 0;
            return nd::Array::from({batch, 1}, v);
        }

        nd::Array choose(const nd::Array& mask, const nd::Array& yes, const nd::Array& no)
        {
            return nd::add(nd::multiply(mask, yes), nd::multiply(nd::scale(mask, -1, 1), no));
        }

        std::vector<NamedParameter> prefix_parameters(const Module& module,
                                                      const std::string& prefix)
        {
            auto p = module.named_parameters();
            for (auto& n : p)
                n.name = prefix + n.name;
            return p;
        }
    } // namespace detail
    RNNCell::RNNCell(int input, int hidden, const std::uint32_t seed)
        : input_(input), hidden_(hidden)
    {
        nd::numel({input, hidden});
        std::mt19937 gen(seed);
        weight_x =
            normal_parameter({input, hidden}, 1 / std::sqrt(static_cast<double>(input)), gen);
        weight_h =
            normal_parameter({hidden, hidden}, 1 / std::sqrt(static_cast<double>(hidden)), gen);
        bias = std::make_shared<Parameter>(nd::Array({hidden}));
    }

    nd::Array RNNCell::step(const nd::Array& x, const nd::Array& hidden)
    {
        if (x.shape().size() != 2 || hidden.shape() != nd::Shape({x.shape()[0], hidden_}) ||
            x.shape()[1] != input_)
            throw std::invalid_argument("RNNCell input/state mismatch");
        return nd::activate(nd::add(nd::add(nd::matmul(x, weight_x->value()),
                                            nd::matmul(hidden, weight_h->value())),
                                    bias->value()),
                            Activation::Tanh);
    }

    std::vector<NamedParameter> RNNCell::named_parameters() const
    {
        return {{"weight_x", weight_x}, {"weight_h", weight_h}, {"bias", bias}};
    }

    std::vector<NamedParameter> RNN::named_parameters() const
    {
        return detail::prefix_parameters(cell, "cell.");
    }

    RNNResult RNN::run(const nd::Array& a, const std::vector<int>& lengths,
                       const std::optional<nd::Array>& initial, const std::size_t detach_every)
    {
        if (a.backend() != Backend::CPU)
            throw std::invalid_argument(
                "Recurrent CUDA sequences are not implemented in this CNN-focused release");
        const auto valid = detail::sequence_lengths(a, cell.input_size(), lengths);
        int b = a.shape()[0], h = cell.hidden_size();
        const int t = a.shape()[1];
        auto state = initial.value_or(nd::Array({b, h}));
        if (state.shape() != nd::Shape({b, h}))
            throw std::invalid_argument("Initial hidden shape mismatch");
        std::vector<nd::Array> outputs;
        for (int i = 0; i < t; ++i)
        {
            if (detach_every && i && static_cast<std::size_t>(i) % detach_every == 0)
                state = state.detach();
            auto mask = detail::timestep_mask(b, i, valid);
            state = detail::choose(mask, cell.step(detail::timestep(a, i), state), state);
            outputs.push_back(nd::reshape(nd::multiply(mask, state), {b, 1, h}));
        }
        return {nd::concat(outputs, 1), state};
    }
} // namespace ml::nn
