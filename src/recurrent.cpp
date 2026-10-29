#include "recurrent.h"
#include <cmath>
#include <stdexcept>
namespace ml::nn
{
    namespace detail
    {
        std::vector<int> sequence_lengths(const nd::Array &a, int features,
                                          const std::vector<int> &lengths)
        {
            if (a.shape().size() != 3 || a.shape()[2] != features)
                throw std::invalid_argument("Sequence expects [B,T,input_size]");
            auto result = lengths.empty() ? std::vector<int>(a.shape()[0], a.shape()[1]) : lengths;
            if (result.size() != std::size_t(a.shape()[0]))
                throw std::invalid_argument("Length count mismatch");
            for (int l : result)
                if (l < 0 || l > a.shape()[1])
                    throw std::invalid_argument("Invalid sequence length");
            return result;
        }
        nd::Array timestep(const nd::Array &a, int t)
        {
            return nd::reshape(nd::slice(a, 1, t, 1), {a.shape()[0], a.shape()[2]});
        }
        nd::Array timestep_mask(int batch, int t, const std::vector<int> &lengths)
        {
            nd::Vector v(batch);
            for (int b = 0; b < batch; ++b)
                v[b] = t < lengths[b] ? 1 : 0;
            return nd::Array::from({batch, 1}, v);
        }
        nd::Array choose(const nd::Array &mask, const nd::Array &yes, const nd::Array &no)
        {
            return nd::add(nd::multiply(mask, yes), nd::multiply(nd::scale(mask, -1, 1), no));
        }
        std::vector<NamedParameter> prefix_parameters(const Module &module,
                                                      const std::string &prefix)
        {
            auto p = module.named_parameters();
            for (auto &n : p)
                n.name = prefix + n.name;
            return p;
        }
    } // namespace detail
    RNNCell::RNNCell(int input, int hidden, std::uint32_t seed) : input_(input), hidden_(hidden)
    {
        nd::numel({input, hidden});
        std::mt19937 gen(seed);
        weight_x = normal_parameter({input, hidden}, 1 / std::sqrt(double(input)), gen);
        weight_h = normal_parameter({hidden, hidden}, 1 / std::sqrt(double(hidden)), gen);
        bias = std::make_shared<Parameter>(nd::Array({hidden}));
    }
    nd::Array RNNCell::step(const nd::Array &x, const nd::Array &h)
    {
        if (x.shape().size() != 2 || h.shape() != nd::Shape({x.shape()[0], hidden_}) ||
            x.shape()[1] != input_)
            throw std::invalid_argument("RNNCell input/state mismatch");
        return nd::activate(
            nd::add(nd::add(nd::matmul(x, weight_x->value()), nd::matmul(h, weight_h->value())),
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
    RNNResult RNN::run(const nd::Array &a, const std::vector<int> &lengths,
                       std::optional<nd::Array> initial, std::size_t detach_every)
    {
        auto valid = detail::sequence_lengths(a, cell.input_size(), lengths);
        int b = a.shape()[0], t = a.shape()[1], h = cell.hidden_size();
        auto state = initial.value_or(nd::Array({b, h}));
        if (state.shape() != nd::Shape({b, h}))
            throw std::invalid_argument("Initial hidden shape mismatch");
        std::vector<nd::Array> outputs;
        for (int i = 0; i < t; ++i)
        {
            if (detach_every && i && std::size_t(i) % detach_every == 0)
                state = state.detach();
            auto mask = detail::timestep_mask(b, i, valid);
            state = detail::choose(mask, cell.step(detail::timestep(a, i), state), state);
            outputs.push_back(nd::reshape(nd::multiply(mask, state), {b, 1, h}));
        }
        return {nd::concat(outputs, 1), state};
    }
} // namespace ml::nn
