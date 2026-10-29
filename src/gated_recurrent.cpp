#include "gated_recurrent.h"
#include <cmath>
#include <stdexcept>
namespace ml::nn
{
    namespace
    {
        void check_state(const nd::Array &x, const nd::Array &h, int input, int hidden)
        {
            if (x.shape().size() != 2 || x.shape()[1] != input ||
                h.shape() != nd::Shape({x.shape()[0], hidden}))
                throw std::invalid_argument("Recurrent input/state mismatch");
        }
        nd::Array gate(const nd::Array &a, int offset, int hidden, Activation activation)
        {
            return nd::activate(nd::slice(a, 1, offset * hidden, hidden), activation);
        }
    } // namespace
    GRUCell::GRUCell(int input, int hidden, std::uint32_t seed) : input_(input), hidden_(hidden)
    {
        nd::numel({hidden, 3});
        int gates = hidden * 3;
        std::mt19937 gen(seed);
        weight_x = normal_parameter({input, gates}, 1 / std::sqrt(double(input)), gen);
        weight_h = normal_parameter({hidden, gates}, 1 / std::sqrt(double(hidden)), gen);
        bias_x = std::make_shared<Parameter>(nd::Array({gates}));
        bias_h = std::make_shared<Parameter>(nd::Array({gates}));
    }
    nd::Array GRUCell::step(const nd::Array &x, const nd::Array &h)
    {
        check_state(x, h, input_, hidden_);
        auto ix = nd::add(nd::matmul(x, weight_x->value()), bias_x->value());
        auto hh = nd::add(nd::matmul(h, weight_h->value()), bias_h->value());
        auto r = gate(nd::add(ix, hh), 0, hidden_, Activation::Sigmoid),
             z = gate(nd::add(ix, hh), 1, hidden_, Activation::Sigmoid);
        auto n = nd::activate(nd::add(nd::slice(ix, 1, 2 * hidden_, hidden_),
                                      nd::multiply(r, nd::slice(hh, 1, 2 * hidden_, hidden_))),
                              Activation::Tanh);
        return detail::choose(z, h, n);
    }
    std::vector<NamedParameter> GRUCell::named_parameters() const
    {
        return {
            {"weight_x", weight_x}, {"weight_h", weight_h}, {"bias_x", bias_x}, {"bias_h", bias_h}};
    }
    RNNResult GRU::run(const nd::Array &a, const std::vector<int> &lengths,
                       std::optional<nd::Array> initial, std::size_t detach_every)
    {
        auto valid = detail::sequence_lengths(a, cell.input_size(), lengths);
        int b = a.shape()[0], t = a.shape()[1], h = cell.hidden_size();
        auto state = initial.value_or(nd::Array({b, h}));
        if (state.shape() != nd::Shape({b, h}))
            throw std::invalid_argument("Initial GRU state mismatch");
        std::vector<nd::Array> out;
        for (int i = 0; i < t; ++i)
        {
            if (detach_every && i && std::size_t(i) % detach_every == 0)
                state = state.detach();
            auto mask = detail::timestep_mask(b, i, valid);
            state = detail::choose(mask, cell.step(detail::timestep(a, i), state), state);
            out.push_back(nd::reshape(nd::multiply(mask, state), {b, 1, h}));
        }
        return {nd::concat(out, 1), state};
    }
    LSTMCell::LSTMCell(int input, int hidden, std::uint32_t seed) : input_(input), hidden_(hidden)
    {
        nd::numel({hidden, 4});
        int gates = hidden * 4;
        std::mt19937 gen(seed);
        weight_x = normal_parameter({input, gates}, 1 / std::sqrt(double(input)), gen);
        weight_h = normal_parameter({hidden, gates}, 1 / std::sqrt(double(hidden)), gen);
        nd::Vector values(gates, 0);
        for (int i = hidden; i < 2 * hidden; ++i)
            values[i] = 1;
        bias = std::make_shared<Parameter>(nd::Array::from({gates}, values));
    }
    LSTMState LSTMCell::step(const nd::Array &x, const LSTMState &state)
    {
        check_state(x, state.hidden, input_, hidden_);
        if (state.cell.shape() != state.hidden.shape())
            throw std::invalid_argument("LSTM cell state mismatch");
        auto a = nd::add(
            nd::add(nd::matmul(x, weight_x->value()), nd::matmul(state.hidden, weight_h->value())),
            bias->value());
        auto i = gate(a, 0, hidden_, Activation::Sigmoid),
             f = gate(a, 1, hidden_, Activation::Sigmoid),
             g = gate(a, 2, hidden_, Activation::Tanh),
             o = gate(a, 3, hidden_, Activation::Sigmoid);
        auto c = nd::add(nd::multiply(f, state.cell), nd::multiply(i, g));
        return {nd::multiply(o, nd::activate(c, Activation::Tanh)), c};
    }
    std::vector<NamedParameter> LSTMCell::named_parameters() const
    {
        return {{"weight_x", weight_x}, {"weight_h", weight_h}, {"bias", bias}};
    }
    LSTMResult LSTM::run(const nd::Array &a, const std::vector<int> &lengths,
                         std::optional<LSTMState> initial, std::size_t detach_every)
    {
        auto valid = detail::sequence_lengths(a, cell.input_size(), lengths);
        int b = a.shape()[0], t = a.shape()[1], h = cell.hidden_size();
        auto state = initial.value_or(LSTMState{nd::Array({b, h}), nd::Array({b, h})});
        if (state.hidden.shape() != nd::Shape({b, h}) || state.cell.shape() != state.hidden.shape())
            throw std::invalid_argument("Initial LSTM state mismatch");
        std::vector<nd::Array> out;
        for (int i = 0; i < t; ++i)
        {
            if (detach_every && i && std::size_t(i) % detach_every == 0)
            {
                state.hidden = state.hidden.detach();
                state.cell = state.cell.detach();
            }
            auto mask = detail::timestep_mask(b, i, valid);
            auto next = cell.step(detail::timestep(a, i), state);
            state = {detail::choose(mask, next.hidden, state.hidden),
                     detail::choose(mask, next.cell, state.cell)};
            out.push_back(nd::reshape(nd::multiply(mask, state.hidden), {b, 1, h}));
        }
        return {nd::concat(out, 1), state.hidden, state.cell};
    }
} // namespace ml::nn
