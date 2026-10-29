#include "attention.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ml::nn
{
    namespace
    {
        using nd::Array;
        using nd::Shape;
        using nd::Vector;

        std::vector<NamedParameter>
        collect(const std::vector<std::pair<std::string, const Module*>>& children)
        {
            std::vector<NamedParameter> out;
            for (auto& [prefix, module] : children)
                for (auto p : module->named_parameters())
                {
                    p.name = prefix + p.name;
                    out.push_back(std::move(p));
                }
            return out;
        }

        Array mask_queries(const Array& a, const std::vector<int>& lengths)
        {
            Vector mask(static_cast<std::size_t>(a.shape()[0]) * a.shape()[1]);
            for (int b = 0; b < a.shape()[0]; ++b)
                for (int t = 0; t < a.shape()[1]; ++t)
                    mask[static_cast<std::size_t>(b) * a.shape()[1] + t] = t < lengths[b] ? 1 : 0;
            return nd::multiply(a, Array::from({a.shape()[0], a.shape()[1], 1}, mask));
        }

        Array masked_softmax(const Array& a, const std::vector<bool>& mask)
        {
            int rows = a.shape()[0], cols = a.shape()[1];
            if (mask.size() != a.size())
                throw std::invalid_argument("Attention mask size mismatch");
            const auto x = a.values();
            Vector y(a.size(), 0);
            for (int i = 0; i < rows; ++i)
            {
                double maximum = -std::numeric_limits<double>::infinity();
                bool valid = false;
                for (int j = 0; j < cols; ++j)
                    if (mask[static_cast<std::size_t>(i) * cols + j])
                    {
                        maximum = (std::max)(maximum, x[static_cast<std::size_t>(i) * cols + j]);
                        valid = true;
                    }
                if (!valid)
                    continue;
                if (!std::isfinite(maximum))
                    throw std::invalid_argument("Non-finite attention logits");
                double total = 0;
                for (int j = 0; j < cols; ++j)
                    if (mask[static_cast<std::size_t>(i) * cols + j])
                    {
                        const auto index = static_cast<std::size_t>(i) * cols + j;
                        y[index] = std::exp(x[index] - maximum);
                        total += y[index];
                    }
                for (int j = 0; j < cols; ++j)
                    y[static_cast<std::size_t>(i) * cols + j] /= total;
            }
            return nd::operation(a.shape(), y, {
                                     {
                                         a, [y, rows, cols](const Vector& g)
                                         {
                                             Vector r(g.size(), 0);
                                             for (int i = 0; i < rows; ++i)
                                             {
                                                 double dot = 0;
                                                 for (int j = 0; j < cols; ++j)
                                                 {
                                                     const auto k = static_cast<std::size_t>(i) * cols + j;
                                                     dot += g[k] * y[k];
                                                 }
                                                 for (int j = 0; j < cols; ++j)
                                                 {
                                                     const auto k = static_cast<std::size_t>(i) * cols + j;
                                                     r[k] = y[k] * (g[k] - dot);
                                                 }
                                             }
                                             return r;
                                         }
                                     }
                                 }
            );
        }
    } // namespace
    LayerNorm::LayerNorm(int features, const double epsilon) : features_(features), epsilon_(epsilon)
    {
        nd::numel({features});
        if (!std::isfinite(epsilon) || epsilon <= 0)
            throw std::invalid_argument("Invalid normalization epsilon");
        weight = std::make_shared<Parameter>(Array({features}, 1));
        bias = std::make_shared<Parameter>(Array({features}));
    }

    Array LayerNorm::forward(const Array& a)
    {
        if (a.shape().empty() || a.shape().back() != features_)
            throw std::invalid_argument("LayerNorm feature mismatch");
        const auto x = a.values();
        Vector y(a.size()), inverses(a.size() / features_);
        int d = features_;
        for (std::size_t row = 0; row < inverses.size(); ++row)
        {
            double mean = 0, m2 = 0;
            for (int j = 0; j < d; ++j)
            {
                const double delta = x[row * d + j] - mean;
                mean += delta / (j + 1);
                m2 += delta * (x[row * d + j] - mean);
            }
            inverses[row] = 1 / std::sqrt((std::max)(0.0, m2 / d) + epsilon_);
            for (int j = 0; j < d; ++j)
                y[row * d + j] = (x[row * d + j] - mean) * inverses[row];
        }
        const auto normalized = nd::operation(
            a.shape(), y,
            {
                {
                    a, [y, inverses, d](const Vector& g)
                    {
                        Vector result(g.size());
                        for (std::size_t row = 0; row < inverses.size(); ++row)
                        {
                            double sum = 0, dot = 0;
                            for (int j = 0; j < d; ++j)
                            {
                                sum += g[row * d + j];
                                dot += g[row * d + j] * y[row * d + j];
                            }
                            for (int j = 0; j < d; ++j)
                                result[row * d + j] =
                                    inverses[row] * (g[row * d + j] - (sum + y[row * d + j] * dot) / d);
                        }
                        return result;
                    }
                }
            });
        return nd::add(nd::multiply(normalized, weight->value()), bias->value());
    }

    Array scaled_dot_product_attention(const Array& q, const Array& k, const Array& v,
                                       const std::vector<bool>& allowed)
    {
        if (q.shape().size() != 2 || k.shape().size() != 2 || v.shape().size() != 2 ||
            q.shape()[1] != k.shape()[1] || k.shape()[0] != v.shape()[0])
            throw std::invalid_argument("Attention Q/K/V shape mismatch");
        const auto scores =
            nd::scale(nd::matmul(q, nd::permute(k, {1, 0})), 1 / std::sqrt(static_cast<double>(q.shape()[1])));
        const auto mask = allowed.empty() ? std::vector<bool>(scores.size(), true) : allowed;
        return nd::matmul(masked_softmax(scores, mask), v);
    }

    MultiHeadSelfAttention::MultiHeadSelfAttention(const int features, const int heads, const std::uint32_t seed)
        : features_(features), heads_(heads), query(features, features, seed),
          key(features, features, seed + 1), value(features, features, seed + 2),
          projection(features, features, seed + 3)
    {
        if (heads <= 0 || features % heads)
            throw std::invalid_argument("Attention heads must divide features");
    }

    Array MultiHeadSelfAttention::attend(const Array& a, const std::vector<int>& lengths,
                                         const bool causal)
    {
        const auto valid = detail::sequence_lengths(a, features_, lengths);
        const int b = a.shape()[0];
        int t = a.shape()[1], d = features_ / heads_;
        const auto q = query.forward(a);
        const auto k = key.forward(a);
        const auto v = value.forward(a);
        std::vector<Array> batches;
        for (int n = 0; n < b; ++n)
        {
            std::vector<Array> head_values;
            std::vector allowed(static_cast<std::size_t>(t) * t, false);
            for (int i = 0; i < t; ++i)
                for (int j = 0; j < t; ++j)
                    allowed[static_cast<std::size_t>(i) * t + j] =
                        i < valid[n] && j < valid[n] && (!causal || j <= i);
            auto get = [&](const Array& x, const int head)
            {
                return nd::reshape(nd::slice(nd::slice(x, 0, n, 1), 2, head * d, d), {t, d});
            };
            for (int head = 0; head < heads_; ++head)
                head_values.push_back(scaled_dot_product_attention(get(q, head), get(k, head),
                                                                   get(v, head), allowed));
            batches.push_back(nd::reshape(nd::concat(head_values, 1), {1, t, features_}));
        }
        return mask_queries(projection.forward(nd::concat(batches, 0)), valid);
    }

    std::vector<NamedParameter> MultiHeadSelfAttention::named_parameters() const
    {
        return collect(
            {{"query.", &query}, {"key.", &key}, {"value.", &value}, {"projection.", &projection}});
    }

    void MultiHeadSelfAttention::train(const bool enabled)
    {
        Module::train(enabled);
        query.train(enabled);
        key.train(enabled);
        value.train(enabled);
        projection.train(enabled);
    }

    SinusoidalPosition::SinusoidalPosition(int features, int max_length)
        : features_(features), max_length_(max_length)
    {
        nd::numel({features, max_length});
    }

    Array SinusoidalPosition::add(const Array& a, const int start)
    {
        if (a.shape().size() != 3 || a.shape()[2] != features_ || start < 0 ||
            a.shape()[1] > max_length_ || start > max_length_ - a.shape()[1])
            throw std::invalid_argument("Positional encoding shape/range mismatch");
        Vector values(static_cast<std::size_t>(a.shape()[1]) * features_);
        for (int t = 0; t < a.shape()[1]; ++t)
            for (int d = 0; d < features_; ++d)
            {
                const double angle =
                    (static_cast<double>(start) + t) / std::pow(10000.0, 2.0 * (d / 2) / features_);
                values[static_cast<std::size_t>(t) * features_ + d] = d % 2 ? std::cos(angle) : std::sin(angle);
            }
        return nd::add(a, Array::from({1, a.shape()[1], features_}, values));
    }

    TransformerBlock::TransformerBlock(const int features, const int heads, const int feedforward,
                                       const std::uint32_t seed)
        : features_(features), norm1(features), norm2(features), attention(features, heads, seed),
          expand(features, feedforward, seed + 4), contract(feedforward, features, seed + 5)
    {
    }

    Array TransformerBlock::run(const Array& a, const std::vector<int>& lengths, const bool causal)
    {
        const auto valid = detail::sequence_lengths(a, features_, lengths);
        const auto x = nd::add(a, attention.attend(norm1.forward(a), valid, causal));
        const auto y = nd::add(
            x, contract.forward(nd::activate(expand.forward(norm2.forward(x)), Activation::Relu)));
        return mask_queries(y, valid);
    }

    std::vector<NamedParameter> TransformerBlock::named_parameters() const
    {
        return collect({
            {"norm1.", &norm1},
            {"norm2.", &norm2},
            {"attention.", &attention},
            {"expand.", &expand},
            {"contract.", &contract}
        });
    }

    void TransformerBlock::train(const bool enabled)
    {
        Module::train(enabled);
        norm1.train(enabled);
        norm2.train(enabled);
        attention.train(enabled);
        expand.train(enabled);
        contract.train(enabled);
    }

    TransformerLM::TransformerLM(const int vocabulary, const int features, const int heads, const int feedforward,
                                 const int max_length, const std::uint32_t seed)
        : embedding(vocabulary, features, -1, seed), positions(features, max_length),
          block(features, heads, feedforward, seed + 1), norm(features),
          head(features, vocabulary, seed + 7)
    {
    }

    Array TransformerLM::logits(const TokenBatch& batch, const std::vector<int>& lengths)
    {
        if (batch.shape.size() != 2)
            throw std::invalid_argument("TransformerLM IDs must be [B,T]");
        return head.forward(
            norm.forward(block.run(positions.forward(embedding.lookup(batch)), lengths, true)));
    }

    std::vector<NamedParameter> TransformerLM::named_parameters() const
    {
        return collect(
            {{"embedding.", &embedding}, {"block.", &block}, {"norm.", &norm}, {"head.", &head}});
    }

    void TransformerLM::train(const bool enabled)
    {
        Module::train(enabled);
        embedding.train(enabled);
        positions.train(enabled);
        block.train(enabled);
        norm.train(enabled);
        head.train(enabled);
    }
} // namespace ml::nn
