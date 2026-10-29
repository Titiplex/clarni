#pragma once
#include "embedding.h"
#include "recurrent.h"

namespace ml::nn
{
    class LayerNorm final : public Module
    {
        int features_;
        double epsilon_;

    public:
        Param weight, bias;
        explicit LayerNorm(int features, double epsilon = 1e-5);
        nd::Array forward(const nd::Array&) override;

        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override
        {
            return {{"weight", weight}, {"bias", bias}};
        }
    };

    // Rank-2 [queries,features], [keys,features], [keys,value_features].
    // Allowed mask row-major [queries,keys]. All-masked rows yield zero and zero VJP.
    nd::Array scaled_dot_product_attention(const nd::Array& q, const nd::Array& k,
                                           const nd::Array& v,
                                           const std::vector<bool>& allowed = {});

    class MultiHeadSelfAttention final : public Module
    {
        int features_, heads_;

    public:
        Linear query, key, value, projection;
        MultiHeadSelfAttention(int features, int heads, std::uint32_t seed = 42);
        nd::Array attend(const nd::Array&, const std::vector<int>& lengths = {},
                         bool causal = true);

        nd::Array forward(const nd::Array& a) override
        {
            return attend(a);
        }

        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override;
        void train(bool enabled = true) override;
    };

    class SinusoidalPosition final : public Module
    {
        int features_, max_length_;

    public:
        SinusoidalPosition(int features, int max_length);
        nd::Array add(const nd::Array&, int start = 0);

        nd::Array forward(const nd::Array& a) override
        {
            return add(a);
        }
    };

    // Deterministic pre-norm block: attention + residual, feedforward + residual.
    // No dropout, KV cache or optimized attention kernel in this reference implementation.
    class TransformerBlock final : public Module
    {
        int features_;

    public:
        LayerNorm norm1, norm2;
        MultiHeadSelfAttention attention;
        Linear expand, contract;
        TransformerBlock(int features, int heads, int feedforward, std::uint32_t seed = 42);
        nd::Array run(const nd::Array&, const std::vector<int>& lengths = {}, bool causal = true);

        nd::Array forward(const nd::Array& a) override
        {
            return run(a);
        }

        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override;
        void train(bool enabled = true) override;
    };

    class TransformerLM final : public Module
    {
    public:
        Embedding embedding;
        SinusoidalPosition positions;
        TransformerBlock block;
        LayerNorm norm;
        Linear head;
        TransformerLM(int vocabulary, int features, int heads, int feedforward, int max_length,
                      std::uint32_t seed = 42);
        nd::Array logits(const TokenBatch&, const std::vector<int>& lengths = {}); // [B,T,V]
        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override;
        void train(bool enabled = true) override;
    };
} // namespace ml::nn
