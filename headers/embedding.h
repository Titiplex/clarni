#pragma once
#include "module.h"
#include <unordered_map>
namespace ml::nn
{
    class Vocabulary
    {
        std::vector<std::string> tokens_{"<pad>", "<unk>"};
        std::unordered_map<std::string, int> indices_{{"<pad>", 0}, {"<unk>", 1}};

      public:
        static constexpr int pad_id = 0, unk_id = 1;
        explicit Vocabulary(const std::vector<std::string> &corpus = {}, std::size_t min_count = 1);
        int id(const std::string &) const;
        const std::string &token(int) const;
        int size() const
        {
            return static_cast<int>(tokens_.size());
        }
        std::vector<int> encode(const std::vector<std::string> &) const;
        void save(const std::string &) const;
        static Vocabulary load(const std::string &);
    };
    // Whitespace only: normalization/tokenization (especially Unicode/IPA) is caller-owned.
    std::vector<std::string> whitespace_tokens(const std::string &);
    struct TokenBatch
    {
        std::vector<int> ids;
        nd::Shape shape;
    };
    class Embedding final : public Module
    {
        int vocabulary_, dimension_, padding_;

      public:
        Param weight; // [vocabulary,dimension], dense gradients
        Embedding(int vocabulary, int dimension, int padding_idx = -1, std::uint32_t seed = 42);
        nd::Array lookup(const TokenBatch &);
        std::vector<NamedParameter> named_parameters() const override
        {
            return {{"weight", weight}};
        }
    };
    struct ContextPairs
    {
        std::vector<int> centers, contexts;
    };
    ContextPairs skipgram_pairs(const std::vector<int> &sentence, int window, int padding_id = 0);
    // Full-softmax Skip-gram: two embedding tables, no negative sampling.
    class SkipGram final : public Module
    {
      public:
        Embedding input, output;
        SkipGram(int vocabulary, int dimension, std::uint32_t seed = 42);
        nd::Array logits(const std::vector<int> &centers);
        std::vector<NamedParameter> named_parameters() const override;
    };
} // namespace ml::nn
