#include "embedding.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ml::nn
{
    Vocabulary::Vocabulary(const std::vector<std::string>& corpus, std::size_t min_count)
    {
        if (!min_count)
            throw std::invalid_argument("min_count must be positive");
        std::unordered_map<std::string, std::size_t> counts;
        for (auto& token : corpus)
        {
            if (token.empty())
                throw std::invalid_argument("Empty vocabulary token");
            ++counts[token];
        }
        std::vector<std::string> words;
        for (auto& [word, count] : counts)
            if (count >= min_count && !indices_.contains(word))
                words.push_back(word);
        std::ranges::sort(words);
        for (auto& word : words)
        {
            if (tokens_.size() >= static_cast<std::size_t>((std::numeric_limits<int>::max)()))
                throw std::length_error("Vocabulary too large");
            indices_[word] = static_cast<int>(tokens_.size());
            tokens_.push_back(word);
        }
    }

    int Vocabulary::id(const std::string& word) const
    {
        const auto it = indices_.find(word);
        return it == indices_.end() ? unk_id : it->second;
    }

    const std::string& Vocabulary::token(const int i) const
    {
        if (i < 0 || i >= size())
            throw std::out_of_range("Token id");
        return tokens_[i];
    }

    std::vector<int> Vocabulary::encode(const std::vector<std::string>& words) const
    {
        std::vector<int> ids;
        for (auto& w : words)
            ids.push_back(id(w));
        return ids;
    }

    void Vocabulary::save(const std::string& path) const
    {
        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("Cannot write vocabulary");
        out << "CLARNI_VOCAB 1\n" << size() << '\n';
        for (auto& token : tokens_)
            out << std::quoted(token) << '\n';
        if (!out)
            throw std::runtime_error("Vocabulary write failed");
    }

    Vocabulary Vocabulary::load(const std::string& path)
    {
        std::ifstream in(path);
        std::string magic;
        int version, count;
        if (!(in >> magic >> version >> count) || magic != "CLARNI_VOCAB" || version != 1 ||
            count < 2)
            throw std::invalid_argument("Invalid vocabulary header");
        Vocabulary out;
        out.tokens_.clear();
        out.indices_.clear();
        for (int i = 0; i < count; ++i)
        {
            std::string token;
            if (!(in >> std::quoted(token)) || token.empty() || out.indices_.contains(token))
                throw std::invalid_argument("Invalid vocabulary token");
            out.indices_[token] = i;
            out.tokens_.push_back(token);
        }
        in >> std::ws;
        if (!in.eof() || out.tokens_[0] != "<pad>" || out.tokens_[1] != "<unk>")
            throw std::invalid_argument("Invalid vocabulary specials/trailer");
        return out;
    }

    std::vector<std::string> whitespace_tokens(const std::string& text)
    {
        std::istringstream in(text);
        std::vector<std::string> out;
        std::string token;
        while (in >> token)
            out.push_back(token);
        return out;
    }

    Embedding::Embedding(int vocabulary, int dimension, int padding_idx, const std::uint32_t seed)
        : vocabulary_(vocabulary), dimension_(dimension), padding_(padding_idx)
    {
        nd::numel({vocabulary, dimension});
        if (padding_idx < -1 || padding_idx >= vocabulary)
            throw std::invalid_argument("Invalid padding index");
        std::mt19937 gen(seed);
        weight = normal_parameter({vocabulary, dimension}, 1.0 / std::sqrt(static_cast<double>(dimension)), gen);
        if (padding_idx >= 0)
        {
            auto values = weight->value().values();
            std::fill(values.begin() + padding_idx * dimension, values.begin() + (padding_idx + 1) * dimension, 0);
            weight->assign(nd::Array::from({vocabulary, dimension}, values));
        }
    }

    nd::Array Embedding::lookup(const TokenBatch& batch)
    {
        if (nd::numel(batch.shape) != batch.ids.size())
            throw std::invalid_argument("Embedding ID shape mismatch");
        auto shape = batch.shape;
        shape.push_back(dimension_);
        nd::Vector values(nd::numel(shape));
        const auto table = weight->value().values();
        for (std::size_t i = 0; i < batch.ids.size(); ++i)
        {
            const int id = batch.ids[i];
            if (id < 0 || id >= vocabulary_)
                throw std::invalid_argument("Embedding ID outside vocabulary");
            for (int d = 0; d < dimension_; ++d)
                values[i * dimension_ + d] = table[id * dimension_ + d];
        }
        return nd::operation(
            shape, std::move(values),
            {
                {
                    weight->value(), [ids = batch.ids, dim = dimension_, vocab = vocabulary_,
                        pad = padding_](const nd::Vector& g)
                    {
                        nd::Vector result(static_cast<std::size_t>(vocab) * dim, 0);
                        for (std::size_t i = 0; i < ids.size(); ++i)
                            if (ids[i] != pad)
                            {
                                for (int d = 0; d < dim; ++d)
                                    result[ids[i] * dim + d] += g[i * dim + d];
                            }
                        return result;
                    }
                }
            });
    }

    ContextPairs skipgram_pairs(const std::vector<int>& sentence, const int window, int padding_id)
    {
        if (window <= 0)
            throw std::invalid_argument("Invalid context window");
        ContextPairs pairs;
        for (std::size_t i = 0; i < sentence.size(); ++i)
            if (sentence[i] != padding_id)
            {
                const auto first = i > static_cast<std::size_t>(window) ? i - window : 0;
                const auto last = i + (std::min)(static_cast<std::size_t>(window), sentence.size() - 1 - i);
                for (auto j = first; j <= last; ++j)
                    if (j != i && sentence[j] != padding_id)
                    {
                        pairs.centers.push_back(sentence[i]);
                        pairs.contexts.push_back(sentence[j]);
                    }
            }
        return pairs;
    }

    SkipGram::SkipGram(const int vocabulary, const int dimension, const std::uint32_t seed)
        : input(vocabulary, dimension, -1, seed), output(vocabulary, dimension, -1, seed + 1)
    {
    }

    nd::Array SkipGram::logits(const std::vector<int>& centers)
    {
        return nd::matmul(input.lookup({centers, {static_cast<int>(centers.size())}}),
                          nd::permute(output.weight->value(), {1, 0}));
    }

    std::vector<NamedParameter> SkipGram::named_parameters() const
    {
        return {{"input.weight", input.weight}, {"output.weight", output.weight}};
    }
} // namespace ml::nn
