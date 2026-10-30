#pragma once
#include <cstddef>
#include "embedding.h"
#include <functional>
#include <optional>
#include <random>
#include <variant>

namespace ml::data
{
    using SampleInput = std::variant<nd::Array, std::vector<int>>;

    struct Example
    {
        SampleInput input;
        std::vector<int> targets;
    };

    struct Batch
    {
        std::variant<nd::Array, nn::TokenBatch> input;
        std::vector<int> targets, lengths;
        std::vector<std::size_t> indices;
        std::size_t valid_targets(int ignore_index = -1) const;
    };

    class Dataset
    {
    public:
        virtual ~Dataset() = default;
        virtual std::size_t size() const = 0;
        virtual Example get(std::size_t index) const = 0;
    };

    class VectorDataset final : public Dataset
    {
        std::vector<Example> data_;

    public:
        explicit VectorDataset(std::vector<Example> data) : data_(std::move(data))
        {
        }

        std::size_t size() const override
        {
            return data_.size();
        }

        Example get(const std::size_t i) const override
        {
            return data_.at(i);
        }
    };

    class Subset final : public Dataset
    {
        std::shared_ptr<const Dataset> source_;
        std::vector<std::size_t> indices_;

    public:
        Subset(std::shared_ptr<const Dataset>, std::vector<std::size_t>);

        std::size_t size() const override
        {
            return indices_.size();
        }

        Example get(const std::size_t i) const override
        {
            return source_->get(indices_.at(i));
        }
    };

    struct Split
    {
        std::vector<std::size_t> training, validation, test;
    };

    Split random_split(std::size_t size, std::size_t validation, std::size_t test,
                       std::uint32_t seed = 42);

    enum class InputLayout
    {
        Stack,
        PadTokens,
        PadFeatures
    };

    enum class TargetLayout
    {
        Class,
        Sequence
    };

    struct CollateOptions
    {
        InputLayout input = InputLayout::Stack;
        TargetLayout targets = TargetLayout::Class;
        int padding_id = 0, ignore_index = -1;
    };

    using Collator =
    std::function<Batch(const std::vector<Example>&, const std::vector<std::size_t>&)>;
    Collator make_collator(CollateOptions options = {});

    struct LoaderOptions
    {
        std::size_t batch_size = 32;
        bool shuffle = true, drop_last = false;
        std::uint32_t seed = 42;
        std::string dataset_key; // caller-provided identity checked when restoring
    };

    struct LoaderState
    {
        std::vector<std::size_t> order;
        std::size_t cursor = 0, epoch = 0, batch_size = 0;
        bool shuffle = false, drop_last = false;
        std::string rng, dataset_key;
    };

    class DataLoader
    {
        std::shared_ptr<const Dataset> dataset_;
        LoaderOptions options_;
        Collator collate_;
        std::mt19937 generator_;
        std::vector<std::size_t> order_;
        std::size_t cursor_, epoch_ = 0;

    public:
        DataLoader(std::shared_ptr<const Dataset>, LoaderOptions options = {},
                   Collator collator = make_collator());
        void reset_epoch(); // starts a new epoch; constructor is initially exhausted
        std::optional<Batch> next();

        std::size_t size() const
        {
            return order_.size();
        }

        const LoaderOptions& options() const
        {
            return options_;
        }

        LoaderState state() const;
        void restore(const LoaderState&); // validates entire state before mutation
        void save_state(const std::string&) const;
        void load_state(const std::string&);
    };

    // Raw (uncompressed) MNIST IDX files; bytes retained, samples converted on demand.
    class MNIST final : public Dataset
    {
        int rows_ = 0, cols_ = 0;
        std::vector<unsigned char> pixels_, labels_;

    public:
        MNIST(const std::string& images, const std::string& labels, std::size_t limit = 0);

        std::size_t size() const override
        {
            return labels_.size();
        }

        Example get(std::size_t) const override;

        int rows() const
        {
            return rows_;
        }

        int cols() const
        {
            return cols_;
        }
    };
} // namespace ml::data
