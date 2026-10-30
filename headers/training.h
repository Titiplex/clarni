#pragma once
#include <cstddef>
#include "data_loader.h"
#include <functional>

namespace ml::training
{
    struct Metrics
    {
        double loss = 0, accuracy = 0, macro_f1 = 0, perplexity = 0;
        std::size_t count = 0;
    };

    // O(classes) memory; macro F1 averages over every declared class (absent -> 0).
    class ClassificationMetrics
    {
        int classes_, ignore_;
        std::size_t count_ = 0, correct_ = 0;
        long double mean_loss_ = 0;
        std::vector<std::size_t> tp_, fp_, fn_;

    public:
        explicit ClassificationMetrics(int classes, int ignore_index = -1);
        void add(const nd::Array& logits, const std::vector<int>& targets, double mean_loss);
        Metrics result() const;
    };

    using Forward = std::function<nd::Array(const data::Batch&)>;

    struct EpochResult
    {
        std::size_t epoch = 0;
        double learning_rate = 0;
        Metrics training, validation;
    };

    struct FitOptions
    {
        std::size_t epochs = 10, patience = 0;
        int classes = 2, ignore_index = -1;
        double learning_rate = .001, max_grad_norm = 0, min_delta = 0;
        bool restore_best = true;
        // epoch index starts at zero; empty means constant learning rate.
        std::function<double(std::size_t)> schedule;
        std::function<void(const EpochResult&)> on_epoch;
    };

    struct FitResult
    {
        std::vector<EpochResult> history;
        std::size_t best_epoch = 0;
        bool stopped_early = false;
    };

    class SupervisedTrainer
    {
        nn::Module& model_;
        nn::Trainer trainer_;

    public:
        explicit SupervisedTrainer(nn::Module& model, const OptimizerOptions& options = {})
            : model_(model), trainer_(model, options)
        {
        }

        nn::Trainer& optimizer()
        {
            return trainer_;
        }

        // Evaluation uses all examples, disables recording and preserves loader state.
        Metrics evaluate(data::DataLoader&, const Forward&, int classes, int ignore_index = -1) const;
        FitResult fit(data::DataLoader& training, data::DataLoader& validation, const Forward&,
                      const FitOptions& options = {});
    };

    // Deterministic epoch schedules. Total epochs includes warmup. Cosine hits min_rate.
    std::function<double(std::size_t)> cosine_schedule(double rate, double min_rate,
                                                       std::size_t epochs,
                                                       std::size_t warmup_epochs = 0);
    std::function<double(std::size_t)> step_schedule(double rate, std::size_t every, double factor);
} // namespace ml::training
