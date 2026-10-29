#include "training.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
namespace ml::training
{
    namespace
    {
        class ModeGuard
        {
            nn::Module &model_;
            bool previous_;

          public:
            ModeGuard(nn::Module &model, bool mode) : model_(model), previous_(model.training())
            {
                model_.train(mode);
            }
            ~ModeGuard()
            {
                model_.train(previous_);
            }
        };
        class LoaderGuard
        {
            data::DataLoader &loader_;
            data::LoaderState previous_;

          public:
            explicit LoaderGuard(data::DataLoader &loader)
                : loader_(loader), previous_(loader.state())
            {
            }
            ~LoaderGuard()
            {
                loader_.restore(previous_);
            }
        };
        void validate_evaluation(const data::DataLoader &loader)
        {
            if (loader.options().shuffle || loader.options().drop_last)
                throw std::invalid_argument(
                    "Evaluation loader must use shuffle=false and drop_last=false");
            if (!loader.size())
                throw std::invalid_argument("Empty evaluation dataset");
        }
    } // namespace
    ClassificationMetrics::ClassificationMetrics(int classes, int ignore)
        : classes_(classes), ignore_(ignore)
    {
        if (classes <= 0)
            throw std::invalid_argument("Invalid metric class count");
        tp_.assign(classes, 0);
        fp_.assign(classes, 0);
        fn_.assign(classes, 0);
    }
    void ClassificationMetrics::add(const nd::Array &logits, const std::vector<int> &targets,
                                    double mean_loss)
    {
        if (logits.shape().empty() || logits.shape().back() != classes_ ||
            logits.size() / classes_ != targets.size() || !std::isfinite(mean_loss) ||
            mean_loss < 0)
            throw std::invalid_argument("Invalid metric logits/targets/loss");
        const auto values = logits.variable().value().cpu_values();
        // Validate before updating accumulated statistics.
        for (int label : targets)
            if (label != ignore_ && (label < 0 || label >= classes_))
                throw std::invalid_argument("Invalid metric label");
        for (double v : values)
            if (!std::isfinite(v))
                throw std::invalid_argument("Non-finite metric logits");
        for (std::size_t i = 0; i < targets.size(); ++i)
        {
            int target = targets[i];
            if (target == ignore_)
                continue;
            auto first = values.begin() + static_cast<std::ptrdiff_t>(i * classes_);
            int predicted = static_cast<int>(std::max_element(first, first + classes_) - first);
            ++count_;
            mean_loss_ += (static_cast<long double>(mean_loss) - mean_loss_) / count_;
            if (predicted == target)
            {
                ++correct_;
                ++tp_[target];
            }
            else
            {
                ++fp_[predicted];
                ++fn_[target];
            }
        }
    }
    Metrics ClassificationMetrics::result() const
    {
        if (!count_)
            throw std::invalid_argument("No unmasked targets in dataset");
        Metrics m;
        m.count = count_;
        m.loss = static_cast<double>(mean_loss_);
        m.accuracy = double(correct_) / count_;
        for (int c = 0; c < classes_; ++c)
        {
            double denom = 2.0 * tp_[c] + fp_[c] + fn_[c];
            if (denom)
                m.macro_f1 += 2.0 * tp_[c] / denom;
        }
        m.macro_f1 /= classes_;
        m.perplexity = m.loss > std::log((std::numeric_limits<double>::max)())
                           ? std::numeric_limits<double>::infinity()
                           : std::exp(m.loss);
        return m;
    }
    Metrics SupervisedTrainer::evaluate(data::DataLoader &loader, const Forward &forward,
                                        int classes, int ignore)
    {
        if (!forward)
            throw std::invalid_argument("Missing forward callback");
        validate_evaluation(loader);
        ModeGuard mode(model_, false);
        ad::NoGradGuard no_grad;
        LoaderGuard restore(loader);
        loader.reset_epoch();
        ClassificationMetrics metrics(classes, ignore);
        while (auto batch = loader.next())
        {
            if (!batch->valid_targets(ignore))
                continue;
            auto logits = forward(*batch);
            double loss = nd::cross_entropy(logits, batch->targets, ignore).item();
            metrics.add(logits, batch->targets, loss);
        }
        return metrics.result();
    }
    FitResult SupervisedTrainer::fit(data::DataLoader &training, data::DataLoader &validation,
                                     const Forward &forward, const FitOptions &options)
    {
        if (!ad::grad_enabled())
            throw std::logic_error("fit cannot run inside NoGradGuard");
        if (&training == &validation || !training.size() || !forward || !options.epochs ||
            options.classes <= 0 || !std::isfinite(options.learning_rate) ||
            options.learning_rate <= 0 || !std::isfinite(options.max_grad_norm) ||
            options.max_grad_norm < 0 || !std::isfinite(options.min_delta) || options.min_delta < 0)
            throw std::invalid_argument("Invalid fit configuration");
        validate_evaluation(validation);
        ModeGuard mode(model_, true);
        FitResult result;
        double best = std::numeric_limits<double>::infinity(), patience_reference = best;
        std::size_t bad_epochs = 0;
        auto params = nn::parameters(model_);
        std::vector<nd::Array> best_values;
        std::vector<Optimizer::State> best_optimizer;
        std::optional<data::LoaderState> best_loader;
        for (std::size_t epoch = 0; epoch < options.epochs; ++epoch)
        {
            double rate = options.schedule ? options.schedule(epoch) : options.learning_rate;
            if (!std::isfinite(rate) || rate <= 0)
                throw std::invalid_argument("Schedule returned an invalid learning rate");
            training.reset_epoch();
            ClassificationMetrics metrics(options.classes, options.ignore_index);
            while (auto batch = training.next())
            {
                if (!batch->valid_targets(options.ignore_index))
                    continue;
                trainer_.zero_grad();
                auto logits = forward(*batch);
                auto loss = nd::cross_entropy(logits, batch->targets, options.ignore_index);
                const double value = loss.item();
                metrics.add(logits, batch->targets, value);
                loss.backward();
                trainer_.step(rate, options.max_grad_norm);
            }
            EpochResult record{
                epoch + 1, rate, metrics.result(),
                evaluate(validation, forward, options.classes, options.ignore_index)};
            result.history.push_back(record);
            // Track the actual lowest validation loss; min_delta controls patience only.
            if (record.validation.loss < best)
            {
                best = record.validation.loss;
                result.best_epoch = epoch + 1;
                if (options.restore_best)
                {
                    best_values.clear();
                    for (auto &p : params)
                        best_values.push_back(p.parameter->value().detach());
                    best_optimizer = trainer_.optimizer().states();
                    best_loader = training.state();
                }
            }
            if (record.validation.loss < patience_reference - options.min_delta)
            {
                patience_reference = record.validation.loss;
                bad_epochs = 0;
            }
            else
                ++bad_epochs;
            if (options.on_epoch)
                options.on_epoch(record);
            if (options.patience && bad_epochs >= options.patience)
            {
                result.stopped_early = true;
                break;
            }
        }
        if (options.restore_best)
        {
            trainer_.restore_optimizer(best_optimizer);
            for (std::size_t i = 0; i < params.size(); ++i)
                params[i].parameter->assign(best_values[i]);
            training.restore(*best_loader);
        }
        return result;
    }
    std::function<double(std::size_t)> cosine_schedule(double rate, double minimum,
                                                       std::size_t epochs, std::size_t warmup)
    {
        if (!std::isfinite(rate) || !std::isfinite(minimum) || rate <= 0 || minimum <= 0 ||
            minimum > rate || !epochs || warmup >= epochs)
            throw std::invalid_argument("Invalid cosine schedule");
        return [=](std::size_t epoch)
        {
            if (epoch >= epochs)
                throw std::out_of_range("Schedule epoch");
            if (epoch < warmup)
                return rate * double(epoch + 1) / double(warmup);
            auto remaining = epochs - warmup;
            double progress = remaining == 1 ? 0.0 : double(epoch - warmup) / double(remaining - 1);
            return minimum + (rate - minimum) * .5 * (1 + std::cos(std::numbers::pi * progress));
        };
    }
    std::function<double(std::size_t)> step_schedule(double rate, std::size_t every, double factor)
    {
        if (!std::isfinite(rate) || rate <= 0 || !every || !std::isfinite(factor) || factor <= 0 ||
            factor > 1)
            throw std::invalid_argument("Invalid step schedule");
        return [=](std::size_t epoch)
        {
            double next = rate * std::pow(factor, double(epoch / every));
            if (!std::isfinite(next) || next <= 0)
                throw std::underflow_error("Learning rate underflow");
            return next;
        };
    }
} // namespace ml::training
