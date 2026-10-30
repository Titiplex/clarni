#pragma once
#include "nd.h"
#include "optimizer.h"
#include <memory>
#include <random>
#include <string>

namespace ml::nn
{
    class Parameter
    {
        nd::Array value_;

      public:
        explicit Parameter(const nd::Array &);

        [[nodiscard]] const nd::Array &value() const
        {
            return value_;
        }

        void assign(const nd::Array &); // same shape, fresh leaf
        void zero_grad();
        void to(Backend);
    };

    using Param = std::shared_ptr<Parameter>;

    struct NamedParameter
    {
        std::string name;
        Param parameter;
    };

    class Module
    {
        bool training_ = true;

      public:
        Module() = default;
        Module(const Module &) = delete;
        Module &operator=(const Module &) = delete;
        virtual ~Module() = default;
        virtual nd::Array forward(const nd::Array &);

        [[nodiscard]] virtual std::vector<NamedParameter> named_parameters() const
        {
            return {};
        }

        void to(Backend);
        [[nodiscard]] Backend backend() const;
        virtual void train(const bool enabled = true)
        {
            training_ = enabled;
        }

        void eval()
        {
            train(false);
        }

        [[nodiscard]] bool training() const
        {
            return training_;
        }
    };

    std::vector<NamedParameter> parameters(const Module &); // deduplicated identities
    Param normal_parameter(nd::Shape, double stddev, std::mt19937 &);

    class Linear final : public Module
    {
        int in_, out_;

      public:
        Param weight, bias; // [in,out], [out]
        Linear(int inputs, int outputs, std::uint32_t seed = 42);
        nd::Array forward(const nd::Array &) override;
        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override;
    };

    class ActivationLayer final : public Module
    {
        Activation kind_;

      public:
        explicit ActivationLayer(Activation kind) : kind_(kind) {}

        nd::Array forward(const nd::Array &a) override
        {
            return nd::activate(a, kind_);
        }
    };

    class Flatten final : public Module
    {
      public:
        nd::Array forward(const nd::Array &) override; // keep batch dimension
    };

    class Sequential final : public Module
    {
        std::vector<std::shared_ptr<Module>> modules_;

      public:
        void add(std::shared_ptr<Module>);
        nd::Array forward(const nd::Array &) override;
        [[nodiscard]] std::vector<NamedParameter> named_parameters() const override;
        void train(bool enabled = true) override;
    };

    class Trainer
    {
        Module &module_;
        std::vector<NamedParameter> bindings_;
        Optimizer optimizer_;
        void validate_bindings() const;

      public:
        explicit Trainer(Module &, const OptimizerOptions &options = {});
        void zero_grad() const; // BEFORE forward; backward resets visited graph gradients
        [[nodiscard]] double grad_norm() const;
        void step(double rate, double max_grad_norm = 0);

        [[nodiscard]] const Optimizer &optimizer() const
        {
            return optimizer_;
        }

        void restore_optimizer(const std::vector<Optimizer::State> &);
        void validate_model(const Module &) const;
    };

    // Caller reconstructs topology. Strict names/shapes. Optimizer optional.
    void save_checkpoint(const std::string &, const Module &, const Trainer * = nullptr);
    void load_checkpoint(const std::string &, Module &, Trainer * = nullptr);
} // namespace ml::nn
