#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include "layer.h"
#include "dataset.h"

namespace ml
{
    class NeuralNetwork
    {
    public:
        enum LearningType { BATCH, ONLINE, MINI_BATCH, AUTO };

        enum Error { SIMPLE, BINARY_CROSS_ENTROPY, CROSS_ENTROPY };

    private:
        std::vector<Layer> layers;
        std::size_t layer_nb;
        double learning_rate;
        LearningType learning_type;
        Error error_type_;
        std::mt19937 generator;
        Optimizer optimizer_;
        std::vector<std::uint64_t> observed_revisions_;
        double l2_ = 0;
        void validate_topology() const;
        void validate_dataset(const Dataset& dataset) const;
        LossKind loss_kind() const;
        Tensor batch_loss_input(const Tensor& input);

    public:
        NeuralNetwork(std::size_t layer_nb, double learning_rate, LearningType type = AUTO, Error error = SIMPLE,
                      std::uint32_t seed = 42);
        void add_layer(int inputs, int outputs, Layer::Activation type = Layer::SIGMOID);
        Layer& get(std::size_t i);
        const Layer& get(std::size_t i) const;
        Layer& operator[](std::size_t i) { return get(i); }
        const Layer& operator[](std::size_t i) const { return get(i); }
        void set_optimizer(OptimizerOptions options);
        const Optimizer& optimizer() const { return optimizer_; }
        std::size_t size() const { return layers.size(); }
        std::vector<double> learn(const Dataset& dataset, std::size_t epochs, std::size_t batch_size = 0,
                                  bool shuffle = false);
        math::Vector output(const std::vector<double>& input) const;
        Tensor output_batch(const Tensor& input) const;
        double mean_loss(const Dataset& dataset) const;
    };
}
