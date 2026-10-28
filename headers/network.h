#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
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

        enum Error { SIMPLE };

    private:
        std::vector<Layer> layers;
        std::size_t layer_nb;
        double learning_rate;
        LearningType learning_type;
        std::mt19937 generator;
        std::function<math::Vector(const math::Vector&, const math::Vector&)> error;
        std::function<double(const math::Vector&, const math::Vector&)> loss_value;
        void validate_topology() const;
        void validate_dataset(const Dataset& dataset) const;

    public:
        NeuralNetwork(std::size_t layer_nb, double learning_rate,
                      LearningType learning_type = AUTO, Error error_type = SIMPLE,
                      std::uint32_t seed = 42);
        void add_layer(int nb_params, int layer_size,
                       Layer::Activation activation_type = Layer::SIGMOID);
        Layer& get(std::size_t i);
        const Layer& get(std::size_t i) const;
        Layer& operator[](std::size_t i) { return get(i); }
        const Layer& operator[](std::size_t i) const { return get(i); }
        // AUTO : taille 0 -> min(32,N). MINI_BATCH exige une taille > 0.
        // ONLINE et BATCH ignorent la taille explicite.
        std::vector<double> learn(const Dataset& dataset, std::size_t max_epochs,
                                  std::size_t batch_size = 0, bool shuffle = false);
        [[nodiscard]] math::Vector output(const std::vector<double>& input) const;
        [[nodiscard]] double mean_loss(const Dataset& dataset) const;
    };
}
