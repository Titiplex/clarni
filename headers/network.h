#pragma once
#include <vector>
#include "layer.h"
#include "dataset.h"
#include <functional>
#include <stdexcept>

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
        std::function<math::Vector(math::Vector, math::Vector)> error;

    public:
        NeuralNetwork(const std::size_t layer_nb, const double learning_rate, const LearningType learning_type = AUTO, const Error error_type = SIMPLE) :
            layer_nb(layer_nb),
            learning_rate(learning_rate),
            learning_type(learning_type)
        {
            layers.reserve(layer_nb);

            switch (error_type)
            {
            default:
                error = math::func::simple;
                break;
            }
        }

        void add_layer(
            const int nb_params,
            const int layer_size,
            const Layer::Activation activation_type
        )
        {
            if (layers.size() >= layer_nb) throw std::invalid_argument("Declared layer number too small");

            layers.emplace_back(nb_params, layer_size, activation_type);
        }

        Layer get(const int i)
        {
            return layers[i];
        }

        Layer operator[](const int i)
        {
            return get(i);
        }

        std::vector<double> learn(const Dataset& dataset, std::size_t max_epochs, std::size_t batch_size = 1,
                                  bool shuffle = false);
        [[nodiscard]] math::Vector output(const std::vector<double>& input) const;
        [[nodiscard]] double mean_loss(const Dataset& dataset) const;
    };
}
