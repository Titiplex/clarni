#pragma once

#include <functional>

#include "matrix.h"
#include "func.h"

namespace ml
{
    class Layer
    {
        math::Matrix weights;
        int nb_params, layer_size;
        std::function<math::Vector(math::Vector)> activation;
        std::function<math::Vector(math::Vector)> derivative;
        bool cache_valid = false;
        math::Vector cached_output;
        math::Vector cached_input;
        math::Matrix gradients;

    public:
        enum Activation { SIGMOID };

        Layer(
            const int nb_params,
            const int layer_size,
            const Activation activation_type
        ) :
            weights(math::Matrix(layer_size, nb_params)),
            nb_params(nb_params),
            layer_size(layer_size),
            cached_output(layer_size),
            cached_input(layer_size),
            gradients(math::Matrix(layer_size, nb_params))
        {
            switch (activation_type)
            {
            default:
                activation = math::func::sigmoid;
                derivative = math::func::sigmoid_derivative;
                break;
            }
        }

        ~Layer() = default;

        /**
         * Initialize the weights with a uniform real distribution between -1 and 1
         */
        void init_random();

        /**
         * Outputs the result for the Layer, with current weights.
         * @param input entry Vector
         * @return Output Vector
         */
        [[nodiscard]] math::Vector output(const math::Vector& input) const;

        // Entraînement : mémorise les données nécessaires au backward.
        math::Vector forward(const math::Vector& input);

        // Accumule les gradients et renvoie le gradient des entrées.
        math::Vector backward(const math::Vector& grad_output);

        void zero_grad();

        // Applique la moyenne des gradients accumulés.
        void step(double learning_rate, std::size_t sample_count);

        [[nodiscard]] int input_size() const { return nb_params; }
        [[nodiscard]] int output_size() const { return layer_size; }
    };
}
