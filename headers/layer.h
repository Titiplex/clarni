#pragma once
#include <cstddef>
#include <functional>
#include <random>
#include "matrix.h"
#include "func.h"

namespace ml
{
    class Layer
    {
    public:
        enum Activation { IDENTITY, SIGMOID, TANH, RELU, LEAKY_RELU };

    private:
        int nb_params, layer_size;
        math::Matrix weights; // M x (N+1), biais en colonne 0.
        std::function<math::Vector(const math::Vector&)> activation;
        std::function<math::Vector(const math::Vector&)> derivative;
        bool cache_valid = false;
        math::Vector cached_output, cached_input;
        math::Matrix gradients;

    public:
        // Couche seule : poids a zero jusqu'a init_random/set_weights.
        // NeuralNetwork::add_layer initialise automatiquement les poids.
        Layer(int nb_params, int layer_size, Activation activation_type = SIGMOID);
        void init_random();
        void init_random(std::mt19937& generator);
        [[nodiscard]] math::Vector output(const math::Vector& input) const;
        math::Vector forward(const math::Vector& input);
        math::Vector backward(const math::Vector& grad_output);
        void zero_grad();
        void step(double learning_rate, std::size_t sample_count);
        [[nodiscard]] int input_size() const { return nb_params; }
        [[nodiscard]] int output_size() const { return layer_size; }
        [[nodiscard]] const math::Matrix& get_weights() const { return weights; }
        [[nodiscard]] const math::Matrix& get_gradients() const { return gradients; }
        void set_weights(const math::Matrix& values);
    };
}
