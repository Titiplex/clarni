#include "layer.h"
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace
{
    int checked_inputs(int size)
    {
        if (size <= 0 || size == (std::numeric_limits<int>::max)())
            throw std::invalid_argument("Layer: invalid input dimension");
        return size;
    }

    int checked_neurons(int size)
    {
        if (size <= 0) throw std::invalid_argument("Layer: invalid output dimension");
        return size;
    }
}

namespace ml
{
    Layer::Layer(int inputs, int neurons, Activation type)
        : nb_params(checked_inputs(inputs)), layer_size(checked_neurons(neurons)),
          activation_type(type),
          weights(layer_size, nb_params + 1), cached_output(layer_size),
          cached_input(nb_params + 1), gradients(layer_size, nb_params + 1)
    {
        switch (type)
        {
        case IDENTITY:
            activation = math::func::identity;
            derivative = math::func::identity_derivative;
            break;
        case SIGMOID:
            activation = math::func::sigmoid;
            derivative = math::func::sigmoid_derivative;
            break;
        case TANH:
            activation = math::func::tanh;
            derivative = math::func::tanh_derivative;
            break;
        case RELU:
            activation = math::func::relu;
            derivative = math::func::relu_derivative;
            break;
        case LEAKY_RELU:
            activation = math::func::leaky_relu;
            derivative = math::func::leaky_relu_derivative;
            break;
        default: throw std::invalid_argument("Unknown activation");
        }
    }

    void Layer::init_random()
    {
        std::mt19937 generator(std::random_device{}());
        init_random(generator);
    }

    void Layer::init_random(std::mt19937& generator)
    {
        const double variance = activation_type == RELU
                                    ? 2.0 / nb_params
                                    : activation_type == LEAKY_RELU
                                    ? 2.0 / ((1.0 + 0.01 * 0.01) * nb_params)
                                    : 2.0 / (static_cast<double>(nb_params) + layer_size);
        const double limit = std::sqrt(3.0 * variance);
        std::uniform_real_distribution<double> distribution(-limit, limit);
        for (int i = 0; i < layer_size; ++i)
        {
            weights(i, 0) = 0.0;
            for (int j = 1; j <= nb_params; ++j) weights(i, j) = distribution(generator);
        }
        zero_grad();
        cache_valid = false;
    }

    math::Vector Layer::output(const math::Vector& input) const
    {
        if (input.get_dim() != nb_params)
            throw std::invalid_argument("Layer: incorrect input dimension");
        math::Vector augmented(nb_params + 1);
        augmented[0] = 1.0;
        for (int j = 0; j < nb_params; ++j)
        {
            if (!std::isfinite(input[j])) throw std::invalid_argument("Layer: non-finite input");
            augmented[j + 1] = input[j];
        }
        return activation(weights * augmented);
    }

    math::Vector Layer::forward(const math::Vector& input)
    {
        cache_valid = false;
        cached_output = output(input);
        cached_input[0] = 1.0;
        for (int j = 0; j < nb_params; ++j) cached_input[j + 1] = input[j];
        cache_valid = true;
        return cached_output;
    }

    math::Vector Layer::backward(const math::Vector& grad_output)
    {
        if (!cache_valid) throw std::logic_error("forward required before backward");
        if (grad_output.get_dim() != layer_size)
            throw std::invalid_argument("Incorrect gradient size");
        for (int i = 0; i < layer_size; ++i)
            if (!std::isfinite(grad_output[i])) throw std::invalid_argument("Non-finite upstream gradient");
        cache_valid = false;
        const auto f_prime = derivative(cached_output);
        math::Vector grad_input(nb_params, 0.0);
        for (int i = 0; i < layer_size; ++i)
        {
            const double delta = grad_output[i] * f_prime[i];
            for (int j = 0; j <= nb_params; ++j) gradients(i, j) += delta * cached_input[j];
            for (int j = 0; j < nb_params; ++j) grad_input[j] += weights(i, j + 1) * delta;
        }
        return grad_input;
    }

    void Layer::zero_grad()
    {
        for (int i = 0; i < layer_size; ++i)
            for (int j = 0; j <= nb_params; ++j) gradients(i, j) = 0.0;
    }

    void Layer::step(double learning_rate, std::size_t sample_count)
    {
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0 || sample_count == 0)
            throw std::invalid_argument("Invalid step parameters");
        const double scale = learning_rate / static_cast<double>(sample_count);
        math::Matrix next = weights - gradients * scale;
        for (int i = 0; i < layer_size; ++i)
            for (int j = 0; j <= nb_params; ++j)
                if (!std::isfinite(next(i, j))) throw std::overflow_error("Non-finite weights after update");
        weights = std::move(next);
        zero_grad();
        cache_valid = false;
    }

    void Layer::set_weights(const math::Matrix& values)
    {
        if (values.get_x() != layer_size || values.get_y() != nb_params + 1)
            throw std::invalid_argument("Layer: incorrect weight dimensions");
        for (int i = 0; i < layer_size; ++i)
            for (int j = 0; j <= nb_params; ++j)
                if (!std::isfinite(values(i, j))) throw std::invalid_argument("Layer: non-finite weight");
        weights = values;
        zero_grad();
        cache_valid = false;
    }
}
