#include "layer.h"
#include <random>

using namespace std;
using namespace ml;

void Layer::init_random()
{
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution weight_dist(-1.0, 1.0);

    for (int i = 0; i < weights.get_x(); i++)
        for (int j = 0; j < weights.get_y(); j++)
            weights(i, j) = weight_dist(gen);
}

math::Vector Layer::output(const math::Vector& input) const
{
    return this->activation(math::Vector::convert(this->weights * input));
}

math::Vector Layer::forward(const math::Vector& input)
{
    cached_input = input;
    cached_output = output(input);
    cache_valid = true;
    return cached_output;
}

math::Vector Layer::backward(const math::Vector& grad_output)
{
    if (!cache_valid)
        throw std::logic_error("forward required before backward");
    if (grad_output.get_dim() != layer_size)
        throw std::invalid_argument("Incorrect gradient size");

    math::Vector grad_input(nb_params, 0.0);

    const auto f_prime = derivative(cached_output);
    const auto delta = cached_output * f_prime;

    gradients += delta * grad_input;
    grad_input = weights * delta;

    cache_valid = false;
    return grad_input;
}

/**
 * Puts gradients back to 0.0
 */
void Layer::zero_grad()
{
    gradients = math::Matrix(gradients.get_x(), gradients.get_y(), 0.0);
}

void Layer::step(const double learning_rate, const std::size_t sample_count)
{
    if (!std::isfinite(learning_rate)
        || learning_rate <= 0.0
        || sample_count == 0)
        throw std::invalid_argument("Invalid step parameters");

    const double scale =
        learning_rate / static_cast<double>(sample_count);

    weights -= gradients * scale;
    zero_grad();
    cache_valid = false;
}
