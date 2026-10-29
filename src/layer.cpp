#include "layer.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
    int inputs_checked(const int n)
    {
        if (n <= 0 || n == (std::numeric_limits<int>::max)())throw std::invalid_argument("Layer input dimension");
        return n;
    }

    int outputs_checked(const int n)
    {
        if (n <= 0)throw std::invalid_argument("Layer output dimension");
        return n;
    }

    ml::Tensor column(const ml::math::Vector& v, const ml::Backend backend)
    {
        ml::math::Matrix result(v.get_dim(), 1);
        for (int i = 0; i < v.get_dim(); ++i)result(i, 0) = v[i];
        return ml::Tensor::from_matrix(result, backend);
    }
}

namespace ml
{
    Layer::Layer(const int nb_params, const int layer_size, const Activation activation_type)
        : nb_params_(inputs_checked(nb_params)), layer_size_(outputs_checked(layer_size)), type_(activation_type),
          weights_host_(layer_size_, nb_params_ + 1), gradients_host_(layer_size_, nb_params_ + 1)
    {
        if (activation_type < IDENTITY || activation_type > SOFTMAX)throw std::invalid_argument("Unknown activation");
        zero_grad();
    }

    void Layer::invalidated()
    {
        tensor_valid_ = false;
        host_dirty_ = false;
        cache_valid_ = false;
        ++revision_;
        zero_grad();
    }

    const Tensor& Layer::tensor_weights() const
    {
        const auto backend = compute_device().selected;
        if (!tensor_valid_)
        {
            weights_ = Tensor::from_matrix(weights_host_, backend);
            tensor_valid_ = true;
        }
        else if (weights_.backend() != backend)weights_ = weights_.to(backend);
        return weights_;
    }

    const math::Matrix& Layer::get_weights() const
    {
        if (host_dirty_)
        {
            weights_host_ = weights_.to_matrix();
            host_dirty_ = false;
        }
        return weights_host_;
    }

    const math::Matrix& Layer::get_gradients() const
    {
        if (gradient_dirty_)
        {
            gradients_host_ = gradients_.to_matrix();
            gradient_dirty_ = false;
        }
        return gradients_host_;
    }

    void Layer::init_random()
    {
        std::mt19937 rng(std::random_device{}());
        init_random(rng);
    }

    void Layer::init_random(std::mt19937& generator)
    {
        const double variance = type_ == RELU
                                    ? 2.0 / nb_params_
                                    : type_ == LEAKY_RELU
                                    ? 2.0 / ((1 + 0.01 * 0.01) * nb_params_)
                                    : 2.0 / (nb_params_ + static_cast<double>(layer_size_));
        std::uniform_real_distribution distribution(-std::sqrt(3 * variance), std::sqrt(3 * variance));
        for (int i = 0; i < layer_size_; ++i)
        {
            weights_host_(i, 0) = 0;
            for (int j = 1; j <= nb_params_; ++j)weights_host_(i, j) = distribution(generator);
        }
        invalidated();
    }

    void Layer::set_weights(const math::Matrix& values)
    {
        if (values.get_x() != layer_size_ || values.get_y() != nb_params_ + 1)
            throw std::invalid_argument(
                "Weight dimensions");
        for (int i = 0; i < values.get_x(); ++i)
            for (int j = 0; j < values.get_y(); ++j)
                if (!
                    std::isfinite(values(i, j)))
                    throw std::invalid_argument("Non-finite weight");
        weights_host_ = values;
        invalidated();
    }

    Tensor Layer::preactivation(const Tensor& input) const
    {
        if (input.rows() != nb_params_ || input.cols() <= 0 || !input.all_finite())
            throw std::invalid_argument(
                "Layer batch input");
        return tensor_weights().matmul(input.with_bias_row());
    }

    Tensor Layer::output_batch(const Tensor& input) const
    {
        return preactivation(input).activate(static_cast<ml::Activation>(type_));
    }

    Tensor Layer::forward_batch(const Tensor& input)
    {
        cache_valid_ = false;
        cached_z_ = preactivation(input);
        cached_input_ = input.with_bias_row();
        cached_output_ = cached_z_.activate(static_cast<ml::Activation>(type_));
        cache_valid_ = true;
        return cached_output_;
    }

    const Tensor& Layer::cached_logits() const
    {
        if (!cache_valid_)throw std::logic_error("No valid forward cache");
        return cached_z_;
    }

    Tensor Layer::backward_batch(const Tensor& gradient, const bool already_logits)
    {
        if (!cache_valid_)throw std::logic_error("forward required before backward");
        if (gradient.rows() != layer_size_ || gradient.cols() != cached_output_.cols() || !gradient.all_finite())
            throw
                std::invalid_argument("Layer gradient shape/values");
        cache_valid_ = false;
        const auto delta = already_logits
                               ? gradient
                               : cached_z_.activation_backward(cached_output_, gradient,
                                                               static_cast<ml::Activation>(type_));
        if (gradients_.backend() != delta.backend())gradients_ = gradients_.to(delta.backend());
        gradients_ = gradients_ + delta.matmul(cached_input_.transpose());
        gradient_dirty_ = true;
        return tensor_weights().transpose().matmul(delta).slice_rows(1, nb_params_);
    }

    math::Vector Layer::output(const math::Vector& input) const
    {
        return math::Vector::convert(output_batch(column(input, compute_device().selected)).to_matrix());
    }

    math::Vector Layer::forward(const math::Vector& input)
    {
        cache_valid_ = false;
        return math::Vector::convert(forward_batch(column(input, compute_device().selected)).to_matrix());
    }

    math::Vector Layer::backward(const math::Vector& gradient)
    {
        return math::Vector::convert(backward_batch(column(gradient, compute_device().selected)).to_matrix());
    }

    void Layer::zero_grad()
    {
        gradients_ = Tensor(layer_size_, nb_params_ + 1, 0, compute_device().selected);
        gradient_dirty_ = true;
    }

    void Layer::apply(Optimizer& optimizer, const std::size_t key, const double rate, const std::size_t count,
                      const double l2)
    {
        if (count == 0 || !std::isfinite(l2) || l2 < 0)throw std::invalid_argument("Invalid update parameters");
        const auto& w = tensor_weights();
        auto gradient = gradients_.to(w.backend()).affine(1.0 / static_cast<double>(count));
        if (l2)gradient = gradient + w.zero_first_column().affine(l2);
        weights_ = optimizer.update(key, w, gradient, rate);
        tensor_valid_ = true;
        host_dirty_ = true;
        cache_valid_ = false;
        zero_grad();
    }

    void Layer::step(const double learning_rate, const std::size_t sample_count)
    {
        Optimizer sgd;
        apply(sgd, 0, learning_rate, sample_count);
        ++revision_;
    }
}
