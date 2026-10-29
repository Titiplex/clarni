#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include "matrix.h"
#include "func.h"
#include "tensor.h"
#include "optimizer.h"

namespace ml
{
    class Layer
    {
    public:
        enum Activation { IDENTITY, SIGMOID, TANH, RELU, LEAKY_RELU, SOFTMAX };

    private:
        int nb_params_, layer_size_;
        Activation type_;
        mutable math::Matrix weights_host_, gradients_host_;
        mutable Tensor weights_;
        Tensor gradients_, cached_input_, cached_z_, cached_output_;
        mutable bool tensor_valid_ = false, host_dirty_ = false, gradient_dirty_ = false;
        bool cache_valid_ = false;
        std::uint64_t revision_ = 0;
        void invalidated();

    public:
        Layer(int nb_params, int layer_size, Activation activation_type = SIGMOID);
        void init_random();
        void init_random(std::mt19937& generator);
        math::Vector output(const math::Vector& input) const;
        math::Vector forward(const math::Vector& input);
        math::Vector backward(const math::Vector& gradient);
        Tensor preactivation(const Tensor& input) const;
        Tensor output_batch(const Tensor& input) const;
        Tensor forward_batch(const Tensor& input);
        Tensor backward_batch(const Tensor& gradient, bool already_logits = false);
        const Tensor& cached_logits() const;
        const Tensor& tensor_weights() const;
        const Tensor& tensor_gradients() const { return gradients_; }
        void apply(Optimizer& optimizer, std::size_t key, double rate, std::size_t count, double l2 = 0);
        void zero_grad();
        void step(double learning_rate, std::size_t sample_count);
        int input_size() const { return nb_params_; }
        int output_size() const { return layer_size_; }
        Activation activation_type() const { return type_; }
        std::uint64_t revision() const { return revision_; }
        const math::Matrix& get_weights() const;
        const math::Matrix& get_gradients() const;
        void set_weights(const math::Matrix& values);
    };
}
