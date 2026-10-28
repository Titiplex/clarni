#include "network.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ml
{
    NeuralNetwork::NeuralNetwork(std::size_t maximum_layers, double rate,
                                 LearningType type, Error loss, std::uint32_t seed)
        : layer_nb(maximum_layers), learning_rate(rate), learning_type(type), generator(seed)
    {
        if (!std::isfinite(rate) || rate <= 0.0) throw std::invalid_argument("Invalid learning rate");
        if (type != BATCH && type != ONLINE && type != MINI_BATCH && type != AUTO)
            throw std::invalid_argument("Unknown learning type");
        switch (loss)
        {
        case SIMPLE: error = math::func::simple;
            loss_value = math::func::simple_loss;
            break;
        default: throw std::invalid_argument("Unknown loss");
        }
        layers.reserve(layer_nb);
    }

    void NeuralNetwork::add_layer(int inputs, int neurons, Layer::Activation type)
    {
        if (layers.size() >= layer_nb) throw std::invalid_argument("Declared layer number too small");
        if (!layers.empty() && layers.back().output_size() != inputs)
            throw std::invalid_argument("Incompatible adjacent layers");
        Layer layer(inputs, neurons, type);
        layer.init_random(generator);
        layers.push_back(std::move(layer));
    }

    Layer& NeuralNetwork::get(std::size_t i) { return layers.at(i); }
    const Layer& NeuralNetwork::get(std::size_t i) const { return layers.at(i); }

    void NeuralNetwork::validate_topology() const
    {
        if (layers.empty()) throw std::logic_error("Network has no layers");
        for (std::size_t i = 1; i < layers.size(); ++i)
            if (layers[i - 1].output_size() != layers[i].input_size())
                throw std::invalid_argument("Incompatible adjacent layers");
    }

    void NeuralNetwork::validate_dataset(const Dataset& dataset) const
    {
        validate_topology();
        if (dataset.empty()) throw std::invalid_argument("Empty dataset");
        for (const auto& [input,target] : dataset)
        {
            if (input.size() != static_cast<std::size_t>(layers.front().input_size())
                || target.size() != static_cast<std::size_t>(layers.back().output_size()))
                throw std::invalid_argument("Incorrect dataset dimensions");
            for (double x : input) if (!std::isfinite(x)) throw std::invalid_argument("Non-finite input");
            for (double y : target) if (!std::isfinite(y)) throw std::invalid_argument("Non-finite target");
        }
    }

    std::vector<double> NeuralNetwork::learn(const Dataset& dataset, std::size_t max_epochs,
                                             std::size_t batch_size, bool shuffle)
    {
        validate_dataset(dataset);
        switch (learning_type)
        {
        case ONLINE: batch_size = 1;
            break;
        case BATCH: batch_size = dataset.size();
            break;
        case AUTO:
            if (batch_size == 0) batch_size = 32;
            batch_size = std::min(batch_size, dataset.size());
            break;
        case MINI_BATCH:
            if (batch_size == 0) throw std::invalid_argument("Mini-batch size must be positive");
            batch_size = std::min(batch_size, dataset.size());
            break;
        }
        std::vector<std::size_t> order(dataset.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::vector<double> losses;
        losses.reserve(max_epochs);
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch)
        {
            if (shuffle) std::shuffle(order.begin(), order.end(), generator);
            for (std::size_t begin = 0; begin < dataset.size();)
            {
                const auto count = std::min(batch_size, dataset.size() - begin);
                const auto end = begin + count;
                for (auto& layer : layers) layer.zero_grad();
                for (std::size_t p = begin; p < end; ++p)
                {
                    const Sample& sample = dataset[order[p]];
                    math::Vector prediction(sample.input);
                    for (auto& layer : layers) prediction = layer.forward(prediction);
                    math::Vector gradient = error(prediction, math::Vector(sample.target));
                    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                        gradient = it->backward(gradient);
                }
                for (auto& layer : layers) layer.step(learning_rate, count);
                begin = end;
            }
            losses.push_back(mean_loss(dataset));
        }
        return losses;
    }

    math::Vector NeuralNetwork::output(const std::vector<double>& input) const
    {
        validate_topology();
        math::Vector result(input);
        for (const auto& layer : layers) result = layer.output(result);
        return result;
    }

    double NeuralNetwork::mean_loss(const Dataset& dataset) const
    {
        validate_dataset(dataset);
        double total = 0.0;
        for (const auto& sample : dataset)
            total += loss_value(output(sample.input), math::Vector(sample.target));
        if (!std::isfinite(total)) throw std::overflow_error("Non-finite loss");
        return total / static_cast<double>(dataset.size());
    }
}
