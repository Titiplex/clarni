#include "network.h"
#include <exception>
#include <iostream>

int main()
{
    try
    {
        ml::initialize_compute(ml::Backend::Auto);
        ml::Dataset data(std::vector<ml::Sample>{{{-2, -2}, {1, 0, 0}}, {{2, -2}, {0, 1, 0}}, {{0, 2}, {0, 0, 1}}});
        ml::NeuralNetwork net(1, 0.05, ml::NeuralNetwork::BATCH, ml::NeuralNetwork::CROSS_ENTROPY);
        net.add_layer(2, 3, ml::Layer::SOFTMAX);
        ml::OptimizerOptions options;
        options.kind = ml::OptimizerKind::Adam;
        net.set_optimizer(options);
        net.learn(data, 250);
        std::cout << "Cross entropy: " << net.mean_loss(data) << '\n';
        for (const auto& sample : data)
        {
            auto probabilities = net.output(sample.input);
            for (std::size_t i = 0; i < probabilities.size(); ++i)std::cout << probabilities[i] << ' ';
            std::cout << '\n';
        }
        return net.mean_loss(data) < 0.05 ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
