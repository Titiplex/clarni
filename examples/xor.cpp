#include "compute_device.h"
#include "network.h"
#include <exception>
#include <iomanip>
#include <iostream>

int main()
{
    try
    {
        ml::initialize_compute(ml::Backend::CPU);
        ml::NeuralNetwork net(2, 0.8, ml::NeuralNetwork::MINI_BATCH, ml::NeuralNetwork::SIMPLE, 42);
        net.add_layer(2, 4);
        net.add_layer(4, 1);
        const ml::Dataset data(std::vector<ml::Sample>{
            {{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}
        });
        const double before = net.mean_loss(data);
        const auto history = net.learn(data, 20000, 2, true);
        std::cout << std::fixed << std::setprecision(6)
            << "Loss : " << before << " -> " << history.back() << '\n';
        for (const auto& sample : data)
            std::cout << sample.input[0] << " XOR " << sample.input[1]
                << " = " << net.output(sample.input)[0] << '\n';
        return history.back() < 0.005 ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
