#include <stdexcept>
#include "network.h"
#include "compute_device.h"
#include <exception>
#include <iostream>

int main()
{
    try
    {
        ml::initialize_compute(ml::Backend::CPU);
        ml::NeuralNetwork net(2, 0.1, ml::NeuralNetwork::MINI_BATCH);
        net.add_layer(2, 3, ml::Layer::SIGMOID);
        net.add_layer(3, 1, ml::Layer::SIGMOID);
        std::cout << "Sortie initiale : " << net.output({0.2, 0.7})[0] << '\n';
        std::cout << "Ce programme verifie seulement le forward, pas l'apprentissage.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
