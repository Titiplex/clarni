#include "test_support.h"
#include "network.h"
#include "compute_device.h"

using namespace ml;
using ml::math::Vector;

int main()
{
    initialize_compute(Backend::CPU);
    return run_tests({
        {
            "backward_requires_forward", []
            {
                Layer layer(1, 1, Layer::SIGMOID);
                expect_throw<std::logic_error>([&] { (void)layer.backward(Vector(1, 1)); });
            }
        },
        {
            "rectangular_layer_backward", []
            {
                Layer layer(2, 3, Layer::SIGMOID);
                (void)layer.forward(Vector(std::vector<double>{1, 2}));
                const auto dx = layer.backward(Vector(3, 1));
                require(dx.get_dim() == 2, "gradient d'entree de mauvaise taille");
            }
        },
        {
            "positive_gradient_updates_weight", []
            {
                Layer layer(1, 1, Layer::SIGMOID);
                const Vector x(1, 1);
                const double before = layer.forward(x)[0];
                (void)layer.backward(Vector(1, 1));
                layer.step(0.4, 1);
                require(layer.output(x)[0] < before, "poids non mis a jour par le gradient");
            }
        },
        {
            "bias_learns_on_zero_input", []
            {
                Layer layer(1, 1, Layer::SIGMOID);
                const Vector x(1, 0);
                const double before = layer.forward(x)[0];
                (void)layer.backward(Vector(1, 1));
                layer.step(0.4, 1);
                require(layer.output(x)[0] < before, "aucun biais appris sur entree nulle");
            }
        },
        {
            "scalar_network_learns", []
            {
                NeuralNetwork net(1, 0.5, NeuralNetwork::ONLINE);
                net.add_layer(1, 1, Layer::SIGMOID);
                const Dataset data(std::vector<Sample>{{{1.0}, {1.0}}});
                const double before = net.mean_loss(data);
                const auto history = net.learn(data, 40);
                require(history.size() == 40, "historique incomplet");
                require(net.mean_loss(data) < before, "perte inchangee apres 40 epoques");
            }
        },
        {
            "mean_loss_matches_half_squared_error", []
            {
                NeuralNetwork net(1, 0.1);
                net.add_layer(1, 1, Layer::SIGMOID);
                const Dataset data(std::vector<Sample>{{{1.0}, {1.0}}});
                const double e = net.output({1.0})[0] - 1.0;
                near(net.mean_loss(data), 0.5 * e * e);
            }
        },
        {
            "empty_network_output_rejected", []
            {
                NeuralNetwork net(0, 0.1);
                expect_throw<std::logic_error>([&] { (void)net.output({1.0}); });
            }
        },
        {
            "incompatible_layers_rejected_on_add", []
            {
                NeuralNetwork net(2, 0.1);
                net.add_layer(2, 3, Layer::SIGMOID);
                expect_throw<std::invalid_argument>([&] { net.add_layer(4, 1, Layer::SIGMOID); });
            }
        },
        {
            "failed_forward_invalidates_cache", []
            {
                Layer layer(1, 1, Layer::SIGMOID);
                (void)layer.forward(Vector(1, 1));
                expect_throw<std::invalid_argument>([&] { (void)layer.forward(Vector(2, 1)); });
                expect_throw<std::logic_error>([&] { (void)layer.backward(Vector(1, 1)); });
            }
        }
    });
}
