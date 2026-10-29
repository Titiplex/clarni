#include "test_support.h"
#include "network.h"
#include "compute_device.h"
#include <limits>

using namespace ml;
using namespace ml::math;

int main()
{
    initialize_compute(Backend::CPU);
    return run_tests({
        {
            "all_weights_and_biases_finite_differences", []
            {
                NeuralNetwork net(2, 0.1, NeuralNetwork::BATCH, NeuralNetwork::SIMPLE, 19);
                net.add_layer(2, 3);
                net.add_layer(3, 2);
                const std::vector<double> x{0.3, -0.7}, target{0.2, 0.8};
                const Dataset data(std::vector<Sample>{{x, target}});
                auto prediction = net[1].forward(net[0].forward(Vector(x)));
                auto dx = net[0].backward(net[1].backward(func::simple(prediction, Vector(target))));
                const Matrix gradients[]{net[0].get_gradients(), net[1].get_gradients()};
                constexpr double epsilon = 1e-6;
                double maximum_error = 0.0;
                int checked = 0;
                for (std::size_t layer = 0; layer < 2; ++layer)
                {
                    const Matrix original = net[layer].get_weights();
                    for (int i = 0; i < original.get_x(); ++i)
                    {
                        for (int j = 0; j < original.get_y(); ++j)
                        {
                            Matrix w = original;
                            w(i, j) += epsilon;
                            net[layer].set_weights(w);
                            const double plus = net.mean_loss(data);
                            w(i, j) -= 2 * epsilon;
                            net[layer].set_weights(w);
                            const double minus = net.mean_loss(data);
                            const double numerical = (plus - minus) / (2 * epsilon);
                            const double analytic = gradients[layer](i, j);
                            near(analytic, numerical, 1e-7);
                            maximum_error = std::max(maximum_error, std::abs(analytic - numerical));
                            ++checked;
                            net[layer].set_weights(original);
                        }
                    }
                }
                for (std::size_t j = 0; j < x.size(); ++j)
                {
                    auto perturbed = x;
                    perturbed[j] += epsilon;
                    const double plus = net.mean_loss(Dataset(std::vector<Sample>{{perturbed, target}}));
                    perturbed[j] -= 2 * epsilon;
                    const double minus = net.mean_loss(Dataset(std::vector<Sample>{{perturbed, target}}));
                    near(dx[j], (plus - minus) / (2 * epsilon), 1e-7);
                }
                std::cout << "Checked " << checked << " weights/biases; max absolute error "
                    << maximum_error << '\n';
            }
        },
        {
            "zero_upstream_and_gradient_accumulation", []
            {
                Layer layer(2, 2);
                Matrix w(2, 3, 0.2);
                layer.set_weights(w);
                const Vector x(std::vector<double>{0.4, -0.3});
                (void)layer.forward(x);
                const auto zero = layer.backward(Vector(2, 0));
                near(zero[0], 0);
                near(zero[1], 0);
                for (int i = 0; i < 2; ++i)for (int j = 0; j < 3; ++j)near(layer.get_gradients()(i, j), 0);
                (void)layer.forward(x);
                (void)layer.backward(Vector(2, 1));
                const Matrix once = layer.get_gradients();
                (void)layer.forward(x);
                (void)layer.backward(Vector(2, 1));
                for (int i = 0; i < 2; ++i)for (int j = 0; j < 3; ++j)near(layer.get_gradients()(i, j), 2 * once(i, j));
                layer.step(0.1, 2);
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 3; ++j)
                        near(
                            layer.get_weights()(i, j), w(i, j) - 0.1 * once(i, j));
            }
        },
        {
            "cache_invalidated_by_weight_change", []
            {
                Layer layer(1, 1);
                (void)layer.forward(Vector(1, 1));
                layer.set_weights(Matrix(1, 2, 0.2));
                expect_throw<std::logic_error>([&] { (void)layer.backward(Vector(1, 1)); });
                (void)layer.forward(Vector(1, 1));
                std::mt19937 generator(1);
                layer.init_random(generator);
                expect_throw<std::logic_error>([&] { (void)layer.backward(Vector(1, 1)); });
            }
        },
        {
            "validated_public_api", []
            {
                expect_throw<std::invalid_argument>([] { Layer bad(0, 1); });
                expect_throw<std::invalid_argument>([] { Layer bad(1, 0); });
                expect_throw<std::invalid_argument>([] { NeuralNetwork bad(1, 0); });
                expect_throw<std::invalid_argument>([] { NeuralNetwork bad(1, -0.1); });
                NeuralNetwork net(1, 0.1);
                net.add_layer(1, 1);
                auto& layer = net.get(0);
                layer.set_weights(Matrix(1, 2, 0.0));
                near(net.output({1})[0], 0.5);
                Matrix w(1, 2, 0);
                w(0, 0) = 1;
                layer.set_weights(w);
                near(net.output({0})[0], 1.0 / (1.0 + std::exp(-1.0)));
                const NeuralNetwork& read = net;
                near(read[0].get_weights()(0, 0), 1);
                expect_throw<std::out_of_range>([&] { (void)net.get(1); });
                expect_throw<std::invalid_argument>([&] { layer.set_weights(Matrix(1, 1)); });
                expect_throw<std::invalid_argument>(
                    [&] { (void)net.output({std::numeric_limits<double>::infinity()}); });
            }
        }
    });
}
