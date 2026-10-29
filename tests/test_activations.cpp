#include "test_support.h"
#include "layer.h"
#include "compute_device.h"

int main()
{
    ml::initialize_compute(ml::Backend::CPU);
    return run_tests({
        {
            "activation_gradients", []
            {
                for (auto type : {
                         ml::Layer::IDENTITY, ml::Layer::SIGMOID, ml::Layer::TANH,
                         ml::Layer::RELU, ml::Layer::LEAKY_RELU
                     })
                {
                    ml::Layer layer(2, 2, type);
                    ml::math::Matrix weights(2, 3);
                    weights(0, 0) = 0.3;
                    weights(0, 1) = 0.7;
                    weights(0, 2) = -0.2;
                    weights(1, 0) = -0.4;
                    weights(1, 1) = -0.3;
                    weights(1, 2) = 0.6;
                    layer.set_weights(weights);
                    ml::math::Vector x(std::vector{0.2, 0.5});
                    (void)layer.forward(x);
                    const auto gradient = layer.backward(ml::math::Vector(2, 1));
                    for (int j = 0; j < 2; ++j)
                    {
                        auto perturbed = x;
                        perturbed[j] += 1e-6;
                        const auto plus = layer.output(perturbed);
                        perturbed[j] -= 2e-6;
                        const auto minus = layer.output(perturbed);
                        near(gradient[j], ((plus[0] + plus[1]) - (minus[0] + minus[1])) / 2e-6, 1e-7);
                    }
                }
            }
        }
    });
}
