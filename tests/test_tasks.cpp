#include "test_support.h"
#include "network.h"
#include "compute_device.h"
using namespace ml;

int main()
{
    initialize_compute(Backend::CPU);
    return run_tests({
        {
            "affine_regression_recovers_weights_and_bias", []
            {
                std::vector<Sample> samples;
                for (double x : {-1., 0., 1.})for (double y : {-1., 0., 1.})samples.push_back({
                    {x, y}, {2 * x - 3 * y + 0.7}
                });
                NeuralNetwork net(1, 0.1, NeuralNetwork::BATCH);
                net.add_layer(2, 1, Layer::IDENTITY);
                net.learn(Dataset(samples), 500);
                near(net[0].get_weights()(0, 0), 0.7, 1e-5);
                near(net[0].get_weights()(0, 1), 2, 1e-5);
                near(net[0].get_weights()(0, 2), -3, 1e-5);
            }
        },
        {
            "binary_classifier_and_fused_gradient", []
            {
                Dataset data(std::vector<Sample>{{{-2}, {0}}, {{-1}, {0}}, {{1}, {1}}, {{2}, {1}}});
                NeuralNetwork net(1, 0.05, NeuralNetwork::BATCH, NeuralNetwork::BINARY_CROSS_ENTROPY);
                net.add_layer(1, 1, Layer::SIGMOID);
                OptimizerOptions options;
                options.kind = OptimizerKind::Adam;
                net.set_optimizer(options);
                net.learn(data, 300);
                require(net.mean_loss(data) < 0.05, "BCE did not learn");
                for (const auto& sample : data)require((net.output(sample.input)[0] > 0.5) == bool(sample.target[0]),
                                                       "binary class");
            }
        },
        {
            "three_class_softmax", []
            {
                Dataset data(std::vector<Sample>{
                    {{-2, -2}, {1, 0, 0}}, {{-1.8, -2}, {1, 0, 0}}, {{2, -2}, {0, 1, 0}}, {{2, -1.8}, {0, 1, 0}},
                    {{0, 2}, {0, 0, 1}}, {{0.2, 2}, {0, 0, 1}}
                });
                NeuralNetwork net(1, 0.1, NeuralNetwork::BATCH, NeuralNetwork::CROSS_ENTROPY);
                net.add_layer(2, 3, Layer::SOFTMAX);
                net.learn(data, 300);
                require(net.mean_loss(data) < 0.05, "CE did not learn");
                for (const auto& sample : data)
                {
                    auto p = net.output(sample.input);
                    double sum = 0;
                    int best = 0;
                    for (int i = 0; i < 3; ++i)
                    {
                        sum += p[i];
                        if (p[i] > p[best])best = i;
                    }
                    near(sum, 1);
                    near(sample.target[best], 1);
                }
            }
        },
        {
            "classification_parameter_gradient_check", []
            {
                for (auto error : {NeuralNetwork::BINARY_CROSS_ENTROPY, NeuralNetwork::CROSS_ENTROPY})
                {
                    int outputs = error == NeuralNetwork::CROSS_ENTROPY ? 3 : 1;
                    NeuralNetwork net(2, 0.1, NeuralNetwork::BATCH, error);
                    net.add_layer(2, 2, Layer::TANH);
                    net.add_layer(2, outputs, error == NeuralNetwork::CROSS_ENTROPY ? Layer::SOFTMAX : Layer::SIGMOID);
                    std::vector<double> x{0.3, -0.4}, y(outputs, 0);
                    y[0] = 1;
                    Dataset data(std::vector<Sample>{{x, y}});
                    math::Matrix m(2, 1);
                    m(0, 0) = x[0];
                    m(1, 0) = x[1];
                    auto h = net[0].forward_batch(Tensor::from_matrix(m));
                    (void)net[1].forward_batch(h);
                    math::Matrix target(outputs, 1);
                    target(0, 0) = 1;
                    auto loss = net[1].cached_logits().loss(Tensor::from_matrix(target),
                                                            error == NeuralNetwork::CROSS_ENTROPY
                                                                ? LossKind::CrossEntropy
                                                                : LossKind::BinaryCrossEntropy);
                    (void)net[0].backward_batch(net[1].backward_batch(loss.gradient, true));
                    const math::Matrix gradients[]{net[0].get_gradients(), net[1].get_gradients()};
                    for (int l = 0; l < 2; ++l)
                    {
                        auto original = net[l].get_weights();
                        for (int i = 0; i < original.get_x(); ++i)for (int j = 0; j < original.get_y(); ++j)
                        {
                            auto w = original;
                            w(i, j) += 1e-6;
                            net[l].set_weights(w);
                            double plus = net.mean_loss(data);
                            w(i, j) -= 2e-6;
                            net[l].set_weights(w);
                            double minus = net.mean_loss(data);
                            near(gradients[l](i, j), (plus - minus) / 2e-6, 1e-7);
                            net[l].set_weights(original);
                        }
                    }
                }
            }
        }
    });
}
