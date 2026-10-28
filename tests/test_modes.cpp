#include "test_support.h"
#include "network.h"
#include "compute_device.h"

using namespace ml;

namespace
{
    NeuralNetwork make_network(NeuralNetwork::LearningType mode)
    {
        NeuralNetwork net(2, 0.3, mode, NeuralNetwork::SIMPLE, 42);
        net.add_layer(2, 3);
        net.add_layer(3, 1);
        return net;
    }

    void equal_weights(const NeuralNetwork& a, const NeuralNetwork& b)
    {
        for (std::size_t l = 0; l < 2; ++l)
        {
            const auto& wa = a[l].get_weights();
            const auto& wb = b[l].get_weights();
            require(wa.get_x() == wb.get_x() && wa.get_y() == wb.get_y(), "different shapes");
            for (int i = 0; i < wa.get_x(); ++i)for (int j = 0; j < wa.get_y(); ++j)near(wa(i, j), wb(i, j), 1e-13);
        }
    }

    const std::vector<Sample> samples{
        {{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}, {{0.2, 0.7}, {0.8}}
    };
}

int main()
{
    initialize_compute(Backend::CPU);
    return run_tests({
        {
            "online_equals_minibatch_one", []
            {
                auto a = make_network(NeuralNetwork::ONLINE), b = make_network(NeuralNetwork::MINI_BATCH);
                const Dataset data(samples);
                const double before = a.mean_loss(data);
                a.learn(data, 30, 0, false);
                b.learn(data, 30, 1, false);
                equal_weights(a, b);
                require(a.mean_loss(data) < before, "both modes failed to learn");
            }
        },
        {
            "batch_equals_minibatch_N_and_auto", []
            {
                auto a = make_network(NeuralNetwork::BATCH), b = make_network(NeuralNetwork::MINI_BATCH);
                auto c = make_network(NeuralNetwork::AUTO);
                const Dataset data(samples);
                const double before = a.mean_loss(data);
                a.learn(data, 30);
                b.learn(data, 30, data.size());
                c.learn(data, 30);
                equal_weights(a, b);
                equal_weights(a, c);
                require(a.mean_loss(data) < before, "batch failed to learn");
            }
        },
        {
            "last_partial_batch_uses_actual_size", []
            {
                auto a = make_network(NeuralNetwork::MINI_BATCH), b = make_network(NeuralNetwork::BATCH);
                a.learn(Dataset(samples), 1, 2);
                for (std::size_t begin = 0; begin < samples.size(); begin += 2)
                {
                    const auto end = std::min(begin + 2, samples.size());
                    const std::vector<Sample> group(samples.begin() + begin, samples.begin() + end);
                    b.learn(Dataset(group), 1);
                }
                equal_weights(a, b);
            }
        },
        {
            "seed_and_shuffle_are_reproducible", []
            {
                auto a = make_network(NeuralNetwork::MINI_BATCH), b = make_network(NeuralNetwork::MINI_BATCH);
                a.learn(Dataset(samples), 20, 2, true);
                b.learn(Dataset(samples), 20, 2, true);
                equal_weights(a, b);
            }
        },
        {
            "invalid_minibatch_size_rejected", []
            {
                auto net = make_network(NeuralNetwork::MINI_BATCH);
                expect_throw<std::invalid_argument>([&] { net.learn(Dataset(samples), 1, 0); });
            }
        },
        {
            "xor_learns_in_all_three_modes", []
            {
                const Dataset data(std::vector<Sample>{{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}});
                for (auto mode : {NeuralNetwork::ONLINE, NeuralNetwork::BATCH, NeuralNetwork::MINI_BATCH})
                {
                    NeuralNetwork net(2, 0.8, mode, NeuralNetwork::SIMPLE, 42);
                    net.add_layer(2, 4);
                    net.add_layer(4, 1);
                    // Initialisation explicite identique sur les bibliotheques standard.
                    math::Matrix w1(4, 3), w2(1, 5);
                    const double hidden[4][2]{{0.3, 0.7}, {-0.4, 0.2}, {0.5, -0.6}, {-0.2, -0.8}};
                    const double output[4]{0.4, -0.3, 0.6, -0.5};
                    for (int i = 0; i < 4; ++i)
                    {
                        w1(i, 1) = hidden[i][0];
                        w1(i, 2) = hidden[i][1];
                        w2(0, i + 1) = output[i];
                    }
                    net[0].set_weights(w1);
                    net[1].set_weights(w2);
                    const double before = net.mean_loss(data);
                    const auto history = net.learn(data, 20000, 2, false);
                    require(history.back() < 0.005 && history.back() < before, "XOR loss did not decrease enough");
                    for (const auto& sample : data)
                    {
                        const double p = net.output(sample.input)[0];
                        require(sample.target[0] == 1 ? p > 0.9 : p < 0.1, "XOR prediction incorrect");
                    }
                    std::cout << "XOR mode " << mode << " loss " << before << " -> " << history.back() << '\n';
                }
            }
        }
    });
}
