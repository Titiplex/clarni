#include "test_support.h"
#include "network.h"
#include "tensor_backend.h"
using namespace ml;

void same(const Tensor& a, const Tensor& b)
{
    auto x = a.to_matrix(), y = b.to_matrix();
    require(x.get_x() == y.get_x() && x.get_y() == y.get_y(), "shape mismatch");
    for (int i = 0; i < x.get_x(); ++i)for (int j = 0; j < x.get_y(); ++j)near(x(i, j), y(i, j), 1e-8);
}

int main()
{
    auto info = initialize_compute();
    if (!info.cuda_usable)
    {
        std::cout << "SKIP resident CUDA unavailable\n";
        return 77;
    }
#ifdef CLARNI_MOCK_GPU
    std::cout << "Simulation only: no CUDA kernel validation\n";
#endif
    return run_tests({
        {
            "resident_tensor_operations_match_cpu", []
            {
                math::Matrix m(3, 2);
                for (int i = 0; i < 3; ++i)for (int j = 0; j < 2; ++j)m(i, j) = 0.2 * (i - j) + 0.4;
                Tensor cpu = Tensor::from_matrix(m), gpu = cpu.to(Backend::CUDA);
                same(gpu.transpose(), cpu.transpose());
                same(gpu.matmul(gpu.transpose()), cpu.matmul(cpu.transpose()));
                same(gpu + gpu, cpu + cpu);
                same(gpu - gpu, cpu - cpu);
                same(gpu.hadamard(gpu), cpu.hadamard(cpu));
                same(gpu.divide(gpu.affine(1, 1)), cpu.divide(cpu.affine(1, 1)));
                same(gpu.affine(0.2, -0.3), cpu.affine(0.2, -0.3));
                same(gpu.with_bias_row().slice_rows(1, 3), cpu);
                same(gpu.sum_columns().broadcast_columns(2), cpu.sum_columns().broadcast_columns(2));
                same(gpu.sum_rows().broadcast_rows(3), cpu.sum_rows().broadcast_rows(3));
                same(gpu.zero_first_column(), cpu.zero_first_column());
                for (const auto operation : {Unary::Exp, Unary::Log, Unary::Square, Unary::Sqrt, Unary::Reciprocal})
                    same(
                        gpu.unary(operation), cpu.unary(operation));
                for (const auto activation : {
                         Activation::Identity, Activation::Sigmoid, Activation::Tanh, Activation::Relu,
                         Activation::LeakyRelu, Activation::Softmax
                     })
                {
                    auto ca = cpu.activate(activation), ga = gpu.activate(activation);
                    same(ga, ca);
                    same(gpu.activation_backward(ga, gpu, activation), cpu.activation_backward(ca, cpu, activation));
                }
                const Tensor target(3, 2, 1.0 / 3);
                for (const auto kind : {LossKind::Squared, LossKind::BinaryCrossEntropy, LossKind::CrossEntropy})
                {
                    auto c = cpu.loss(target, kind), g = gpu.loss(target.to(Backend::CUDA), kind);
                    same(c.values, g.values);
                    same(c.gradient, g.gradient);
                }
                near(cpu.sum(), gpu.sum(), 1e-8);
                require(gpu.all_finite(), "finite reduction");
            }
        },
        {
            "network_keeps_parameters_and_optimizer_on_device", []
            {
                Dataset data(std::vector<Sample>{{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {2}}, {{1, 1}, {3}}});
                initialize_compute(Backend::CPU);
                NeuralNetwork cpu(2, 0.02, NeuralNetwork::MINI_BATCH);
                cpu.add_layer(2, 3, Layer::TANH);
                cpu.add_layer(3, 1, Layer::IDENTITY);
                OptimizerOptions options;
                options.kind = OptimizerKind::Adam;
                cpu.set_optimizer(options);
                cpu.learn(data, 2, 2);
                const math::Matrix expected[]{cpu[0].get_weights(), cpu[1].get_weights()};
                initialize_compute(Backend::CUDA);
                NeuralNetwork gpu(2, 0.02, NeuralNetwork::MINI_BATCH);
                gpu.add_layer(2, 3, Layer::TANH);
                gpu.add_layer(3, 1, Layer::IDENTITY);
                gpu.set_optimizer(options);
                gpu.learn(data, 1, 2);
                gpu::resident::reset_statistics();
                gpu.learn(data, 1, 2);
                auto stats = gpu::resident::statistics();
                require(stats.uploads == 6,
                        "parameters/intermediates unexpectedly uploaded: expected only X/Y batches and evaluation");
                for (std::size_t i = 0; i < 2; ++i)same(gpu[i].tensor_weights(), Tensor::from_matrix(expected[i]));
            }
        }
    });
}
