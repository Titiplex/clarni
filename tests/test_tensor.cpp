#include "test_support.h"
#include "tensor.h"
using namespace ml;

Tensor tensor(std::initializer_list<std::initializer_list<double>> rows)
{
    math::Matrix m(static_cast<int>(rows.size()), static_cast<int>(rows.begin()->size()));
    int i = 0;
    for (const auto& row : rows)
    {
        int j = 0;
        for (const double x : row)m(i, j++) = x;
        ++i;
    }
    return Tensor::from_matrix(m);
}

int main()
{
    return run_tests({
        {
            "tensor_matrix_algebra", []
            {
                const auto a = tensor({{1, 2, 3}, {4, 5, 6}});
                auto c = a.matmul(a.transpose()).to_matrix();
                near(c(0, 0), 14);
                near(c(1, 1), 77);
                near(a.sum(), 21);
                near(a.sum_rows().to_matrix()(0, 1), 7);
                near(a.sum_columns().to_matrix()(1, 0), 15);
                near(a.with_bias_row().slice_rows(1, 2).sum(), 21);
            }
        },
        {
            "stable_logits_and_loss_gradients", []
            {
                const auto binary = tensor({{-1000, 1000}}).loss(tensor({{1, 0}}), LossKind::BinaryCrossEntropy);
                near(binary.mean(), 1000);
                near(binary.gradient.to_matrix()(0, 0), -1);
                const auto z = tensor({{1000}, {1001}, {999}});
                auto y = tensor({{0}, {1}, {0}});
                const auto p = z.activate(Activation::Softmax);
                near(p.sum(), 1);
                const auto loss = z.loss(y, LossKind::CrossEntropy);
                near(loss.gradient.sum(), 0);
                for (int i = 0; i < 3; ++i)
                {
                    auto m = z.to_matrix();
                    m(i, 0) += 1e-5;
                    const double plus = Tensor::from_matrix(m).loss(y, LossKind::CrossEntropy).mean();
                    m(i, 0) -= 2e-5;
                    const double minus = Tensor::from_matrix(m).loss(y, LossKind::CrossEntropy).mean();
                    near(loss.gradient.to_matrix()(i, 0), (plus - minus) / 2e-5, 1e-7);
                }
            }
        },
        {
            "softmax_vector_jacobian_product", []
            {
                auto z = tensor({{0.1}, {-0.4}, {0.8}}), g = tensor({{0.3}, {0.7}, {-0.2}});
                const auto a = z.activate(Activation::Softmax);
                auto d = z.activation_backward(a, g, Activation::Softmax).to_matrix();
                for (int i = 0; i < 3; ++i)
                {
                    auto m = z.to_matrix();
                    m(i, 0) += 1e-6;
                    const double plus = Tensor::from_matrix(m).activate(Activation::Softmax).hadamard(g).sum();
                    m(i, 0) -= 2e-6;
                    const double minus = Tensor::from_matrix(m).activate(Activation::Softmax).hadamard(g).sum();
                    near(d(i, 0), (plus - minus) / 2e-6, 1e-7);
                }
            }
        }
    });
}
