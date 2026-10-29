#include "test_support.h"
#include "autograd.h"
using namespace ml;

int main()
{
    return run_tests({
        {
            "shared_nodes_accumulate_each_branch_and_reset_per_backward", []
            {
                const ad::Variable x(Tensor(1, 1, 3));
                auto y = ad::add(ad::square(x), x);
                y.backward();
                near(x.grad().item(), 7);
                y.backward();
                near(x.grad().item(), 7);
            }
        },
        {
            "matrix_product_bias_broadcast_and_activation_finite_difference", []
            {
                math::Matrix w(2, 2), xm(2, 3), bm(2, 1);
                w(0, 0) = 0.3;
                w(0, 1) = -0.7;
                w(1, 0) = 0.2;
                w(1, 1) = 0.8;
                bm(0, 0) = 0.2;
                bm(1, 0) = -0.3;
                for (int i = 0; i < 2; ++i)for (int j = 0; j < 3; ++j)xm(i, j) = 0.1 * (i + j);
                const ad::Variable weight(Tensor::from_matrix(w));
                const ad::Variable x(Tensor::from_matrix(xm), false);
                const ad::Variable bias(
                    Tensor::from_matrix(bm));
                auto value = ad::mean(ad::square(
                    ad::activate(ad::add(ad::matmul(weight, x), ad::broadcast_columns(bias, 3)), Activation::Tanh)));
                value.backward();
                auto reference = [&](const math::Matrix& weights, const math::Matrix& b)
                {
                    const auto z = Tensor::from_matrix(weights).matmul(x.value()) + Tensor::from_matrix(b).
                        broadcast_columns(3);
                    return z.activate(Activation::Tanh).unary(Unary::Square).sum() / 6;
                };
                for (int i = 0; i < 2; ++i)
                {
                    auto b = bm;
                    b(i, 0) += 1e-6;
                    double plus = reference(w, b);
                    b(i, 0) -= 2e-6;
                    double minus = reference(w, b);
                    near(bias.grad().to_matrix()(i, 0), (plus - minus) / 2e-6, 1e-7);
                    for (int j = 0; j < 2; ++j)
                    {
                        auto m = w;
                        m(i, j) += 1e-6;
                        plus = reference(m, bm);
                        m(i, j) -= 2e-6;
                        minus = reference(m, bm);
                        near(weight.grad().to_matrix()(i, j), (plus - minus) / 2e-6, 1e-7);
                    }
                }
                expect_throw<std::logic_error>([&] { (void)x.grad(); });
            }
        },
        {
            "fused_loss_autograd", []
            {
                math::Matrix z(2, 1), y(2, 1);
                z(0, 0) = 0.3;
                z(1, 0) = -0.4;
                y(1, 0) = 1;
                const ad::Variable logits(Tensor::from_matrix(z));
                auto objective = ad::loss(logits, Tensor::from_matrix(y), LossKind::CrossEntropy);
                objective.backward();
                auto expected = logits.value().loss(Tensor::from_matrix(y), LossKind::CrossEntropy).gradient.
                                       to_matrix();
                near(logits.grad().to_matrix()(0, 0), expected(0, 0));
                near(logits.grad().to_matrix()(1, 0), expected(1, 0));
            }
        }
    });
}
