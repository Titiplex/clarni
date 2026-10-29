#include "test_support.h"
#include "func.h"

using namespace ml::math;

int main()
{
    return run_tests({
        {
            "sigmoid_derivative_of_activation", []
            {
                const auto a = func::sigmoid(Vector(std::vector<double>{-1, 0, 1}));
                const auto d = func::sigmoid_derivative(a);
                for (std::size_t i = 0; i < a.size(); ++i)near(d[i], a[i] * (1 - a[i]));
            }
        },
        {
            "squared_loss_gradient_sign", []
            {
                // Convention annoncee dans network.cpp : dL/dprediction = prediction-target.
                const auto g = func::simple(Vector(std::vector{0.2}),
                                            Vector(std::vector{1.0}));
                near(g[0], -0.8);
            }
        }
    });
}
