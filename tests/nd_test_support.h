#pragma once
#include "module.h"
#include "test_support.h"

inline void check_input_gradient(const ml::nd::Shape& shape, const ml::nd::Vector& values,
                                 const std::function<ml::nd::Array(const ml::nd::Array&)>& loss,
                                 const double tol = 2e-5)
{
    const auto a = ml::nd::Array::from(shape, values, true);
    loss(a).backward();
    const auto analytical = a.grad();
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        auto plus = values, minus = values;
        plus[i] += 1e-5;
        minus[i] -= 1e-5;
        const double numeric = (loss(ml::nd::Array::from(shape, plus)).item() -
                loss(ml::nd::Array::from(shape, minus)).item()) /
            2e-5;
        near(analytical[i], numeric, tol);
    }
}

inline void check_parameter_gradients(const ml::nn::Module& model,
                                      const std::function<ml::nd::Array()>& loss, const double tol = 2e-5)
{
    auto p = ml::nn::parameters(model);
    for (auto& n : p)
        n.parameter->zero_grad();
    loss().backward();
    std::vector<ml::nd::Vector> gradients;
    for (auto& n : p)
    {
        require(n.parameter->value().has_grad(), "unreached parameter " + n.name);
        gradients.push_back(n.parameter->value().grad());
    }
    for (std::size_t k = 0; k < p.size(); ++k)
    {
        const auto param = p[k].parameter;
        const auto shape = param->value().shape();
        auto original = param->value().values();
        for (std::size_t i = 0; i < original.size(); ++i)
        {
            auto plus = original, minus = original;
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            param->assign(ml::nd::Array::from(shape, plus));
            const double a = loss().item();
            param->assign(ml::nd::Array::from(shape, minus));
            const double b = loss().item();
            near(gradients[k][i], (a - b) / 2e-5, tol);
        }
        param->assign(ml::nd::Array::from(shape, original));
    }
}
