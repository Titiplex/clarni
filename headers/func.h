#pragma once
#include "matrix.h"
#include <cmath>
#include <stdexcept>

namespace ml::math::func
{
    inline Vector sigmoid(const Vector& input)
    {
        Vector output(input.get_dim());
        for (int i = 0; i < input.get_dim(); ++i)
        {
            const double x = input[i];
            if (x >= 0.0) output[i] = 1.0 / (1.0 + std::exp(-x));
            else
            {
                const double e = std::exp(x);
                output[i] = e / (1.0 + e);
            }
        }
        return output;
    }

    // Argument = sigmoid(z), pas z.
    inline Vector sigmoid_derivative(const Vector& activations)
    {
        Vector output(activations.get_dim());
        for (int i = 0; i < activations.get_dim(); ++i)
            output[i] = activations[i] * (1.0 - activations[i]);
        return output;
    }

    // Gradient de L = 1/2 * somme_i (prediction[i]-target[i])^2.
    inline Vector simple(const Vector& prediction, const Vector& target)
    {
        if (prediction.get_dim() != target.get_dim())
            throw std::invalid_argument("Loss: prediction/target dimensions differ");
        Vector result(prediction.get_dim());
        for (int i = 0; i < prediction.get_dim(); ++i) result[i] = prediction[i] - target[i];
        return result;
    }

    inline double simple_loss(const Vector& prediction, const Vector& target)
    {
        if (prediction.get_dim() != target.get_dim())
            throw std::invalid_argument("Loss: prediction/target dimensions differ");
        double result = 0.0;
        for (int i = 0; i < prediction.get_dim(); ++i)
        {
            const double e = prediction[i] - target[i];
            result += 0.5 * e * e;
        }
        return result;
    }
}
