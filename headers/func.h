#pragma once
#include "matrix.h"
#include <cmath>

namespace ml::math::func
{
    inline Vector sigmoid(const Vector& input)
    {
        Vector output(input.get_dim());
        for (int i = 0; i < input.get_dim(); i++)
        {
            output[i] = 1.0 / (1.0 + exp(-input[i]));
        }
        return output;
    }

    inline Vector sigmoid_derivative(const Vector& input)
    {
        Vector output(input.get_dim());
        for (int i = 0; i < input.get_dim(); i++)
        {
            const auto x = input[i];
            output[i] = x * (1 - x);
        }
        return output;
    }

    inline Vector simple(Vector output, Vector gold)
    {
        Vector result(output.get_dim());
        for (int i = 0; i < output.get_dim(); i++)
            result[i] = gold[i] - output[i];
        return result;
    }
}
