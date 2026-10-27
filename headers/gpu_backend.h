#pragma once

#include <string>

namespace ml::gpu
{
    struct ProbeResult
    {
        bool usable = false;
        int device = -1;
        std::string name;
        std::string diagnostic;
    };

    enum class Elementwise { Add, Subtract, Scale };

    bool compiled() noexcept;
    ProbeResult probe();

    void matmul(const double* a, const double* b, double* c,
                int rows, int inner, int cols, int device);
    void elementwise(const double* a, const double* b, double* c,
                     std::size_t count, Elementwise operation,
                     double scalar, int device);
}
