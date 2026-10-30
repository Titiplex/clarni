#pragma once
#include <cstddef>
namespace ml::gpu::resident
{
    // Fixed-size POD copied as kernel arguments. Shapes/strides never require a host tensor copy.
    inline constexpr int nd_max_rank = 8;
    enum class NDOperation
    {
        Index,
        IndexBackward,
        Convolution,
        ConvInputGradient,
        ConvWeightGradient,
        ConvBiasGradient,
        Pool,
        PoolGradient,
        CrossEntropy,
        CrossEntropyGradient,
        Norm,
        Argmax
    };
    struct NDDescriptor
    {
        int rank = 0, src[nd_max_rank]{}, dst[nd_max_rank]{}, axes[nd_max_rank]{};
        int mode = 0, axis = 0, first = 0;
        int x[4]{}, w[4]{}, y[4]{}, stride[2]{1, 1}, padding[2]{}, dilation[2]{1, 1};
        int samples = 0, classes = 0, valid = 0, ignore = -1;
    };
    void nd_operation(NDOperation, const NDDescriptor &, const double *a, const double *b,
                      const double *c, double *out, std::size_t count, int device);
} // namespace ml::gpu::resident
