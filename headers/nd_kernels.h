#pragma once
#include "nd_backend.h"
#include <cfloat>
#include <cmath>
// The same per-output math is executed by CUDA kernels and the CPU mock tests.
#if defined(__CUDACC__)
#define CLARNI_ND_HD __host__ __device__
#else
#define CLARNI_ND_HD
#endif
namespace ml::gpu::detail
{
    using resident::NDDescriptor;
    using resident::NDOperation;
    CLARNI_ND_HD inline std::size_t offset4(int b, int c, int h, int w, const int *shape)
    {
        return ((std::size_t(b) * shape[1] + c) * shape[2] + h) * shape[3] + w;
    }
    CLARNI_ND_HD inline std::size_t stride_at(const int *shape, int rank, int axis)
    {
        std::size_t s = 1;
        for (int j = axis + 1; j < rank; ++j)
            s *= shape[j];
        return s;
    }
    CLARNI_ND_HD inline std::size_t source_index(std::size_t index, const NDDescriptor &d)
    {
        std::size_t result = 0;
        for (int axis = 0; axis < d.rank; ++axis)
        {
            int coordinate = int(index / stride_at(d.dst, d.rank, axis) % d.dst[axis]);
            if (d.mode == 0)
            {
                if (d.src[axis] != 1)
                    result += std::size_t(coordinate) * stride_at(d.src, d.rank, axis);
            }
            else if (d.mode == 1)
                result += std::size_t(coordinate) * stride_at(d.src, d.rank, d.axes[axis]);
            else
                result += std::size_t(coordinate + (axis == d.axis ? d.first : 0)) *
                          stride_at(d.src, d.rank, axis);
        }
        return result;
    }
    CLARNI_ND_HD inline double index_backward(std::size_t index, const double *g,
                                              const NDDescriptor &d)
    {
        std::size_t base = 0, repeats = 1;
        if (d.mode == 1)
        {
            for (int axis = 0; axis < d.rank; ++axis)
            {
                auto coordinate =
                    index / stride_at(d.src, d.rank, d.axes[axis]) % d.src[d.axes[axis]];
                base += coordinate * stride_at(d.dst, d.rank, axis);
            }
            return g[base];
        }
        for (int axis = 0; axis < d.rank; ++axis)
        {
            int coordinate = int(index / stride_at(d.src, d.rank, axis) % d.src[axis]);
            if (d.mode == 2 && axis == d.axis)
            {
                coordinate -= d.first;
                if (coordinate < 0 || coordinate >= d.dst[axis])
                    return 0;
            }
            if (d.mode == 0 && d.src[axis] == 1)
                repeats *= d.dst[axis];
            else
                base += std::size_t(coordinate) * stride_at(d.dst, d.rank, axis);
        }
        double sum = 0;
        for (std::size_t repeat = 0; repeat < repeats; ++repeat)
        {
            auto q = repeat, output = base;
            for (int axis = d.rank - 1; axis >= 0; --axis)
                if (d.mode == 0 && d.src[axis] == 1)
                {
                    output += (q % d.dst[axis]) * stride_at(d.dst, d.rank, axis);
                    q /= d.dst[axis];
                }
            sum += g[output];
        }
        return sum;
    }
    CLARNI_ND_HD inline std::size_t pool_winner(const double *x, const NDDescriptor &d, int b,
                                                int channel, int oh, int ow)
    {
        std::size_t winner = offset4(b, channel, oh * d.stride[0], ow * d.stride[1], d.x);
        double maximum = x[winner];
        for (int r = 0; r < d.w[2]; ++r)
            for (int q = 0; q < d.w[3]; ++q)
            {
                auto i = offset4(b, channel, oh * d.stride[0] + r, ow * d.stride[1] + q, d.x);
                if (x[i] > maximum)
                {
                    maximum = x[i];
                    winner = i;
                }
            }
        return winner;
    }
    CLARNI_ND_HD inline double nd_value(NDOperation op, const NDDescriptor &d, const double *a,
                                        const double *b, const double *c, std::size_t index)
    {
        if (op == NDOperation::Index)
            return a[source_index(index, d)];
        if (op == NDOperation::IndexBackward)
            return index_backward(index, a, d);
        if (op == NDOperation::Norm)
        {
            double scale = 0, ssq = 1;
            for (int i = 0; i < d.samples; ++i)
            {
                double v = a[i];
                if (!(v <= DBL_MAX && v >= -DBL_MAX))
                    return HUGE_VAL;
                v = v < 0 ? -v : v;
                if (v != 0)
                {
                    if (scale < v)
                    {
                        double r = scale / v;
                        ssq = 1 + ssq * r * r;
                        scale = v;
                    }
                    else
                    {
                        double r = v / scale;
                        ssq += r * r;
                    }
                }
            }
            return scale == 0 ? 0 : scale * sqrt(ssq);
        }
        if (op == NDOperation::Argmax)
        {
            auto start = index * d.classes;
            int maximum = 0;
            for (int j = 1; j < d.classes; ++j)
                if (a[start + j] > a[start + maximum])
                    maximum = j;
            return maximum;
        }
        if (op == NDOperation::CrossEntropy || op == NDOperation::CrossEntropyGradient)
        {
            auto row = op == NDOperation::CrossEntropy ? index : index / d.classes;
            int label = int(b[row]);
            if (label == d.ignore)
                return 0;
            auto start = row * d.classes;
            double maximum = a[start], denominator = 0;
            for (int j = 1; j < d.classes; ++j)
                if (a[start + j] > maximum)
                    maximum = a[start + j];
            for (int j = 0; j < d.classes; ++j)
                denominator += exp(a[start + j] - maximum);
            if (op == NDOperation::CrossEntropy)
                return maximum - a[start + label] + log(denominator);
            return (exp(a[index] - maximum) / denominator -
                    (int(index % d.classes) == label ? 1.0 : 0.0)) /
                   d.valid;
        }
        if (op == NDOperation::ConvBiasGradient)
        {
            double sum = 0;
            for (int batch = 0; batch < d.y[0]; ++batch)
                for (int h = 0; h < d.y[2]; ++h)
                    for (int w = 0; w < d.y[3]; ++w)
                        sum += a[offset4(batch, int(index), h, w, d.y)];
            return sum;
        }
        if (op == NDOperation::Convolution)
        {
            int ow = int(index % d.y[3]), oh = int(index / d.y[3] % d.y[2]),
                oc = int(index / (std::size_t(d.y[3]) * d.y[2]) % d.y[1]),
                batch = int(index / (std::size_t(d.y[3]) * d.y[2] * d.y[1]));
            double sum = c[oc];
            for (int ic = 0; ic < d.x[1]; ++ic)
                for (int r = 0; r < d.w[2]; ++r)
                    for (int q = 0; q < d.w[3]; ++q)
                    {
                        long long ih = static_cast<long long>(oh) * d.stride[0] - d.padding[0] +
                                       static_cast<long long>(r) * d.dilation[0];
                        long long iw = static_cast<long long>(ow) * d.stride[1] - d.padding[1] +
                                       static_cast<long long>(q) * d.dilation[1];
                        if (ih >= 0 && ih < d.x[2] && iw >= 0 && iw < d.x[3])
                            sum += a[offset4(batch, ic, int(ih), int(iw), d.x)] *
                                   b[((std::size_t(oc) * d.x[1] + ic) * d.w[2] + r) * d.w[3] + q];
                    }
            return sum;
        }
        if (op == NDOperation::ConvInputGradient)
        {
            int iw = int(index % d.x[3]), ih = int(index / d.x[3] % d.x[2]),
                ic = int(index / (std::size_t(d.x[3]) * d.x[2]) % d.x[1]),
                batch = int(index / (std::size_t(d.x[3]) * d.x[2] * d.x[1]));
            double sum = 0;
            for (int oc = 0; oc < d.y[1]; ++oc)
                for (int r = 0; r < d.w[2]; ++r)
                    for (int q = 0; q < d.w[3]; ++q)
                    {
                        long long oh = static_cast<long long>(ih) + d.padding[0] -
                                       static_cast<long long>(r) * d.dilation[0],
                                  ow = static_cast<long long>(iw) + d.padding[1] -
                                       static_cast<long long>(q) * d.dilation[1];
                        if (oh < 0 || ow < 0 || oh % d.stride[0] || ow % d.stride[1])
                            continue;
                        oh /= d.stride[0];
                        ow /= d.stride[1];
                        if (oh < d.y[2] && ow < d.y[3])
                            sum += c[offset4(batch, oc, int(oh), int(ow), d.y)] *
                                   b[((std::size_t(oc) * d.x[1] + ic) * d.w[2] + r) * d.w[3] + q];
                    }
            return sum;
        }
        if (op == NDOperation::ConvWeightGradient)
        {
            int q = int(index % d.w[3]), r = int(index / d.w[3] % d.w[2]),
                ic = int(index / (std::size_t(d.w[3]) * d.w[2]) % d.w[1]),
                oc = int(index / (std::size_t(d.w[3]) * d.w[2] * d.w[1]));
            double sum = 0;
            for (int batch = 0; batch < d.x[0]; ++batch)
                for (int oh = 0; oh < d.y[2]; ++oh)
                    for (int ow = 0; ow < d.y[3]; ++ow)
                    {
                        long long ih = static_cast<long long>(oh) * d.stride[0] - d.padding[0] +
                                       static_cast<long long>(r) * d.dilation[0],
                                  iw = static_cast<long long>(ow) * d.stride[1] - d.padding[1] +
                                       static_cast<long long>(q) * d.dilation[1];
                        if (ih >= 0 && ih < d.x[2] && iw >= 0 && iw < d.x[3])
                            sum += a[offset4(batch, ic, int(ih), int(iw), d.x)] *
                                   c[offset4(batch, oc, oh, ow, d.y)];
                    }
            return sum;
        }
        if (op == NDOperation::Pool)
        {
            int ow = int(index % d.y[3]), oh = int(index / d.y[3] % d.y[2]),
                channel = int(index / (std::size_t(d.y[3]) * d.y[2]) % d.y[1]),
                batch = int(index / (std::size_t(d.y[3]) * d.y[2] * d.y[1]));
            if (d.mode == 0)
                return a[pool_winner(a, d, batch, channel, oh, ow)];
            double sum = 0;
            for (int r = 0; r < d.w[2]; ++r)
                for (int q = 0; q < d.w[3]; ++q)
                    sum +=
                        a[offset4(batch, channel, oh * d.stride[0] + r, ow * d.stride[1] + q, d.x)];
            return sum / (double(d.w[2]) * d.w[3]);
        }
        if (op == NDOperation::PoolGradient)
        {
            int iw = int(index % d.x[3]), ih = int(index / d.x[3] % d.x[2]),
                channel = int(index / (std::size_t(d.x[3]) * d.x[2]) % d.x[1]),
                batch = int(index / (std::size_t(d.x[3]) * d.x[2] * d.x[1]));
            double sum = 0;
            int hfirst = ih >= d.w[2] ? (ih - d.w[2]) / d.stride[0] + 1 : 0,
                wfirst = iw >= d.w[3] ? (iw - d.w[3]) / d.stride[1] + 1 : 0;
            for (int oh = hfirst; oh <= ih / d.stride[0] && oh < d.y[2]; ++oh)
                for (int ow = wfirst; ow <= iw / d.stride[1] && ow < d.y[3]; ++ow)
                {
                    if (d.mode == 1)
                        sum += b[offset4(batch, channel, oh, ow, d.y)] / (double(d.w[2]) * d.w[3]);
                    else if (pool_winner(a, d, batch, channel, oh, ow) == index)
                        sum += b[offset4(batch, channel, oh, ow, d.y)];
                }
            return sum;
        }
        return 0;
    }
} // namespace ml::gpu::detail
#undef CLARNI_ND_HD
