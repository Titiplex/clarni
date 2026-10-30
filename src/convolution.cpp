#include "convolution.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ml::nn
{
    namespace
    {
        using nd::Shape;
        using nd::Vector;

        Shape spatial_output(const Shape &input, const Shape &kernel, const Shape &stride,
                             const Shape &padding, const Shape &dilation)
        {
            const auto rank = kernel.size();
            if ((rank != 1 && rank != 2) || input.size() != rank + 2 || stride.size() != rank ||
                padding.size() != rank || dilation.size() != rank)
                throw std::invalid_argument("Convolution/pool rank mismatch");
            Shape result;
            for (std::size_t i = 0; i < rank; ++i)
            {
                if (kernel[i] <= 0 || stride[i] <= 0 || padding[i] < 0 || dilation[i] <= 0)
                    throw std::invalid_argument("Invalid spatial options");
                const auto extent = static_cast<long long>(input[i + 2]) + 2LL * padding[i] -
                                    static_cast<long long>(dilation[i]) * (kernel[i] - 1) - 1;
                if (extent < 0 || extent / stride[i] + 1 > (std::numeric_limits<int>::max)())
                    throw std::invalid_argument("Invalid spatial output size");
                result.push_back(static_cast<int>(extent / stride[i] + 1));
            }
            return result;
        }

        // Iterate all valid input/weight/output connections, without storing an im2col buffer.
        template <class F>
        void connections(const Shape &x, const Shape &w, const Shape &y, const Shape &kernel,
                         const Shape &stride, const Shape &padding, const Shape &dilation, F fn)
        {
            const int ih = x.size() == 4 ? x[2] : 1;
            const int iw = x.back();
            const int oh = y.size() == 4 ? y[2] : 1;
            const int ow = y.back();
            const int kh = kernel.size() == 2 ? kernel[0] : 1;
            const int kw = kernel.back();
            const int sh = kernel.size() == 2 ? stride[0] : 1;
            const int sw = stride.back();
            const int ph = kernel.size() == 2 ? padding[0] : 0;
            const int pw = padding.back();
            const int dh = kernel.size() == 2 ? dilation[0] : 1;
            const int dw = dilation.back();
            for (int b = 0; b < x[0]; ++b)
                for (int oc = 0; oc < w[0]; ++oc)
                    for (int h = 0; h < oh; ++h)
                        for (int l = 0; l < ow; ++l)
                        {
                            std::size_t oi =
                                ((static_cast<std::size_t>(b) * w[0] + oc) * oh + h) * ow + l;
                            for (int ic = 0; ic < x[1]; ++ic)
                                for (int r = 0; r < kh; ++r)
                                    for (int c = 0; c < kw; ++c)
                                    {
                                        const long long xi = static_cast<long long>(h) * sh - ph +
                                                             static_cast<long long>(r) * dh;
                                        const long long xj = static_cast<long long>(l) * sw - pw +
                                                             static_cast<long long>(c) * dw;
                                        if (xi < 0 || xi >= ih || xj < 0 || xj >= iw)
                                            continue;
                                        std::size_t ii =
                                            ((static_cast<std::size_t>(b) * x[1] + ic) * ih +
                                             static_cast<std::size_t>(xi)) *
                                                iw +
                                            static_cast<std::size_t>(xj);
                                        std::size_t wi =
                                            ((static_cast<std::size_t>(oc) * x[1] + ic) * kh + r) *
                                                kw +
                                            c;
                                        fn(ii, wi, oi);
                                    }
                        }
        }
    } // namespace
    namespace
    {
        gpu::resident::NDDescriptor spatial_descriptor(const Shape &x, const Shape &y,
                                                       const Shape &kernel, const Shape &stride,
                                                       const Shape &padding, const Shape &dilation)
        {
            gpu::resident::NDDescriptor d;
            d.x[0] = x[0];
            d.x[1] = x[1];
            d.x[2] = x.size() == 4 ? x[2] : 1;
            d.x[3] = x.back();
            d.y[0] = y[0];
            d.y[1] = y[1];
            d.y[2] = y.size() == 4 ? y[2] : 1;
            d.y[3] = y.back();
            d.w[0] = y[1];
            d.w[1] = x[1];
            d.w[2] = kernel.size() == 2 ? kernel[0] : 1;
            d.w[3] = kernel.back();
            d.stride[0] = kernel.size() == 2 ? stride[0] : 1;
            d.stride[1] = stride.back();
            d.padding[0] = kernel.size() == 2 ? padding[0] : 0;
            d.padding[1] = padding.back();
            d.dilation[0] = kernel.size() == 2 ? dilation[0] : 1;
            d.dilation[1] = dilation.back();
            return d;
        }
    } // namespace

    Convolution::Convolution(const int in, int out, Shape k, Shape s, Shape p, Shape d,
                             const std::uint32_t seed)
        : in_(in), out_(out), kernel_(std::move(k)), stride_(std::move(s)), padding_(std::move(p)),
          dilation_(std::move(d))
    {
        // Validate all options without requiring an output for this dummy image.
        if ((kernel_.size() != 1 && kernel_.size() != 2) || stride_.size() != kernel_.size() ||
            padding_.size() != kernel_.size() || dilation_.size() != kernel_.size())
            throw std::invalid_argument("Convolution rank mismatch");
        for (std::size_t i = 0; i < kernel_.size(); ++i)
            if (stride_[i] <= 0 || padding_[i] < 0 || dilation_[i] <= 0)
                throw std::invalid_argument("Invalid convolution options");
        Shape shape = {out, in};
        shape.insert(shape.end(), kernel_.begin(), kernel_.end());
        nd::numel(shape);
        std::mt19937 gen(seed);
        weight = normal_parameter(
            shape, std::sqrt(2.0 / (static_cast<double>(in) * nd::numel(kernel_))), gen);
        bias = std::make_shared<Parameter>(nd::Array({out}));
    }

    nd::Array Convolution::forward(const nd::Array &a)
    {
        auto xshape = a.shape();
        auto spatial = spatial_output(xshape, kernel_, stride_, padding_, dilation_);
        if (xshape[1] != in_)
            throw std::invalid_argument("Convolution input channels mismatch");
        Shape yshape = {xshape[0], out_};
        yshape.insert(yshape.end(), spatial.begin(), spatial.end());
        if (a.backend() != weight->value().backend() || a.backend() != bias->value().backend())
            throw std::invalid_argument("Convolution input and parameters must share a backend");
        if (a.backend() == Backend::CUDA)
        {
            using gpu::resident::NDOperation;
            auto d = spatial_descriptor(xshape, yshape, kernel_, stride_, padding_, dilation_);
            const auto x = a.variable().value(), w = weight->value().variable().value(),
                       b = bias->value().variable().value();
            auto value = x.nd_operation(NDOperation::Convolution,
                                        static_cast<int>(nd::numel(yshape)), 1, d, &w, &b);
            auto result = ad::operation(
                std::move(value),
                {{a.variable(), [d, x, w, n = static_cast<int>(a.size())](const Tensor &g)
                  { return x.nd_operation(NDOperation::ConvInputGradient, n, 1, d, &w, &g); }},
                 {weight->value().variable(),
                  [d, x, w, n = static_cast<int>(weight->value().size())](const Tensor &g)
                  { return x.nd_operation(NDOperation::ConvWeightGradient, n, 1, d, &w, &g); }},
                 {bias->value().variable(), [d, n = out_](const Tensor &g)
                  { return g.nd_operation(NDOperation::ConvBiasGradient, n, 1, d); }}});
            return nd::Array(yshape, result);
        }
        auto x = a.values(), w = weight->value().values();
        const auto biases = bias->value().values();
        auto wshape = weight->value().shape();
        Vector y(nd::numel(yshape));
        auto spatial_size = nd::numel(spatial);
        for (std::size_t i = 0; i < y.size(); ++i)
            y[i] = biases[i / spatial_size % out_];
        auto k = kernel_, s = stride_, p = padding_, d = dilation_;
        connections(xshape, wshape, yshape, k, s, p, d,
                    [&](auto ii, auto wi, auto oi) { y[oi] += x[ii] * w[wi]; });
        return nd::operation(
            yshape, std::move(y),
            {{a,
              [xshape, wshape, yshape, k, s, p, d, w](const Vector &g)
              {
                  Vector r(nd::numel(xshape), 0);
                  connections(xshape, wshape, yshape, k, s, p, d,
                              [&](auto ii, auto wi, auto oi) { r[ii] += g[oi] * w[wi]; });
                  return r;
              }},
             {weight->value(),
              [xshape, wshape, yshape, k, s, p, d, x](const Vector &g)
              {
                  Vector r(nd::numel(wshape), 0);
                  connections(xshape, wshape, yshape, k, s, p, d,
                              [&](auto ii, auto wi, auto oi) { r[wi] += g[oi] * x[ii]; });
                  return r;
              }},
             {bias->value(), [out = out_, spatial_size](const Vector &g)
              {
                  Vector r(out, 0);
                  for (std::size_t i = 0; i < g.size(); ++i)
                      r[i / spatial_size % out] += g[i];
                  return r;
              }}});
    }

    nd::Array pool(const nd::Array &a, const Shape &kernel, const Shape &stride, PoolKind kind)
    {
        if (kind != PoolKind::Max && kind != PoolKind::Average)
            throw std::invalid_argument("Invalid pool kind");
        auto spatial = spatial_output(a.shape(), kernel, stride, Shape(kernel.size(), 0),
                                      Shape(kernel.size(), 1));
        Shape shape = {a.shape()[0], a.shape()[1]};
        shape.insert(shape.end(), spatial.begin(), spatial.end());
        if (a.backend() == Backend::CUDA)
        {
            using gpu::resident::NDOperation;
            auto d = spatial_descriptor(a.shape(), shape, kernel, stride, Shape(kernel.size(), 0),
                                        Shape(kernel.size(), 1));
            d.mode = kind == PoolKind::Max ? 0 : 1;
            const auto x = a.variable().value();
            auto value =
                x.nd_operation(NDOperation::Pool, static_cast<int>(nd::numel(shape)), 1, d);
            auto result = ad::operation(
                std::move(value),
                {{a.variable(), [d, x, n = static_cast<int>(a.size())](const Tensor &g)
                  { return x.nd_operation(NDOperation::PoolGradient, n, 1, d, &g); }}});
            return nd::Array(shape, result);
        }
        auto x = a.values();
        Vector out(nd::numel(shape),
                   kind == PoolKind::Max ? -std::numeric_limits<double>::infinity() : 0);
        std::vector argmax(out.size(), a.size());
        int ih = a.shape().size() == 4 ? a.shape()[2] : 1, iw = a.shape().back();
        int oh = shape.size() == 4 ? shape[2] : 1, ow = shape.back();
        const int kw = kernel.back();
        const int kh = kernel.size() == 2 ? kernel[0] : 1;
        int sh = kernel.size() == 2 ? stride[0] : 1, sw = stride.back();
        const double divisor = static_cast<double>(kh) * kw;
        auto visit = [=](auto fn)
        {
            for (int b = 0; b < shape[0]; ++b)
                for (int c = 0; c < shape[1]; ++c)
                    for (int h = 0; h < oh; ++h)
                        for (int l = 0; l < ow; ++l)
                        {
                            auto oi =
                                ((static_cast<std::size_t>(b) * shape[1] + c) * oh + h) * ow + l;
                            for (int r = 0; r < kh; ++r)
                                for (int q = 0; q < kw; ++q)
                                {
                                    auto ii = ((static_cast<std::size_t>(b) * shape[1] + c) * ih +
                                               h * sh + r) *
                                                  iw +
                                              l * sw + q;
                                    fn(ii, oi);
                                }
                        }
        };
        visit(
            [&](auto ii, auto oi)
            {
                if (kind == PoolKind::Average)
                    out[oi] += x[ii] / divisor;
                else if (argmax[oi] == a.size() || x[ii] > out[oi])
                {
                    out[oi] = x[ii];
                    argmax[oi] = ii;
                }
            });
        return nd::operation(shape, std::move(out),
                             {{a, [visit, argmax, kind, divisor, n = a.size()](const Vector &g)
                               {
                                   Vector r(n, 0);
                                   if (kind == PoolKind::Average)
                                       visit([&](auto ii, auto oi) { r[ii] += g[oi] / divisor; });
                                   else
                                       for (std::size_t i = 0; i < g.size(); ++i)
                                           r[argmax[i]] += g[i];
                                   return r;
                               }}});
    }

    nd::Array GlobalMaxPool1D::forward(const nd::Array &a)
    {
        if (a.shape().size() != 3)
            throw std::invalid_argument("GlobalMaxPool1D expects [B,C,L]");
        return nd::reshape(pool(a, {a.shape()[2]}, {1}, PoolKind::Max),
                           {a.shape()[0], a.shape()[1]});
    }
} // namespace ml::nn
