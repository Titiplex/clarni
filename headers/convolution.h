#pragma once
#include "module.h"
namespace ml::nn
{
    // Input [B,C,L] or [B,C,H,W]; cross-correlation, zero padding, no groups.
    class Convolution : public Module
    {
        int in_, out_;
        nd::Shape kernel_, stride_, padding_, dilation_;

      protected:
        Convolution(int, int, nd::Shape, nd::Shape, nd::Shape, nd::Shape, std::uint32_t);

      public:
        Param weight, bias;
        nd::Array forward(const nd::Array &) override;
        std::vector<NamedParameter> named_parameters() const override
        {
            return {{"weight", weight}, {"bias", bias}};
        }
    };
    class Conv1D final : public Convolution
    {
      public:
        Conv1D(int in, int out, int kernel, int stride = 1, int padding = 0, int dilation = 1,
               std::uint32_t seed = 42)
            : Convolution(in, out, {kernel}, {stride}, {padding}, {dilation}, seed)
        {
        }
    };
    class Conv2D final : public Convolution
    {
      public:
        Conv2D(int in, int out, nd::Shape kernel, nd::Shape stride = {1, 1},
               nd::Shape padding = {0, 0}, nd::Shape dilation = {1, 1}, std::uint32_t seed = 42)
            : Convolution(in, out, std::move(kernel), std::move(stride), std::move(padding),
                          std::move(dilation), seed)
        {
        }
    };
    enum class PoolKind
    {
        Max,
        Average
    };
    nd::Array pool(const nd::Array &, nd::Shape kernel, nd::Shape stride,
                   PoolKind kind = PoolKind::Max);
    class Pool1D final : public Module
    {
        int kernel_, stride_;
        PoolKind kind_;

      public:
        Pool1D(int kernel, int stride, PoolKind kind = PoolKind::Max)
            : kernel_(kernel), stride_(stride), kind_(kind)
        {
        }
        nd::Array forward(const nd::Array &a) override
        {
            return pool(a, {kernel_}, {stride_}, kind_);
        }
    };
    class Pool2D final : public Module
    {
        nd::Shape kernel_, stride_;
        PoolKind kind_;

      public:
        Pool2D(nd::Shape kernel, nd::Shape stride, PoolKind kind = PoolKind::Max)
            : kernel_(std::move(kernel)), stride_(std::move(stride)), kind_(kind)
        {
        }
        nd::Array forward(const nd::Array &a) override
        {
            return pool(a, kernel_, stride_, kind_);
        }
    };
    class GlobalMaxPool1D final : public Module
    {
      public:
        nd::Array forward(const nd::Array &) override;
    };
} // namespace ml::nn
