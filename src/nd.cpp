#include "nd.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ml::nd
{
    std::size_t numel(const Shape &shape)
    {
        std::size_t result = 1;
        for (const int d : shape)
        {
            if (d <= 0)
                throw std::invalid_argument("N-D dimensions must be positive");
            if (result > static_cast<std::size_t>((std::numeric_limits<int>::max)()) / d)
                throw std::length_error("N-D storage exceeds current Tensor index range");
            result *= d;
        }
        return result;
    }
    namespace
    {
        Tensor pack(const Vector &values)
        {
            if (values.empty() ||
                values.size() > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
                throw std::length_error("Invalid flat storage size");
            math::Matrix m(static_cast<int>(values.size()), 1);
            for (int i = 0; i < m.get_x(); ++i)
                m(i, 0) = values[i];
            return Tensor::from_matrix(m);
        }
        Vector unpack(const Tensor &t)
        {
            if (t.backend() != Backend::CPU)
                throw std::invalid_argument("N-D reference operations require CPU tensors");
            auto m = t.to_matrix();
            Vector result(t.size());
            for (int i = 0; i < m.get_x(); ++i)
                for (int j = 0; j < m.get_y(); ++j)
                    result[static_cast<std::size_t>(i) * m.get_y() + j] = m(i, j);
            return result;
        }
        int axis_index(int axis, const Shape &shape)
        {
            if (axis < 0)
                axis += static_cast<int>(shape.size());
            if (axis < 0 || axis >= static_cast<int>(shape.size()))
                throw std::invalid_argument("Invalid axis");
            return axis;
        }
        std::vector<std::size_t> strides(const Shape &s)
        {
            std::vector<std::size_t> st(s.size(), 1);
            for (std::size_t i = s.size(); i > 1; --i)
                st[i - 2] = st[i - 1] * s[i - 1];
            return st;
        }
        // out[i] = input[index[i]], backward scatter-add (including repeated indices).
        Array gather(const Array &a, Shape shape, std::vector<std::size_t> index)
        {
            const auto source = a.values();
            Vector out(index.size());
            for (std::size_t i = 0; i < index.size(); ++i)
                out[i] = source.at(index[i]);
            return operation(std::move(shape), std::move(out),
                             {{a, [index, n = a.size()](const Vector &g)
                               {
                                   Vector result(n, 0);
                                   for (std::size_t i = 0; i < index.size(); ++i)
                                       result[index[i]] += g[i];
                                   return result;
                               }}});
        }
        Shape broadcast_shape(const Shape &a, const Shape &b)
        {
            Shape out((std::max)(a.size(), b.size()), 1);
            for (std::size_t i = 0; i < out.size(); ++i)
            {
                const int x = i < a.size() ? a[a.size() - 1 - i] : 1;
                const int y = i < b.size() ? b[b.size() - 1 - i] : 1;
                if (x != y && x != 1 && y != 1)
                    throw std::invalid_argument("Broadcast mismatch");
                out[out.size() - 1 - i] = (std::max)(x, y);
            }
            return out;
        }
    } // namespace
    Array::Array(Shape shape, ad::Variable data) : shape_(std::move(shape)), data_(std::move(data))
    {
        if (data_.value().backend() != Backend::CPU)
            throw std::invalid_argument("N-D frontend is CPU reference only");
        if (data_.value().cols() != 1 || data_.value().size() != numel(shape_))
            throw std::invalid_argument("N-D storage mismatch");
    }
    Array::Array(Shape shape, double fill, bool requires_grad)
        : Array(shape, ad::Variable(Tensor(static_cast<int>(numel(shape)), 1, fill), requires_grad))
    {
    }
    Array Array::from(Shape shape, const Vector &values, bool requires_grad)
    {
        if (numel(shape) != values.size())
            throw std::invalid_argument("N-D value count mismatch");
        return Array(std::move(shape), ad::Variable(pack(values), requires_grad));
    }
    Vector Array::values() const
    {
        return unpack(data_.value());
    }
    Vector Array::grad() const
    {
        return unpack(data_.grad());
    }
    Array operation(Shape shape, Vector values, const std::vector<std::pair<Array, VJP>> &parents)
    {
        if (numel(shape) != values.size())
            throw std::invalid_argument("Operation shape mismatch");
        std::vector<std::pair<ad::Variable, ad::Pullback>> edges;
        for (const auto &[parent, vjp] : parents)
            edges.emplace_back(parent.variable(),
                               [vjp, n = parent.size()](const Tensor &g)
                               {
                                   auto result = vjp(unpack(g));
                                   if (result.size() != n)
                                       throw std::logic_error("N-D pullback size mismatch");
                                   return pack(result);
                               });
        return Array(std::move(shape), ad::operation(pack(values), edges));
    }
    Array reshape(const Array &a, Shape shape)
    {
        if (numel(shape) != a.size())
            throw std::invalid_argument("reshape size mismatch");
        return Array(std::move(shape), a.variable());
    }
    Array permute(const Array &a, const std::vector<int> &axes)
    {
        if (axes.size() != a.shape().size())
            throw std::invalid_argument("Permutation rank mismatch");
        Shape shape(axes.size());
        std::vector<bool> seen(axes.size(), false);
        for (std::size_t i = 0; i < axes.size(); ++i)
        {
            int ax = axis_index(axes[i], a.shape());
            if (seen[ax])
                throw std::invalid_argument("Repeated permutation axis");
            seen[ax] = true;
            shape[i] = a.shape()[ax];
        }
        auto src = strides(a.shape()), dst = strides(shape);
        std::vector<std::size_t> index(a.size(), 0);
        for (std::size_t i = 0; i < index.size(); ++i)
            for (std::size_t j = 0; j < shape.size(); ++j)
                index[i] += (i / dst[j] % shape[j]) * src[axis_index(axes[j], a.shape())];
        return gather(a, std::move(shape), std::move(index));
    }
    Array slice(const Array &a, int axis, int first, int count)
    {
        axis = axis_index(axis, a.shape());
        if (first < 0 || count <= 0 || first > a.shape()[axis] - count)
            throw std::invalid_argument("Invalid slice");
        Shape shape = a.shape();
        shape[axis] = count;
        auto src = strides(a.shape()), dst = strides(shape);
        std::vector<std::size_t> index(numel(shape), static_cast<std::size_t>(first) * src[axis]);
        for (std::size_t i = 0; i < index.size(); ++i)
            for (std::size_t j = 0; j < shape.size(); ++j)
                index[i] += (i / dst[j] % shape[j]) * src[j];
        return gather(a, std::move(shape), std::move(index));
    }
    Array concat(const std::vector<Array> &arrays, int axis)
    {
        if (arrays.empty())
            throw std::invalid_argument("Empty concat");
        Shape shape = arrays.front().shape();
        axis = axis_index(axis, shape);
        std::size_t axis_size = 0;
        for (const auto &a : arrays)
        {
            if (a.shape().size() != shape.size())
                throw std::invalid_argument("Concat rank mismatch");
            for (std::size_t j = 0; j < shape.size(); ++j)
                if (static_cast<int>(j) != axis && a.shape()[j] != shape[j])
                    throw std::invalid_argument("Concat mismatch");
            axis_size += a.shape()[axis];
        }
        if (axis_size > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
            throw std::length_error("Concat too large");
        shape[axis] = static_cast<int>(axis_size);
        Vector out(numel(shape));
        auto dst = strides(shape);
        std::vector<std::pair<Array, VJP>> parents;
        int offset = 0;
        for (const auto &a : arrays)
        {
            auto src = strides(a.shape());
            auto v = a.values();
            std::vector<std::size_t> index(a.size(), 0);
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                for (std::size_t j = 0; j < shape.size(); ++j)
                    index[i] +=
                        ((i / src[j] % a.shape()[j]) + (static_cast<int>(j) == axis ? offset : 0)) *
                        dst[j];
                out[index[i]] = v[i];
            }
            parents.emplace_back(a,
                                 [index](const Vector &g)
                                 {
                                     Vector r(index.size());
                                     for (std::size_t i = 0; i < r.size(); ++i)
                                         r[i] = g[index[i]];
                                     return r;
                                 });
            offset += a.shape()[axis];
        }
        return operation(std::move(shape), std::move(out), parents);
    }
    Array broadcast_to(const Array &a, Shape shape)
    {
        if (shape.size() < a.shape().size() || broadcast_shape(a.shape(), shape) != shape)
            throw std::invalid_argument("Invalid broadcast target");
        auto src = strides(a.shape()), dst = strides(shape);
        std::vector<std::size_t> index(numel(shape), 0);
        const auto shift = shape.size() - a.shape().size();
        for (std::size_t i = 0; i < index.size(); ++i)
            for (std::size_t j = 0; j < a.shape().size(); ++j)
                if (a.shape()[j] != 1)
                    index[i] += (i / dst[j + shift] % shape[j + shift]) * src[j];
        return gather(a, std::move(shape), std::move(index));
    }
    Array add(const Array &a, const Array &b)
    {
        auto shape = broadcast_shape(a.shape(), b.shape());
        auto x = broadcast_to(a, shape), y = broadcast_to(b, shape);
        return Array(shape, ad::add(x.variable(), y.variable()));
    }
    Array multiply(const Array &a, const Array &b)
    {
        auto shape = broadcast_shape(a.shape(), b.shape());
        auto x = broadcast_to(a, shape), y = broadcast_to(b, shape);
        return Array(shape, ad::hadamard(x.variable(), y.variable()));
    }
    Array scale(const Array &a, double factor, double offset)
    {
        return Array(a.shape(), ad::affine(a.variable(), factor, offset));
    }
    Array activate(const Array &a, Activation activation)
    {
        if (activation == Activation::Softmax)
            return softmax(a);
        return Array(a.shape(), ad::activate(a.variable(), activation));
    }
    Array sum(const Array &a)
    {
        return Array({}, ad::sum(a.variable()));
    }
    Array mean(const Array &a)
    {
        return scale(sum(a), 1.0 / static_cast<double>(a.size()));
    }
    Array reduce_sum(const Array &a, int axis, bool keepdim)
    {
        axis = axis_index(axis, a.shape());
        Shape shape = a.shape();
        shape[axis] = 1;
        auto dst = strides(shape), src = strides(a.shape());
        std::vector<std::size_t> index(a.size(), 0);
        Vector out(numel(shape), 0), v = a.values();
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            for (std::size_t j = 0; j < shape.size(); ++j)
                if (static_cast<int>(j) != axis)
                    index[i] += (i / src[j] % a.shape()[j]) * dst[j];
            out[index[i]] += v[i];
        }
        if (!keepdim)
            shape.erase(shape.begin() + axis);
        return operation(shape, std::move(out),
                         {{a, [index](const Vector &g)
                           {
                               Vector r(index.size());
                               for (std::size_t i = 0; i < r.size(); ++i)
                                   r[i] = g[index[i]];
                               return r;
                           }}});
    }
    Array matmul(const Array &a, const Array &b)
    {
        if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0])
            throw std::invalid_argument("matmul expects compatible rank-2 arrays");
        auto view = [](const Array &x)
        {
            const auto &v = x.variable();
            return ad::operation(v.value().reshape(x.shape()[0], x.shape()[1]),
                                 {{v, [n = x.size()](const Tensor &g)
                                   { return g.reshape(static_cast<int>(n), 1); }}});
        };
        const Shape output_shape = {a.shape()[0], b.shape()[1]};
        const auto output_size = numel(output_shape);
        auto result = ad::matmul(view(a), view(b));
        auto flat = ad::operation(result.value().reshape(static_cast<int>(output_size), 1),
                                  {{result, [r = a.shape()[0], c = b.shape()[1]](const Tensor &g)
                                    { return g.reshape(r, c); }}});
        return Array({a.shape()[0], b.shape()[1]}, flat);
    }
    Array softmax(const Array &a, int axis)
    {
        axis = axis_index(axis, a.shape());
        const int classes = a.shape()[axis];
        auto st = strides(a.shape());
        const std::size_t inner = st[axis], outer = a.size() / (classes * inner);
        Vector out = a.values();
        for (std::size_t n = 0; n < outer; ++n)
            for (std::size_t k = 0; k < inner; ++k)
            {
                std::size_t base = n * classes * inner + k;
                double maximum = -std::numeric_limits<double>::infinity();
                for (int c = 0; c < classes; ++c)
                    maximum = (std::max)(maximum, out[base + c * inner]);
                double total = 0;
                for (int c = 0; c < classes; ++c)
                {
                    out[base + c * inner] = std::exp(out[base + c * inner] - maximum);
                    total += out[base + c * inner];
                }
                if (!std::isfinite(total) || total <= 0)
                    throw std::invalid_argument("Non-finite softmax input");
                for (int c = 0; c < classes; ++c)
                    out[base + c * inner] /= total;
            }
        return operation(a.shape(), out,
                         {{a, [out, classes, inner, outer](const Vector &g)
                           {
                               Vector r(g.size());
                               for (std::size_t n = 0; n < outer; ++n)
                                   for (std::size_t k = 0; k < inner; ++k)
                                   {
                                       auto base = n * classes * inner + k;
                                       double dot = 0;
                                       for (int c = 0; c < classes; ++c)
                                           dot += g[base + c * inner] * out[base + c * inner];
                                       for (int c = 0; c < classes; ++c)
                                           r[base + c * inner] =
                                               out[base + c * inner] * (g[base + c * inner] - dot);
                                   }
                               return r;
                           }}});
    }
    Array mse(const Array &prediction, const Array &target)
    {
        if (prediction.shape() != target.shape())
            throw std::invalid_argument("MSE shape mismatch");
        auto d = add(prediction, scale(target, -1));
        return mean(multiply(d, d));
    }
    Array cross_entropy(const Array &a, const std::vector<int> &labels, int ignore_index)
    {
        if (a.shape().empty())
            throw std::invalid_argument("Cross entropy requires class dimension");
        const int classes = a.shape().back();
        const std::size_t count = a.size() / classes;
        if (labels.size() != count)
            throw std::invalid_argument("Label count mismatch");
        auto logits = a.values();
        Vector gradient(a.size(), 0);
        double total = 0;
        std::size_t valid = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            if (labels[i] == ignore_index)
                continue;
            if (labels[i] < 0 || labels[i] >= classes)
                throw std::invalid_argument("Invalid class label");
            auto start = logits.begin() + static_cast<std::ptrdiff_t>(i * classes);
            double maximum = *std::max_element(start, start + classes), denominator = 0;
            for (int c = 0; c < classes; ++c)
                denominator += std::exp(logits[i * classes + c] - maximum);
            if (!std::isfinite(maximum) || !std::isfinite(denominator))
                throw std::invalid_argument("Non-finite logits");
            total += maximum - logits[i * classes + labels[i]] + std::log(denominator);
            for (int c = 0; c < classes; ++c)
                gradient[i * classes + c] =
                    std::exp(logits[i * classes + c] - maximum) / denominator;
            gradient[i * classes + labels[i]] -= 1;
            ++valid;
        }
        if (!valid)
            throw std::invalid_argument("No unmasked labels");
        for (double &g : gradient)
            g /= static_cast<double>(valid);
        return operation({}, {total / valid},
                         {{a, [gradient](const Vector &g)
                           {
                               Vector r = gradient;
                               for (double &v : r)
                                   v *= g[0];
                               return r;
                           }}});
    }
} // namespace ml::nd
