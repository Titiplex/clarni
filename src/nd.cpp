#include "nd.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace ml::nd
{
    std::size_t numel(const Shape& shape)
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
        Tensor pack(Vector values)
        {
            if (values.empty() ||
                values.size() > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
                throw std::length_error("Invalid flat storage size");
            const int count = static_cast<int>(values.size());
            return Tensor::from_values(count, 1, std::move(values));
        }

        Vector unpack(const Tensor& t)
        {
            if (t.backend() != Backend::CPU)
                return unpack(t.to(Backend::CPU));
            const auto values = t.cpu_values();
            return {values.begin(), values.end()};
        }

        int axis_index(int axis, const Shape& shape)
        {
            if (axis < 0)
                axis += static_cast<int>(shape.size());
            if (axis < 0 || axis >= static_cast<int>(shape.size()))
                throw std::invalid_argument("Invalid axis");
            return axis;
        }

        std::vector<std::size_t> strides(const Shape& s)
        {
            std::vector<std::size_t> st(s.size(), 1);
            for (std::size_t i = s.size(); i > 1; --i)
                st[i - 2] = st[i - 1] * s[i - 1];
            return st;
        }

        // out[i] = input[index[i]], backward scatter-add (including repeated indices).
        Array gather(const Array& a, Shape shape, std::vector<std::size_t> index)
        {
            const auto source = a.values();
            Vector out(index.size());
            for (std::size_t i = 0; i < index.size(); ++i)
                out[i] = source.at(index[i]);
            return operation(
                std::move(shape),
                std::move(out),
                {
                    {
                        a, [index, n = a.size()](const Vector& g)
                        {
                            Vector result(n, 0);
                            for (std::size_t i = 0; i < index.size(); ++i)
                                result[index[i]] += g[i];
                            return result;
                        }
                    }
                }
            );
        }

        Shape broadcast_shape(const Shape& a, const Shape& b)
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

        gpu::resident::NDDescriptor index_descriptor(const Shape& source, const Shape& target,
                                                     const int mode)
        {
            if (source.size() > gpu::resident::nd_max_rank ||
                target.size() > gpu::resident::nd_max_rank)
                throw std::invalid_argument("CUDA indexing supports ranks up to 8");
            gpu::resident::NDDescriptor d;
            d.rank = static_cast<int>(target.size());
            d.mode = mode;
            for (int j = 0; j < d.rank; ++j)
            {
                d.dst[j] = target[j];
                d.src[j] = 1;
            }
            const auto offset = target.size() - source.size();
            for (std::size_t j = 0; j < source.size(); ++j)
                d.src[j + offset] = source[j];
            return d;
        }

        Array device_index(const Array& a, Shape shape, const gpu::resident::NDDescriptor& d)
        {
            auto value = a.variable().value().nd_operation(gpu::resident::NDOperation::Index,
                                                           static_cast<int>(numel(shape)), 1, d);
            auto result = ad::operation(
                std::move(value),
                {
                    {
                        a.variable(), [d, n = static_cast<int>(a.size())](const Tensor& g)
                        {
                            return g.nd_operation(gpu::resident::NDOperation::IndexBackward, n, 1, d);
                        }
                    }
                });
            return Array(std::move(shape), std::move(result));
        }

        void same_backend(const Array& a, const Array& b)
        {
            if (a.backend() != b.backend())
                throw std::invalid_argument(
                    "N-D operands must share a backend; use to() explicitly");
        }
    } // namespace
    Array::Array(Shape shape, ad::Variable data) : shape_(std::move(shape)), data_(std::move(data))
    {
        if (data_.value().cols() != 1 || data_.value().size() != numel(shape_))
            throw std::invalid_argument("N-D storage mismatch");
    }

    Array::Array(const Shape& shape, const double fill, const bool requires_grad,
                 const Backend backend)
        : Array(shape, ad::Variable(Tensor(static_cast<int>(numel(shape)), 1, fill, backend),
                                    requires_grad))
    {
    }

    Array Array::from(Shape shape, const Vector& values, const bool requires_grad,
                      const Backend backend)
    {
        if (numel(shape) != values.size())
            throw std::invalid_argument("N-D value count mismatch");
        return Array(std::move(shape), ad::Variable(pack(values).to(backend), requires_grad));
    }

    Vector Array::values() const
    {
        return unpack(data_.value());
    }

    Vector Array::grad() const
    {
        return unpack(data_.grad());
    }

    Array operation(Shape shape, Vector values, const std::vector<std::pair<Array, VJP>>& parents)
    {
        if (numel(shape) != values.size())
            throw std::invalid_argument("Operation shape mismatch");
        for (const auto& key : parents | std::views::keys)
            if (key.backend() != Backend::CPU)
                throw std::invalid_argument(
                    "CPU VJP hook cannot process CUDA arrays; implement a resident pullback");
        if (!ad::grad_enabled())
            return Array(std::move(shape), ad::Variable(pack(std::move(values)), false));
        std::vector<std::pair<ad::Variable, ad::Pullback>> edges;
        for (const auto& [parent, vjp] : parents)
            edges.emplace_back(
                parent.variable(),
                [vjp, n = parent.size()](const Tensor& g)
                {
                    auto result = vjp(unpack(g));
                    if (result.size() != n)
                        throw std::logic_error("N-D pullback size mismatch");
                    return pack(std::move(result));
                });
        return Array(std::move(shape), ad::operation(pack(std::move(values)), edges));
    }

    Array reshape(const Array& a, Shape shape)
    {
        if (numel(shape) != a.size())
            throw std::invalid_argument("reshape size mismatch");
        return Array(std::move(shape), a.variable());
    }

    Array permute(const Array& a, const std::vector<int>& axes)
    {
        if (axes.size() != a.shape().size())
            throw std::invalid_argument("Permutation rank mismatch");
        Shape shape(axes.size());
        std::vector seen(axes.size(), false);
        for (std::size_t i = 0; i < axes.size(); ++i)
        {
            const int ax = axis_index(axes[i], a.shape());
            if (seen[ax])
                throw std::invalid_argument("Repeated permutation axis");
            seen[ax] = true;
            shape[i] = a.shape()[ax];
        }
        if (a.backend() == Backend::CUDA)
        {
            auto d = index_descriptor(a.shape(), shape, 1);
            for (std::size_t j = 0; j < axes.size(); ++j)
                d.axes[j] = axis_index(axes[j], a.shape());
            return device_index(a, std::move(shape), d);
        }
        const auto src = strides(a.shape());
        const auto dst = strides(shape);
        std::vector<std::size_t> index(a.size(), 0);
        for (std::size_t i = 0; i < index.size(); ++i)
            for (std::size_t j = 0; j < shape.size(); ++j)
                index[i] += (i / dst[j] % shape[j]) * src[axis_index(axes[j], a.shape())];
        return gather(a, std::move(shape), std::move(index));
    }

    Array slice(const Array& a, int axis, const int first, const int count)
    {
        axis = axis_index(axis, a.shape());
        if (first < 0 || count <= 0 || first > a.shape()[axis] - count)
            throw std::invalid_argument("Invalid slice");
        Shape shape = a.shape();
        shape[axis] = count;
        if (a.backend() == Backend::CUDA)
        {
            auto d = index_descriptor(a.shape(), shape, 2);
            d.axis = axis;
            d.first = first;
            return device_index(a, std::move(shape), d);
        }
        const auto src = strides(a.shape());
        const auto dst = strides(shape);
        std::vector index(numel(shape), static_cast<std::size_t>(first) * src[axis]);
        for (std::size_t i = 0; i < index.size(); ++i)
            for (std::size_t j = 0; j < shape.size(); ++j)
                index[i] += (i / dst[j] % shape[j]) * src[j];
        return gather(a, std::move(shape), std::move(index));
    }

    Array concat(const std::vector<Array>& arrays, int axis)
    {
        if (arrays.empty())
            throw std::invalid_argument("Empty concat");
        for (const auto& a : arrays)
            if (a.backend() != Backend::CPU)
                throw std::invalid_argument(
                    "CUDA concat is not implemented in this CNN-focused release");
        Shape shape = arrays.front().shape();
        axis = axis_index(axis, shape);
        std::size_t axis_size = 0;
        for (const auto& a : arrays)
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
        const auto dst = strides(shape);
        std::vector<std::pair<Array, VJP>> parents;
        int offset = 0;
        for (const auto& a : arrays)
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
                                 [index](const Vector& g)
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

    Array broadcast_to(const Array& a, Shape shape)
    {
        if (a.shape() == shape)
            return a;
        if (shape.size() < a.shape().size() || broadcast_shape(a.shape(), shape) != shape)
            throw std::invalid_argument("Invalid broadcast target");
        if (a.backend() == Backend::CUDA)
            return device_index(a, shape, index_descriptor(a.shape(), shape, 0));
        const auto src = strides(a.shape());
        const auto dst = strides(shape);
        std::vector<std::size_t> index(numel(shape), 0);
        const auto shift = shape.size() - a.shape().size();
        for (std::size_t i = 0; i < index.size(); ++i)
            for (std::size_t j = 0; j < a.shape().size(); ++j)
                if (a.shape()[j] != 1)
                    index[i] += (i / dst[j + shift] % shape[j + shift]) * src[j];
        return gather(a, std::move(shape), std::move(index));
    }

    Array add(const Array& a, const Array& b)
    {
        same_backend(a, b);
        const auto shape = broadcast_shape(a.shape(), b.shape());
        const auto x = broadcast_to(a, shape);
        const auto y = broadcast_to(b, shape);
        return Array(shape, ad::add(x.variable(), y.variable()));
    }

    Array multiply(const Array& a, const Array& b)
    {
        same_backend(a, b);
        const auto shape = broadcast_shape(a.shape(), b.shape());
        const auto x = broadcast_to(a, shape);
        const auto y = broadcast_to(b, shape);
        return Array(shape, ad::hadamard(x.variable(), y.variable()));
    }

    Array scale(const Array& a, const double factor, const double offset)
    {
        return Array(a.shape(), ad::affine(a.variable(), factor, offset));
    }

    Array activate(const Array& a, const Activation activation)
    {
        if (activation == Activation::Softmax)
            return softmax(a);
        return Array(a.shape(), ad::activate(a.variable(), activation));
    }

    Array sum(const Array& a)
    {
        return Array({}, ad::sum(a.variable()));
    }

    Array mean(const Array& a)
    {
        return scale(sum(a), 1.0 / static_cast<double>(a.size()));
    }

    Array reduce_sum(const Array& a, int axis, const bool keepdim)
    {
        axis = axis_index(axis, a.shape());
        Shape shape = a.shape();
        shape[axis] = 1;
        if (a.backend() == Backend::CUDA)
        {
            std::vector<int> axes;
            for (int j = 0; j < static_cast<int>(a.shape().size()); ++j)
                if (j != axis)
                    axes.push_back(j);
            axes.push_back(axis);
            auto p = permute(a, axes);
            int columns = a.shape()[axis], rows = static_cast<int>(a.size() / columns);
            auto view = p.variable().value().reshape(rows, columns);
            auto out = ad::operation(
                view.sum_columns(),
                {
                    {
                        p.variable(), [columns, n = static_cast<int>(a.size())](const Tensor& g)
                        {
                            return g.broadcast_columns(columns).reshape(n, 1);
                        }
                    }
                });
            if (!keepdim)
                shape.erase(shape.begin() + axis);
            return Array(shape, out);
        }
        const auto dst = strides(shape);
        const auto src = strides(a.shape());
        std::vector<std::size_t> index(a.size(), 0);
        Vector out(numel(shape), 0);
        const Vector v = a.values();
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            for (std::size_t j = 0; j < shape.size(); ++j)
                if (static_cast<int>(j) != axis)
                    index[i] += (i / src[j] % a.shape()[j]) * dst[j];
            out[index[i]] += v[i];
        }
        if (!keepdim)
            shape.erase(shape.begin() + axis);
        return operation(
            shape,
            std::move(out),
            {
                {
                    a, [index](const Vector& g)
                    {
                        Vector r(index.size());
                        for (std::size_t i = 0; i < r.size(); ++i)
                            r[i] = g[index[i]];
                        return r;
                    }
                }
            }
        );
    }

    Array matmul(const Array& a, const Array& b)
    {
        same_backend(a, b);
        if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0])
            throw std::invalid_argument("matmul expects compatible rank-2 arrays");
        auto view = [](const Array& x)
        {
            const auto& v = x.variable();
            return ad::operation(
                v.value().reshape(
                    x.shape()[0],
                    x.shape()[1]
                ),
                {
                    {
                        v, [n = x.size()](const Tensor& g)
                        {
                            return g.reshape(static_cast<int>(n), 1);
                        }
                    }
                }
            );
        };
        const Shape output_shape = {a.shape()[0], b.shape()[1]};
        const auto output_size = numel(output_shape);
        auto result = ad::matmul(view(a), view(b));
        const auto flat =
            ad::operation(
                result.value().reshape(
                    static_cast<int>(output_size),
                    1
                ),
                {
                    {
                        result, [r = a.shape()[0], c = b.shape()[1]](const Tensor& g)
                        {
                            return g.reshape(r, c);
                        }
                    }
                });
        return Array({a.shape()[0], b.shape()[1]}, flat);
    }

    Array batched_matmul(const Array& a, const Array& b)
    {
        if (a.backend() != Backend::CPU || b.backend() != Backend::CPU)
            throw std::invalid_argument(
                "CUDA batched matmul is not implemented in this CNN-focused release");
        same_backend(a, b);
        if (a.shape().size() < 2 || b.shape().size() < 2 ||
            a.shape().back() != b.shape()[b.shape().size() - 2])
            throw std::invalid_argument("Batched matmul shape mismatch");
        Shape ap(a.shape().begin(), a.shape().end() - 2),
              bp(b.shape().begin(), b.shape().end() - 2);
        Shape prefix = broadcast_shape(ap, bp);
        const int m = a.shape()[a.shape().size() - 2], k = a.shape().back(), n = b.shape().back();
        Shape shape = prefix;
        shape.push_back(m);
        shape.push_back(n);
        const auto output_size = numel(shape), batches = numel(prefix);
        const auto ps = strides(prefix), as = strides(ap), bs = strides(bp);
        std::vector<std::size_t> ai(batches, 0), bi(batches, 0);
        for (std::size_t batch = 0; batch < batches; ++batch)
        {
            for (std::size_t j = 0; j < ap.size(); ++j)
                if (ap[j] != 1)
                    ai[batch] += (batch / ps[j + prefix.size() - ap.size()] % ap[j]) * as[j];
            for (std::size_t j = 0; j < bp.size(); ++j)
                if (bp[j] != 1)
                    bi[batch] += (batch / ps[j + prefix.size() - bp.size()] % bp[j]) * bs[j];
            ai[batch] *= static_cast<std::size_t>(m) * k;
            bi[batch] *= static_cast<std::size_t>(k) * n;
        }
        const auto x = a.values(), y = b.values();
        Vector out(output_size, 0);
        for (std::size_t batch = 0; batch < batches; ++batch)
            for (int row = 0; row < m; ++row)
                for (int z = 0; z < k; ++z)
                    for (int col = 0; col < n; ++col)
                        out[(batch * m + row) * n + col] +=
                            x[ai[batch] + static_cast<std::size_t>(row) * k + z] *
                            y[bi[batch] + static_cast<std::size_t>(z) * n + col];
        return operation(
            shape, std::move(out),
            {
                {
                    a,
                    [ai, bi, y, m, k, n, size = a.size()](const Vector& g)
                    {
                        Vector r(size, 0);
                        for (std::size_t batch = 0; batch < ai.size(); ++batch)
                            for (int row = 0; row < m; ++row)
                                for (int z = 0; z < k; ++z)
                                    for (int col = 0; col < n; ++col)
                                        r[ai[batch] + static_cast<std::size_t>(row) * k + z] +=
                                            g[(batch * m + row) * n + col] *
                                            y[bi[batch] + static_cast<std::size_t>(z) * n + col];
                        return r;
                    }
                },
                {
                    b, [ai, bi, x, m, k, n, size = b.size()](const Vector& g)
                    {
                        Vector r(size, 0);
                        for (std::size_t batch = 0; batch < ai.size(); ++batch)
                            for (int row = 0; row < m; ++row)
                                for (int z = 0; z < k; ++z)
                                    for (int col = 0; col < n; ++col)
                                        r[bi[batch] + static_cast<std::size_t>(z) * n + col] +=
                                            x[ai[batch] + static_cast<std::size_t>(row) * k + z] *
                                            g[(batch * m + row) * n + col];
                        return r;
                    }
                }
            });
    }

    Array softmax(const Array& a, int axis)
    {
        axis = axis_index(axis, a.shape());
        if (a.backend() == Backend::CUDA)
        {
            std::vector<int> axes{axis}, inverse(a.shape().size());
            for (int j = 0; j < static_cast<int>(a.shape().size()); ++j)
                if (j != axis)
                    axes.push_back(j);
            for (std::size_t j = 0; j < axes.size(); ++j)
                inverse[axes[j]] = static_cast<int>(j);
            auto p = permute(a, axes);
            int rows = a.shape()[axis], cols = static_cast<int>(a.size() / rows),
                n = static_cast<int>(a.size());
            auto view =
                ad::operation(p.variable().value().reshape(rows, cols),
                              {{p.variable(), [n](const Tensor& g) { return g.reshape(n, 1); }}});
            auto activated = ad::activate(view, Activation::Softmax);
            auto flat = ad::operation(
                activated.value().reshape(n, 1),
                {{activated, [rows, cols](const Tensor& g) { return g.reshape(rows, cols); }}});
            return permute(Array(p.shape(), flat), inverse);
        }
        const int classes = a.shape()[axis];
        const auto st = strides(a.shape());
        const std::size_t inner = st[axis], outer = a.size() / (classes * inner);
        Vector out = a.values();
        for (std::size_t n = 0; n < outer; ++n)
            for (std::size_t k = 0; k < inner; ++k)
            {
                const std::size_t base = n * classes * inner + k;
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
                         {
                             {
                                 a, [out, classes, inner, outer](const Vector& g)
                                 {
                                     Vector r(g.size());
                                     for (std::size_t n = 0; n < outer; ++n)
                                         for (std::size_t k = 0; k < inner; ++k)
                                         {
                                             const auto base = n * classes * inner + k;
                                             double dot = 0;
                                             for (int c = 0; c < classes; ++c)
                                                 dot += g[base + c * inner] * out[base + c * inner];
                                             for (int c = 0; c < classes; ++c)
                                                 r[base + c * inner] =
                                                     out[base + c * inner] * (g[base + c * inner] - dot);
                                         }
                                     return r;
                                 }
                             }
                         });
    }

    Array mse(const Array& prediction, const Array& target)
    {
        if (prediction.shape() != target.shape())
            throw std::invalid_argument("MSE shape mismatch");
        const auto d = add(prediction, scale(target, -1));
        return mean(multiply(d, d));
    }

    Array cross_entropy(const Array& a, const std::vector<int>& labels, const int ignore_index)
    {
        if (a.shape().empty())
            throw std::invalid_argument("Cross entropy requires class dimension");
        const int classes = a.shape().back();
        const std::size_t count = a.size() / classes;
        if (labels.size() != count)
            throw std::invalid_argument("Label count mismatch");
        if (a.backend() == Backend::CUDA)
        {
            std::size_t valid = 0;
            for (int label : labels)
                if (label != ignore_index)
                {
                    if (label < 0 || label >= classes)
                        throw std::invalid_argument("Invalid class label");
                    ++valid;
                }
            if (!valid)
                throw std::invalid_argument("No unmasked labels");
            gpu::resident::NDDescriptor d;
            d.samples = static_cast<int>(count);
            d.classes = classes;
            d.valid = static_cast<int>(valid);
            d.ignore = ignore_index;
            Vector ids(labels.begin(), labels.end());
            auto target =
                Tensor::from_values(static_cast<int>(count), 1, std::move(ids), Backend::CUDA);
            const auto logits = a.variable().value();
            auto value = logits
                         .nd_operation(gpu::resident::NDOperation::CrossEntropy,
                                       static_cast<int>(count), 1, d, &target)
                         .sum_rows()
                         .affine(1.0 / valid);
            if (!ad::grad_enabled() || !a.variable().requires_grad())
                return Array({}, ad::Variable(value, false));
            auto result = ad::operation(
                std::move(value),
                {
                    {
                        a.variable(), [d, target, logits, n = static_cast<int>(a.size())](const Tensor& g)
                        {
                            return logits
                                   .nd_operation(gpu::resident::NDOperation::CrossEntropyGradient, n, 1, d,
                                                 &target)
                                   .hadamard(g.broadcast_rows(n));
                        }
                    }
                });
            return Array({}, std::move(result));
        }
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
            const double maximum = *std::max_element(start, start + classes);
            double denominator = 0;
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
        for (double& g : gradient)
            g /= static_cast<double>(valid);
        return operation(
            {},
            {total / valid},
            {
                {
                    a, [gradient](const Vector& g)
                    {
                        Vector r = gradient;
                        for (double& v : r)
                            v *= g[0];
                        return r;
                    }
                }
            }
        );
    }
} // namespace ml::nd
