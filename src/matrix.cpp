#include "matrix.h"
#include "compute_device.h"
#include "gpu_backend.h"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace std;

namespace ml::math
{
    namespace
    {
        int checked_dim(const size_t size)
        {
            if (size > static_cast<size_t>((numeric_limits<int>::max)()))
                throw length_error("Vector too big");
            return static_cast<int>(size);
        }

        void elementwise_cpu(
            const double* a,
            const double* b,
            double* c,
            const size_t count,
            const gpu::Elementwise op,
            const double scalar
        )
        {
            for (size_t i = 0; i < count; ++i)
            {
                switch (op)
                {
                case gpu::Elementwise::Add: c[i] = a[i] + b[i];
                    break;
                case gpu::Elementwise::Subtract: c[i] = a[i] - b[i];
                    break;
                case gpu::Elementwise::Scale: c[i] = a[i] * scalar;
                    break;
                }
            }
        }

        void dispatch_elementwise(
            const double* a,
            const double* b,
            double* c,
            const size_t count,
            const gpu::Elementwise op,
            const double scalar
        )
        {
            if (count == 0) return;
            if (const auto& device = compute_device(); device.selected == Backend::CUDA)
            {
                gpu::elementwise(a, b, c, count, op, scalar, device.cuda_device);
            }
            else
            {
                elementwise_cpu(a, b, c, count, op, scalar);
            }
        }

        void dispatch_matmul(
            const double* a,
            const double* b,
            double* c,
            const int rows,
            const int inner,
            const int cols
        )
        {
            if (rows == 0 || cols == 0 || inner == 0) return;
            if (const auto& device = compute_device(); device.selected == Backend::CUDA)
            {
                gpu::matmul(a, b, c, rows, inner, cols, device.cuda_device);
            }
            else
            {
                for (int i = 0; i < rows; ++i)
                {
                    for (int j = 0; j < cols; ++j)
                    {
                        double sum = 0.0;
                        for (int k = 0; k < inner; ++k)
                            sum += a[static_cast<size_t>(i) * inner + k]
                                * b[static_cast<size_t>(k) * cols + j];
                        c[static_cast<size_t>(i) * cols + j] = sum;
                    }
                }
            }
        }
    }

    size_t Matrix::checked_size(const int rows, const int cols)
    {
        if (rows < 0 || cols < 0)
            throw invalid_argument("Negative dimensions");
        const auto r = static_cast<size_t>(rows);
        const auto c = static_cast<size_t>(cols);
        if (c != 0 && r > vector<double>().max_size() / c)
            throw length_error("Matrix too big");
        return r * c;
    }

    Matrix::Matrix(int rows, int cols) : Matrix(rows, cols, 0.0)
    {
    }

    Matrix::Matrix(int rows, int cols, const double value)
        : n_x(rows), n_y(cols), matrix(checked_size(rows, cols), value)
    {
    }

    size_t Matrix::index(const int i, const int j) const
    {
        if (i < 0 || j < 0 || i >= n_x || j >= n_y)
            throw out_of_range("Matrix index out of bounds");
        return static_cast<size_t>(i) * n_y + j;
    }

    double Matrix::operator()(const int i, const int j) const { return matrix[index(i, j)]; }
    double& Matrix::operator()(const int i, const int j) { return matrix[index(i, j)]; }

    bool Matrix::operator==(const Matrix& other) const
    {
        return n_x == other.n_x && n_y == other.n_y && matrix == other.matrix;
    }

    bool Matrix::operator!=(const Matrix& other) const { return !(*this == other); }

    void Matrix::require_same_shape(const Matrix& other) const
    {
        if (n_x != other.n_x || n_y != other.n_y)
            throw invalid_argument("Matrix dimensions not compatible");
    }

    Matrix Matrix::operator*(const Matrix& other) const
    {
        if (n_y != other.n_x)
            throw invalid_argument("Product : A.columns != B.lines");
        Matrix result(n_x, other.n_y);
        dispatch_matmul(matrix.data(), other.matrix.data(), result.matrix.data(),
                        n_x, n_y, other.n_y);
        return result;
    }

    Matrix Matrix::operator+(const Matrix& other) const
    {
        require_same_shape(other);
        Matrix result(n_x, n_y);
        dispatch_elementwise(matrix.data(), other.matrix.data(), result.matrix.data(),
                             matrix.size(), gpu::Elementwise::Add, 0.0);
        return result;
    }

    Matrix& Matrix::operator+=(const Matrix& other)
    {
        *this = *this + other;
        return *this;
    }

    Matrix Matrix::operator-(const Matrix& other) const
    {
        require_same_shape(other);
        Matrix result(n_x, n_y);
        dispatch_elementwise(matrix.data(), other.matrix.data(), result.matrix.data(),
                             matrix.size(), gpu::Elementwise::Subtract, 0.0);
        return result;
    }

    Matrix& Matrix::operator-=(const Matrix& other)
    {
        *this = *this - other;
        return *this;
    }

    Matrix Matrix::operator*(const double scalar) const
    {
        Matrix result(n_x, n_y);
        dispatch_elementwise(matrix.data(), nullptr, result.matrix.data(),
                             matrix.size(), gpu::Elementwise::Scale, scalar);
        return result;
    }

    Vector::Vector(const int dim) : Matrix(1, dim)
    {
    }

    Vector::Vector(const int dim, const double value) : Matrix(1, dim, value)
    {
    }

    Vector::Vector(const vector<double>& values)
        : Matrix(1, checked_dim(values.size()))
    {
        for (size_t i = 0; i < values.size(); ++i) (*this)[i] = values[i];
    }

    double Vector::operator[](size_t i) const
    {
        if (i >= size()) throw out_of_range("Vector index out of bounds");
        return operator()(0, static_cast<int>(i));
    }

    double& Vector::operator[](const size_t i)
    {
        if (i >= size()) throw out_of_range("Vector index out of bounds");
        return operator()(0, static_cast<int>(i));
    }

    Vector Vector::convert(const Matrix& m)
    {
        if (m.get_x() == 1)
        {
            Vector result(m.get_y());
            for (int i = 0; i < m.get_y(); ++i) result[i] = m(0, i);
            return result;
        }
        if (m.get_y() == 1)
        {
            Vector result(m.get_x());
            for (int i = 0; i < m.get_x(); ++i) result[i] = m(i, 0);
            return result;
        }
        throw invalid_argument("Conversion to Vector : matrice no line no column");
    }

    double Vector::norm() const
    {
        double result = 0.0;
        for (size_t i = 0; i < size(); ++i) result = hypot(result, (*this)[i]);
        return result;
    }

    Vector operator*(const Matrix& m, const Vector& v)
    {
        if (m.n_y != v.get_dim())
            throw invalid_argument("Product M*v : dimensions incompatibles");
        Vector result(m.n_x);
        dispatch_matmul(m.matrix.data(), v.matrix.data(), result.matrix.data(),
                        m.n_x, m.n_y, 1);
        return result;
    }
}
