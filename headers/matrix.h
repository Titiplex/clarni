#pragma once

#include <cstddef>
#include <vector>

namespace ml::math
{
    class Vector;

    class Matrix
    {
        int n_x;
        int n_y;
        std::vector<double> matrix;

        static std::size_t checked_size(int rows, int cols);
        [[nodiscard]] std::size_t index(int i, int j) const;
        void require_same_shape(const Matrix& other) const;
        friend Vector operator*(const Matrix& m, const Vector& v);

    public:
        Matrix(int n_x, int n_y);
        Matrix(int n_x, int n_y, double value);
        ~Matrix() = default;

        [[nodiscard]] int get_x() const { return n_x; }
        [[nodiscard]] int get_y() const { return n_y; }

        double operator()(int i, int j) const;
        double& operator()(int i, int j);

        bool operator==(const Matrix& other) const;
        bool operator!=(const Matrix& other) const;

        Matrix operator*(const Matrix& other) const;
        Matrix operator+(const Matrix& other) const;
        Matrix& operator+=(const Matrix& other);
        Matrix operator-(const Matrix& other) const;
        Matrix& operator-=(const Matrix& other);
        Matrix operator*(double scalar) const;
    };

    class Vector : public Matrix
    {
    public:
        explicit Vector(int dim);
        Vector(int dim, double value);
        explicit Vector(const std::vector<double>& values);

        Vector(const Vector&) = default;
        Vector(Vector&&) = default;
        Vector& operator=(const Vector&) = default;
        Vector& operator=(Vector&&) = default;

        double operator[](std::size_t i) const;
        double& operator[](std::size_t i);

        [[nodiscard]] int get_dim() const { return get_y(); }

        [[nodiscard]] std::size_t size() const
        {
            return static_cast<std::size_t>(get_dim());
        }

        static Vector convert(const Matrix& m);
        [[nodiscard]] double norm() const;
    };

    Vector operator*(const Matrix& m, const Vector& v);
}
