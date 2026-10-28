#include "test_support.h"
#include "matrix.h"
#include "compute_device.h"

using namespace ml::math;

int main()
{
    ml::initialize_compute(ml::Backend::CPU);
    return run_tests({
        {
            "matrix_rectangular_product", []
            {
                Matrix a(2, 3), b(3, 2);
                double x = 1;
                for (int i = 0; i < 2; ++i)for (int j = 0; j < 3; ++j)a(i, j) = x++;
                x = 7;
                for (int i = 0; i < 3; ++i)for (int j = 0; j < 2; ++j)b(i, j) = x++;
                const auto c = a * b;
                require(c.get_x() == 2 && c.get_y() == 2, "mauvaises dimensions");
                near(c(0, 0), 58);
                near(c(0, 1), 64);
                near(c(1, 0), 139);
                near(c(1, 1), 154);
            }
        },
        {
            "matrix_arithmetic_and_aliasing", []
            {
                Matrix a(2, 3, 2), b(2, 3, 5);
                near((a + b)(0, 0), 7);
                near((a - b)(1, 2), -3);
                near((a * 3)(1, 1), 6);
                Matrix& ref = (a += a);
                require(&ref == &a, "+= doit renvoyer une reference");
                near(a(0, 0), 4);
                a -= b;
                near(a(0, 0), -1);
            }
        },
        {
            "matrix_indices_and_errors", []
            {
                Matrix a(2, 3);
                a(0, 0) = 4;
                near(a(0, 0), 4);
                expect_throw<std::out_of_range>([&] { (void)a(-1, 0); });
                expect_throw<std::out_of_range>([&] { (void)a(2, 0); });
                expect_throw<std::invalid_argument>([] { Matrix invalid(-1, 3); });
                expect_throw<std::invalid_argument>([&] { (void)(a * a); });
            }
        },
        {
            "matrix_empty_inner_dimension", []
            {
                const auto c = Matrix(2, 0) * Matrix(0, 3);
                require(c.get_x() == 2 && c.get_y() == 3, "mauvaises dimensions");
                near(c(1, 2), 0);
            }
        },
        {
            "vector_copy_and_assignment", []
            {
                Vector v(std::vector<double>{1, 2, 3}), w(1);
                w = v;
                w = w;
                near(w[0], 1);
                near(w[1], 2);
                near(w[2], 3);
                w[0] = 99;
                near(v[0], 1);
            }
        },
        {
            "vector_conversion", []
            {
                Matrix row(1, 3), col(3, 1);
                for (int i = 0; i < 3; ++i)
                {
                    row(0, i) = i + 1;
                    col(i, 0) = i + 1;
                }
                require(Vector::convert(row) == Vector::convert(col), "conversion incorrecte");
                expect_throw<std::invalid_argument>([] { (void)Vector::convert(Matrix(2, 2)); });
            }
        },
        {
            "matrix_vector_product", []
            {
                Matrix a(2, 3);
                double x = 1;
                for (int i = 0; i < 2; ++i)for (int j = 0; j < 3; ++j)a(i, j) = x++;
                const auto v = a * Vector(std::vector<double>{1, 2, 3});
                require(v.get_dim() == 2, "mauvaise dimension");
                near(v[0], 14);
                near(v[1], 32);
            }
        },
        {
            "vector_norm", []
            {
                near(Vector(std::vector<double>{3, 4}).norm(), 5);
                near(Vector(0).norm(), 0);
            }
        }
    });
}
