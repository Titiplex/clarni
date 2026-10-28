#include "matrix.h"
#include "compute_device.h"
#include <exception>
#include <iostream>

int main()
{
    try
    {
        const auto info = ml::initialize_compute(ml::Backend::Auto);
        std::cout << info.name << " : " << info.diagnostic << '\n';
        ml::math::Matrix a(2, 3), b(3, 2);
        double next = 1;
        for (int i = 0; i < 2; ++i)for (int j = 0; j < 3; ++j)a(i, j) = next++;
        next = 7;
        for (int i = 0; i < 3; ++i)for (int j = 0; j < 2; ++j)b(i, j) = next++;
        const auto c = a * b;
        for (int i = 0; i < c.get_x(); ++i)
        {
            for (int j = 0; j < c.get_y(); ++j)std::cout << c(i, j) << ' ';
            std::cout << '\n';
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
