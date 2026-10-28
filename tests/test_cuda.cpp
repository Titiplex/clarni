#include "test_support.h"
#include "matrix.h"
#include "compute_device.h"

int main()
{
    try
    {
        if (!ml::initialize_compute().cuda_usable)
        {
            std::cout << "SKIP CUDA indisponible\n";
            return 77;
        }
        ml::math::Matrix a(17, 33), b(33, 19);
        for (int i = 0; i < a.get_x(); ++i)for (int j = 0; j < a.get_y(); ++j)a(i, j) = (i - j) * 0.1;
        for (int i = 0; i < b.get_x(); ++i)for (int j = 0; j < b.get_y(); ++j)b(i, j) = (i + j) * 0.02;
        ml::initialize_compute(ml::Backend::CPU);
        const auto cpu = a * b;
        ml::initialize_compute(ml::Backend::CUDA);
        const auto gpu = a * b;
        for (int i = 0; i < cpu.get_x(); ++i)for (int j = 0; j < cpu.get_y(); ++j)near(gpu(i, j), cpu(i, j));
        std::cout << "PASS CPU/CUDA rectangular product\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
