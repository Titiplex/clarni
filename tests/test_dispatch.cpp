#include "test_support.h"
#include "matrix.h"
#include "compute_device.h"
#include "gpu_backend.h"

// Faux backend uniquement pour verifier le routage, sans CUDA ni GPU.
namespace mock
{
    bool available = false, fail = false;
    int probes = 0, calls = 0;
    constexpr double sentinel = 123456.0;
}

namespace ml::gpu
{
    bool compiled() noexcept { return true; }

    ProbeResult probe()
    {
        ++mock::probes;
        return {mock::available, mock::available ? 3 : -1, "mock GPU", "mock probe"};
    }

    void matmul(const double*, const double*, double* c, int rows, int, int cols, int device)
    {
        ++mock::calls;
        require(device == 3, "incorrect device");
        if (mock::fail) throw std::runtime_error("mock GPU failure");
        std::fill(c, c + static_cast<std::size_t>(rows) * cols, mock::sentinel);
    }

    void elementwise(const double*, const double*, double* c, std::size_t count,
                     Elementwise, double, int device)
    {
        ++mock::calls;
        require(device == 3, "incorrect device");
        if (mock::fail) throw std::runtime_error("mock GPU failure");
        std::fill(c, c + count, mock::sentinel);
    }
}

int main()
{
    using namespace ml;
    using namespace ml::math;
    return run_tests({
        {
            "cpu_does_not_call_gpu", []
            {
                initialize_compute(Backend::CPU);
                near((Matrix(2, 3, 1) * Matrix(3, 2, 2))(0, 0), 6);
                require(mock::calls == 0 && mock::probes == 0, "CPU called mock GPU");
            }
        },
        {
            "automatic_fallback_and_strict_cuda", []
            {
                mock::available = false;
                require(initialize_compute().selected == Backend::CPU, "no CPU fallback");
                expect_throw<std::runtime_error>([] { initialize_compute(Backend::CUDA); });
            }
        },
        {
            "one_gpu_call_per_operation_and_no_recompute_on_error", []
            {
                mock::available = true;
                initialize_compute();
                const int probes = mock::probes, calls = mock::calls;
                Matrix a(2, 3, 1), b(3, 2, 2);
                near((a * b)(0, 0), mock::sentinel);
                near((a + a)(0, 0), mock::sentinel);
                near((a - a)(0, 0), mock::sentinel);
                near((a * 2.0)(0, 0), mock::sentinel);
                near((a * Vector(3, 1))[0], mock::sentinel);
                require(mock::calls == calls + 5, "wrong GPU call count");
                require(mock::probes == probes, "probe repeated per operation");
                mock::fail = true;
                expect_throw<std::runtime_error>([&] { (void)(a * b); });
                require(compute_device().selected == Backend::CUDA, "hidden CPU retry");
            }
        }
    });
}
