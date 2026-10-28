#include "test_support.h"
#include "compute_device.h"

int main()
{
    return run_tests({
        {
            "force_cpu", []
            {
                const auto info = ml::initialize_compute(ml::Backend::CPU);
                require(info.selected == ml::Backend::CPU, "CPU non selectionne");
            }
        },
        {
            "backend_matches_build_configuration", []
            {
#if defined(CLARNI_EXPECT_CUDA)
                const auto info = ml::initialize_compute(ml::Backend::CPU);
                require(info.cuda_compiled == bool(CLARNI_EXPECT_CUDA),
                        "backend reel/stub incompatible avec configuration de compilation");
#else
                const auto info = ml::initialize_compute(ml::Backend::Auto);
                if (!info.cuda_compiled)
                    expect_throw<std::runtime_error>([] { ml::initialize_compute(ml::Backend::CUDA); });
#endif
            }
        }
    });
}
