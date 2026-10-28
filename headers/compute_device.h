#pragma once

#include <string>

namespace ml
{
    enum class Backend { Auto, CPU, CUDA };

    struct DeviceInfo
    {
        Backend selected = Backend::CPU;
        bool cuda_compiled = false;
        bool cuda_usable = false;
        int cuda_device = -1;
        std::string name = "CPU";
        std::string diagnostic;
    };

    // Initialiser avant les threads ; ne pas reconfigurer pendant un calcul.
    DeviceInfo initialize_compute(Backend preference = Backend::Auto);
    const DeviceInfo& compute_device();
}
