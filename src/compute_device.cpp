#include "compute_device.h"
#include "gpu_backend.h"

#include <stdexcept>
#include <utility>

namespace ml
{
    namespace
    {
        DeviceInfo& state()
        {
            static DeviceInfo info = []
            {
                DeviceInfo initial;
                initial.cuda_compiled = gpu::compiled();
                initial.diagnostic = "Default CPU ; initialize_compute not called";
                return initial;
            }();
            return info;
        }
    }

    const DeviceInfo& compute_device() { return state(); }

    DeviceInfo initialize_compute(const Backend preference)
    {
        if (preference != Backend::Auto && preference != Backend::CPU
            && preference != Backend::CUDA)
            throw std::invalid_argument("Unknown backend");

        DeviceInfo next;
        next.cuda_compiled = gpu::compiled();
        if (preference == Backend::CPU)
        {
            next.diagnostic = "CPU asked ; CUDA detection not done";
        }
        else
        {
            const auto [usable, device, name, diagnostic] = gpu::probe();
            next.cuda_usable = usable;
            next.diagnostic = diagnostic;
            if (usable)
            {
                next.selected = Backend::CUDA;
                next.cuda_device = device;
                next.name = name;
            }
            else if (preference == Backend::CUDA)
            {
                throw std::runtime_error("CUDA asked but not available : "
                    + diagnostic);
            }
        }
        state() = std::move(next);
        return state();
    }
}
