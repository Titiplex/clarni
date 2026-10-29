#include "module.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
    using Clock = std::chrono::steady_clock;

    double measure(ml::nn::Module& model, const ml::nd::Array& input, int iterations, bool no_grad)
    {
        auto run = [&]
        {
            double checksum = 0;
            for (int i = 0; i < iterations; ++i)
            {
                auto y = model.forward(input);
                checksum += y.variable().value().cpu_values()[0];
            }
            if (!std::isfinite(checksum))
                throw std::runtime_error("Invalid benchmark result");
        };
        const auto start = Clock::now();
        if (no_grad)
        {
            ml::ad::NoGradGuard guard;
            run();
        }
        else
            run();
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count() / iterations;
    }
} // namespace
int main()
{
    using namespace ml;
    nn::Sequential model;
    model.add(std::make_shared<nn::Linear>(64, 64));
    model.add(std::make_shared<nn::ActivationLayer>(Activation::Tanh));
    model.add(std::make_shared<nn::Linear>(64, 10));
    model.eval();
    nd::Vector data(32 * 64);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<double>(i % 23) / 23;
    const auto input = nd::Array::from({32, 64}, data);
    const auto expected = model.forward(input).values();
    {
        ad::NoGradGuard guard;
        auto actual = model.forward(input);
        if (actual.values() != expected || actual.variable().requires_grad())
            return 1;
    }
    measure(model, input, 5, false);
    measure(model, input, 5, true);
    std::cout << "CPU float64; batch=32; Linear(64,64)-Tanh-Linear(64,10); 100 iterations\n";
    std::cout << "recording_ms_per_batch=" << measure(model, input, 100, false)
        << "\nno_grad_ms_per_batch=" << measure(model, input, 100, true) << '\n';
    return 0;
}
