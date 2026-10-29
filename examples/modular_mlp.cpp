#include "module.h"
#include <iostream>

int main()
{
    using namespace ml;
    nn::Sequential model;
    model.add(std::make_shared<nn::Linear>(2, 8));
    model.add(std::make_shared<nn::ActivationLayer>(Activation::Tanh));
    model.add(std::make_shared<nn::Linear>(8, 2));
    const auto x = nd::Array::from({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    const std::vector labels{0, 1, 1, 0};
    OptimizerOptions opts;
    opts.kind = OptimizerKind::Adam;
    nn::Trainer trainer(model, opts);
    for (int i = 0; i < 300; ++i)
    {
        trainer.zero_grad();
        nd::cross_entropy(model.forward(x), labels).backward();
        trainer.step(.03);
    }
    const double loss = nd::cross_entropy(model.forward(x), labels).item();
    std::cout << "Modular XOR: cross entropy=" << loss << '\n';
    return loss < .01 ? 0 : 1;
}
