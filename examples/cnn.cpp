#include "convolution.h"
#include <iostream>

int main()
{
    using namespace ml;
    nn::Sequential model;
    model.add(std::make_shared<nn::Conv1D>(1, 4, 3));
    model.add(std::make_shared<nn::ActivationLayer>(Activation::Relu));
    model.add(std::make_shared<nn::GlobalMaxPool1D>());
    model.add(std::make_shared<nn::Linear>(4, 2));
    const auto x = nd::Array::from({4, 1, 5},
                                   {1, 0, 1, 0, 0, 0, 1, 0, 1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0});
    const std::vector labels{0, 0, 1, 1};
    OptimizerOptions opts;
    opts.kind = OptimizerKind::Adam;
    nn::Trainer trainer(model, opts);
    const double before = nd::cross_entropy(model.forward(x), labels).item();
    for (int i = 0; i < 120; ++i)
    {
        trainer.zero_grad();
        nd::cross_entropy(model.forward(x), labels).backward();
        trainer.step(.02);
    }
    const double after = nd::cross_entropy(model.forward(x), labels).item();
    std::cout << "Conv1D classification: " << before << " -> " << after << '\n';
    return after < .05 ? 0 : 1;
}
