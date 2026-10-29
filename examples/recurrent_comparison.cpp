#include "gated_recurrent.h"
#include <iostream>

int main()
{
    using namespace ml;
    const auto x = nd::Array::from({1, 4, 1}, {-1, 1, -1, 1});
    const auto target = nd::Array::from({1, 4, 1}, {1, -1, 1, -1});
    bool success = true;
    for (int type = 0; type < 3; ++type)
    {
        nn::Sequential model;
        if (type == 0)
            model.add(std::make_shared<nn::RNN>(1, 4));
        if (type == 1)
            model.add(std::make_shared<nn::GRU>(1, 4));
        if (type == 2)
            model.add(std::make_shared<nn::LSTM>(1, 4));
        model.add(std::make_shared<nn::Linear>(4, 1));
        OptimizerOptions opts;
        opts.kind = OptimizerKind::Adam;
        nn::Trainer trainer(model, opts);
        for (int i = 0; i < 150; ++i)
        {
            trainer.zero_grad();
            nd::mse(model.forward(x), target).backward();
            trainer.step(.03, 1);
        }
        const double loss = nd::mse(model.forward(x), target).item();
        std::cout << (type == 0 ? "RNN" : type == 1 ? "GRU" : "LSTM") << ": MSE=" << loss << '\n';
        success = success && loss < .01;
    }
    return success ? 0 : 1;
}
