#include "attention.h"
#include <algorithm>
#include <iostream>

int main()
{
    using namespace ml;
    nn::TransformerLM model(3, 8, 2, 16, 32);
    const nn::TokenBatch batch{{0, 1, 2, 0, 1, 2}, {1, 6}};
    const std::vector labels{1, 2, 0, 1, 2, 0};
    OptimizerOptions opts;
    opts.kind = OptimizerKind::Adam;
    nn::Trainer trainer(model, opts);
    const double before = nd::cross_entropy(model.logits(batch), labels).item();
    for (int i = 0; i < 100; ++i)
    {
        trainer.zero_grad();
        nd::cross_entropy(model.logits(batch), labels).backward();
        trainer.step(.02, 1);
    }
    const auto after = nd::cross_entropy(model.logits(batch), labels).item();
    std::cout << "Tiny Transformer: " << before << " -> " << after << "\n";
    model.eval();
    std::vector generated{0};
    for (int i = 0; i < 18; ++i)
    {
        auto logits = model.logits({generated, {1, static_cast<int>(generated.size())}}).values();
        auto last = logits.end() - 3;
        int token = static_cast<int>(std::max_element(last, logits.end()) - last);
        generated.push_back(token);
    }
    for (const int token : generated)
        std::cout << static_cast<char>('a' + token);
    std::cout << '\n';
    return after < .05 ? 0 : 1;
}
