#include "embedding.h"
#include <iostream>
int main()
{
    using namespace ml;
    const auto words = nn::whitespace_tokens("chat animal chien animal chat mange chien mange "
                                             "pomme fruit poire fruit pomme sucre poire sucre");
    nn::Vocabulary vocab(words);
    auto pairs = nn::skipgram_pairs(vocab.encode(words), 1);
    nn::SkipGram model(vocab.size(), 8);
    OptimizerOptions opts;
    opts.kind = OptimizerKind::Adam;
    nn::Trainer trainer(model, opts);
    double before = nd::cross_entropy(model.logits(pairs.centers), pairs.contexts).item();
    for (int epoch = 0; epoch < 150; ++epoch)
    {
        trainer.zero_grad();
        auto loss = nd::cross_entropy(model.logits(pairs.centers), pairs.contexts);
        loss.backward();
        trainer.step(.03, 5);
    }
    auto after = nd::cross_entropy(model.logits(pairs.centers), pairs.contexts).item();
    std::cout << "Full-softmax SkipGram: " << before << " -> " << after
              << "\nToy corpus; semantic quality requires a real corpus.\n";
    return after < before ? 0 : 1;
}
