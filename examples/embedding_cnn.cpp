#include "convolution.h"
#include "embedding.h"
#include <iostream>
// Fixed-length toy token sequences. A real dataset needs explicit padding masks
// and a Unicode-aware tokenizer; these IDs do not encode French phonology.
class TokenClassifier final : public ml::nn::Module
{
  public:
    ml::nn::Embedding embedding{6, 4};
    ml::nn::Conv1D convolution{4, 5, 2};
    ml::nn::GlobalMaxPool1D pooling;
    ml::nn::Linear classifier{5, 2};
    ml::nd::Array logits(const ml::nn::TokenBatch &batch)
    {
        auto x = ml::nd::permute(embedding.lookup(batch), {0, 2, 1});
        return classifier.forward(
            pooling.forward(ml::nd::activate(convolution.forward(x), ml::Activation::Relu)));
    }
    std::vector<ml::nn::NamedParameter> named_parameters() const override
    {
        std::vector<ml::nn::NamedParameter> out;
        for (auto &[prefix, module] : std::vector<std::pair<std::string, const ml::nn::Module *>>{
                 {"embedding.", &embedding}, {"conv.", &convolution}, {"classifier.", &classifier}})
            for (auto p : module->named_parameters())
            {
                p.name = prefix + p.name;
                out.push_back(std::move(p));
            }
        return out;
    }
};
int main()
{
    using namespace ml;
    TokenClassifier model;
    nn::TokenBatch batch{{0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 4, 5, 3, 4, 5}, {4, 5}};
    std::vector<int> labels{0, 0, 1, 1};
    OptimizerOptions opts;
    opts.kind = OptimizerKind::Adam;
    nn::Trainer trainer(model, opts);
    double before = nd::cross_entropy(model.logits(batch), labels).item();
    for (int i = 0; i < 100; ++i)
    {
        trainer.zero_grad();
        nd::cross_entropy(model.logits(batch), labels).backward();
        trainer.step(.02, 1);
    }
    double after = nd::cross_entropy(model.logits(batch), labels).item();
    std::cout << "Embedding + Conv1D: " << before << " -> " << after << '\n';
    return after < .05 ? 0 : 1;
}
