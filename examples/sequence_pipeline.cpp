#include "gated_recurrent.h"
#include "training.h"
#include <iostream>
class TokenModel final : public ml::nn::Module
{
  public:
    ml::nn::Embedding embedding{4, 8, 0};
    ml::nn::GRU recurrent{8, 12};
    ml::nn::Linear head{12, 4};
    ml::nd::Array logits(const ml::data::Batch &batch)
    {
        return head.forward(
            recurrent
                .run(embedding.lookup(std::get<ml::nn::TokenBatch>(batch.input)), batch.lengths)
                .sequence);
    }
    std::vector<ml::nn::NamedParameter> named_parameters() const override
    {
        std::vector<ml::nn::NamedParameter> out;
        for (auto &[prefix, module] : std::vector<std::pair<std::string, const ml::nn::Module *>>{
                 {"embedding.", &embedding}, {"recurrent.", &recurrent}, {"head.", &head}})
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
    auto data = std::make_shared<data::VectorDataset>(
        std::vector<data::Example>{{std::vector<int>{1, 2, 3, 1}, {2, 3, 1, 2}},
                                   {std::vector<int>{2, 3}, {3, 1}},
                                   {std::vector<int>{3, 1, 2}, {1, 2, 3}}});
    auto collator =
        data::make_collator({data::InputLayout::PadTokens, data::TargetLayout::Sequence, 0, -1});
    data::DataLoader train(data, {2, true, false, 42, "toy-sequences"}, collator),
        valid(data, {2, false, false, 42, "toy-sequences"}, collator);
    TokenModel model;
    OptimizerOptions options;
    options.kind = OptimizerKind::Adam;
    training::SupervisedTrainer trainer(model, options);
    auto forward = [&](const data::Batch &b) { return model.logits(b); };
    training::FitOptions fit;
    fit.classes = 4;
    fit.epochs = 30;
    fit.learning_rate = .03;
    fit.max_grad_norm = 1;
    trainer.fit(train, valid, forward, fit);
    auto metrics = trainer.evaluate(valid, forward, 4);
    std::cout << "Synthetic variable-length sequences: loss=" << metrics.loss
              << " perplexity=" << metrics.perplexity << " accuracy=" << metrics.accuracy
              << " valid_tokens=" << metrics.count
              << "\nThis is an integration demo, not a held-out language benchmark.\n";
    return metrics.accuracy == 1 && metrics.loss < .05 && metrics.count == 9 ? 0 : 1;
}
