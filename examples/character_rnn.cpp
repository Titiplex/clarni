#include "embedding.h"
#include "recurrent.h"
#include <algorithm>
#include <iostream>
class CharacterModel final : public ml::nn::Module
{
  public:
    ml::nn::Embedding embedding{3, 4};
    ml::nn::RNN rnn{4, 8};
    ml::nn::Linear head{8, 3};
    ml::nd::Array logits(const std::vector<int> &ids, int length)
    {
        return head.forward(rnn.forward(embedding.lookup({ids, {1, length}})));
    }
    std::vector<ml::nn::NamedParameter> named_parameters() const override
    {
        std::vector<ml::nn::NamedParameter> out;
        for (auto &[prefix, module] : std::vector<std::pair<std::string, const ml::nn::Module *>>{
                 {"embedding.", &embedding}, {"rnn.", &rnn}, {"head.", &head}})
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
    CharacterModel model;
    OptimizerOptions opts;
    opts.kind = OptimizerKind::Adam;
    nn::Trainer trainer(model, opts);
    std::vector<int> input{0, 1, 2, 0, 1, 2}, target{1, 2, 0, 1, 2, 0};
    double before = nd::cross_entropy(model.logits(input, 6), target).item();
    for (int i = 0; i < 100; ++i)
    {
        trainer.zero_grad();
        nd::cross_entropy(model.logits(input, 6), target).backward();
        trainer.step(.03, 1);
    }
    auto after = nd::cross_entropy(model.logits(input, 6), target).item();
    std::cout << "Character RNN: " << before << " -> " << after << "\n";
    std::optional<nd::Array> hidden;
    int token = 0;
    std::cout << 'a';
    for (int i = 0; i < 24; ++i)
    {
        auto e = model.embedding.lookup({{token}, {1, 1}});
        auto result = model.rnn.run(e, {}, hidden);
        hidden = result.hidden.detach();
        auto values = model.head.forward(result.hidden).values();
        token = int(std::max_element(values.begin(), values.end()) - values.begin());
        std::cout << char('a' + token);
    }
    std::cout << '\n';
    return after < .05 ? 0 : 1;
}
