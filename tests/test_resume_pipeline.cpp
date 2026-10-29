#include "test_support.h"
#include "training.h"
#include <filesystem>
using namespace ml;
using namespace ml::data;
int main()
{
    return run_tests(
        {{"mid-epoch model Adam and sampler resume follows the same trajectory", []
          {
              std::vector<Example> examples;
              for (int i = 0; i < 17; ++i)
                  examples.push_back({nd::Array::from({1}, {(i - 8) / 8.0}), {i >= 8 ? 1 : 0}});
              auto dataset = std::make_shared<VectorDataset>(examples);
              DataLoader a(dataset, {4, true, false, 91, "resume-fixture-v1"}),
                  b(dataset, {4, true, false, 91, "resume-fixture-v1"});
              nn::Linear ma(1, 2), mb(1, 2);
              OptimizerOptions opts;
              opts.kind = OptimizerKind::Adam;
              nn::Trainer ta(ma, opts), tb(mb, opts);
              auto step = [](nn::Linear &model, nn::Trainer &trainer, const Batch &batch)
              {
                  trainer.zero_grad();
                  nd::cross_entropy(model.forward(std::get<nd::Array>(batch.input)), batch.targets)
                      .backward();
                  trainer.step(.02, 1);
              };
              a.reset_epoch();
              step(ma, ta, *a.next());
              step(ma, ta, *a.next());
              auto dir = std::filesystem::temp_directory_path();
              auto model = (dir / "clarni_resume_model_test.nn").string(),
                   loader = (dir / "clarni_resume_loader_test.state").string();
              nn::save_checkpoint(model, ma, &ta);
              a.save_state(loader);
              nn::load_checkpoint(model, mb, &tb);
              b.load_state(loader);
              auto compare = [&]
              {
                  while (auto next = a.next())
                  {
                      auto other = b.next();
                      require(other.has_value() && next->indices == other->indices,
                              "batch mismatch");
                      step(ma, ta, *next);
                      step(mb, tb, *other);
                      require(ma.weight->value().values() == mb.weight->value().values() &&
                                  ma.bias->value().values() == mb.bias->value().values(),
                              "resumed Adam diverged");
                  }
                  require(!b.next(), "unexpected batch");
              };
              compare();
              a.reset_epoch();
              b.reset_epoch();
              compare();
              std::filesystem::remove(model);
              std::filesystem::remove(loader);
          }}});
}
