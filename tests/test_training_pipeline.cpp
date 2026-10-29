#include "test_support.h"
#include "training.h"
using namespace ml;
using namespace ml::training;
using namespace ml::data;
namespace
{
    std::shared_ptr<VectorDataset> tiny()
    {
        return std::make_shared<VectorDataset>(
            std::vector<Example>{{nd::Array::from({1}, {-1}), {0}},
                                 {nd::Array::from({1}, {1}), {1}},
                                 {nd::Array::from({1}, {-2}), {0}},
                                 {nd::Array::from({1}, {2}), {1}}});
    }
    LoaderOptions options(bool shuffle, std::size_t batch = 3)
    {
        return {batch, shuffle, false, 42, "test"};
    }
} // namespace
int main()
{
    return run_tests(
        {{"metrics, ignored labels and absent classes",
          []
          {
              auto x = nd::Array::from({3, 2}, {4, 0, 0, 4, 1, 0});
              std::vector<int> y{0, 1, 1};
              ClassificationMetrics m(2);
              m.add(x, y, nd::cross_entropy(x, y).item());
              auto result = m.result();
              near(result.accuracy, 2.0 / 3);
              near(result.macro_f1, 2.0 / 3);
              near(result.perplexity, std::exp(result.loss));
              require(result.count == 3, "metric count");
              ClassificationMetrics large(2);
              large.add(x, y, 1e308);
              large.add(x, y, 1e308);
              near(large.result().loss, 1e308);
              require(std::isinf(large.result().perplexity),
                      "perplexity overflow should be explicit");
              ClassificationMetrics masked(2);
              masked.add(x, {0, 1, -1}, .5);
              near(masked.result().accuracy, 1);
              near(masked.result().loss, .5);
              expect_throw<std::invalid_argument>([] { ClassificationMetrics(0); });
              expect_throw<std::invalid_argument>([] { ClassificationMetrics(2).result(); });
          }},
         {"validation is weighted by targets, no-grad, preserves mode and sampler",
          []
          {
              nn::Linear unused(2, 2);
              unused.train();
              SupervisedTrainer trainer(unused);
              auto data = std::make_shared<VectorDataset>(
                  std::vector<Example>{{nd::Array::from({2}, {5, 0}), {0}},
                                       {nd::Array::from({2}, {0, 5}), {1}},
                                       {nd::Array::from({2}, {1, 0}), {1}}});
              DataLoader loader(data, options(false, 2));
              auto state = loader.state();
              bool saw_no_grad = true;
              auto forward = [&](const Batch &b)
              {
                  saw_no_grad = saw_no_grad && !ad::grad_enabled() && !unused.training();
                  return std::get<nd::Array>(b.input);
              };
              auto result = trainer.evaluate(loader, forward, 2);
              auto all = nd::Array::from({3, 2}, {5, 0, 0, 5, 1, 0});
              near(result.loss, nd::cross_entropy(all, {0, 1, 1}).item());
              require(result.count == 3, "tail ignored");
              require(saw_no_grad && unused.training() && ad::grad_enabled(), "mode/no-grad");
              require(loader.state().cursor == state.cursor &&
                          loader.state().epoch == state.epoch && loader.state().rng == state.rng,
                      "validation consumed loader");
              expect_throw<std::runtime_error>(
                  [&]
                  {
                      trainer.evaluate(
                          loader, [](auto &) -> nd::Array { throw std::runtime_error("test"); }, 2);
                  });
              require(unused.training() && ad::grad_enabled() && loader.state().rng == state.rng,
                      "exception restoration");
          }},
         {"shared training loop learns classification and restores caller mode",
          []
          {
              nn::Sequential model;
              model.add(std::make_shared<nn::Linear>(1, 4));
              model.add(std::make_shared<nn::ActivationLayer>(Activation::Tanh));
              model.add(std::make_shared<nn::Linear>(4, 2));
              OptimizerOptions opt;
              opt.kind = OptimizerKind::Adam;
              SupervisedTrainer trainer(model, opt);
              DataLoader train(tiny(), options(true)), valid(tiny(), options(false));
              auto forward = [&](auto &batch)
              { return model.forward(std::get<nd::Array>(batch.input)); };
              model.eval();
              FitOptions f;
              f.epochs = 50;
              f.learning_rate = .04;
              f.classes = 2;
              auto result = trainer.fit(train, valid, forward, f);
              auto metrics = trainer.evaluate(valid, forward, 2);
              require(metrics.accuracy == 1 && metrics.loss < .03, "classification learning");
              require(result.best_epoch > 0 && !model.training(), "best epoch/mode");
          }},
         {"early stopping restores weights, Adam counters and training shuffle",
          []
          {
              nn::Linear a(1, 2), reference(1, 2);
              OptimizerOptions opt;
              opt.kind = OptimizerKind::Adam;
              SupervisedTrainer ta(a, opt), tr(reference, opt);
              DataLoader train(tiny(), options(true, 2)), valid(tiny(), options(false, 2)),
                  rt(tiny(), options(true, 2)), rv(tiny(), options(false, 2));
              std::size_t epoch = 0;
              auto fake_validation = [&](nn::Linear &model, const Batch &batch)
              {
                  auto x = std::get<nd::Array>(batch.input);
                  if (model.training())
                      return model.forward(x);
                  nd::Vector values;
                  for (int label : batch.targets)
                  {
                      double correct = epoch == 0 ? 4 : 0;
                      values.push_back(label == 0 ? correct : 0);
                      values.push_back(label == 1 ? correct : 0);
                  }
                  return nd::Array::from({int(batch.targets.size()), 2}, values);
              };
              FitOptions f;
              f.epochs = 8;
              f.patience = 2;
              f.learning_rate = .01;
              f.on_epoch = [&](const EpochResult &r) { epoch = r.epoch; };
              auto result = ta.fit(train, valid, [&](auto &b) { return fake_validation(a, b); }, f);
              require(result.stopped_early && result.history.size() == 3 && result.best_epoch == 1,
                      "patience result");
              epoch = 0;
              f.epochs = 1;
              f.on_epoch = {};
              tr.fit(rt, rv, [&](auto &b) { return fake_validation(reference, b); }, f);
              auto ap = nn::parameters(a), rp = nn::parameters(reference);
              for (std::size_t i = 0; i < ap.size(); ++i)
                  require(ap[i].parameter->value().values() == rp[i].parameter->value().values(),
                          "best weights not restored");
              auto &as = ta.optimizer().optimizer().states();
              auto &rs = tr.optimizer().optimizer().states();
              require(as.size() == rs.size() && as[0].steps == rs[0].steps,
                      "Adam counter mismatch");
              require(train.state().rng == rt.state().rng &&
                          train.state().epoch == rt.state().epoch,
                      "shuffle not restored");
              // Another update must follow the same Adam trajectory.
              train.reset_epoch();
              rt.reset_epoch();
              auto ba = train.next(), br = rt.next();
              ta.optimizer().zero_grad();
              tr.optimizer().zero_grad();
              nd::cross_entropy(a.forward(std::get<nd::Array>(ba->input)), ba->targets).backward();
              nd::cross_entropy(reference.forward(std::get<nd::Array>(br->input)), br->targets)
                  .backward();
              ta.optimizer().step(.01);
              tr.optimizer().step(.01);
              require(a.weight->value().values() == reference.weight->value().values(),
                      "best Adam moments not restored");
          }},
         {"min_delta accumulates progress since last significant improvement",
          []
          {
              nn::Linear model(1, 2);
              SupervisedTrainer trainer(model);
              DataLoader train(tiny(), options(true)), valid(tiny(), options(false));
              std::size_t epoch = 0;
              FitOptions f;
              f.epochs = 4;
              f.patience = 3;
              f.min_delta = .025;
              f.on_epoch = [&](auto &r) { epoch = r.epoch; };
              auto forward = [&](const Batch &b)
              {
                  if (model.training())
                      return model.forward(std::get<nd::Array>(b.input));
                  double loss = .5 - .01 * epoch, z = -std::log(std::expm1(loss));
                  nd::Vector v;
                  for (int label : b.targets)
                  {
                      v.push_back(label == 0 ? z : 0);
                      v.push_back(label == 1 ? z : 0);
                  }
                  return nd::Array::from({int(b.targets.size()), 2}, v);
              };
              auto result = trainer.fit(train, valid, forward, f);
              require(!result.stopped_early && result.history.size() == 4 && result.best_epoch == 4,
                      "min_delta cumulative semantics");
          }},
         {"schedules and invalid pipeline options", []
          {
              near(cosine_schedule(.1, .01, 1)(0), .1);
              auto cosine = cosine_schedule(.1, .01, 6, 2);
              near(cosine(0), .05);
              near(cosine(1), .1);
              near(cosine(2), .1);
              near(cosine(5), .01);
              auto step = step_schedule(.1, 2, .5);
              near(step(0), .1);
              near(step(2), .05);
              expect_throw<std::out_of_range>([&] { cosine(6); });
              expect_throw<std::invalid_argument>([] { cosine_schedule(.1, .01, 2, 2); });
              nn::Linear m(1, 2);
              SupervisedTrainer trainer(m);
              DataLoader train(tiny(), options(true)), valid(tiny(), options(false));
              auto forward = [&](auto &b) { return m.forward(std::get<nd::Array>(b.input)); };
              {
                  ad::NoGradGuard guard;
                  expect_throw<std::logic_error>([&] { trainer.fit(train, valid, forward); });
              }
              expect_throw<std::invalid_argument>([&] { trainer.evaluate(train, forward, 2); });
          }}});
}
