#include "nd_test_support.h"
#include "recurrent.h"
using namespace ml;
using namespace ml::nd;
int main()
{
    return run_tests(
        {{"RNN BPTT numerical parameter and input gradients",
          []
          {
              nn::RNN r(2, 2);
              auto x = Array::from({2, 3, 2}, {.1, .2, -.4, .3, .5, .1, .2, .3, -.1, .4, .3, -.2});
              check_parameter_gradients(r,
                                        [&]
                                        {
                                            auto y = r.run(x, {3, 2}).sequence;
                                            return mean(multiply(y, y));
                                        });
              check_input_gradient(x.shape(), x.values(),
                                   [&](auto &a) { return sum(r.run(a, {3, 2}).sequence); });
          }},
         {"length masks, zero length and initial state",
          []
          {
              nn::RNN r(1, 1);
              auto x = Array::from({2, 3, 1}, {1, 2, 3, 4, 5, 6}, true),
                   h = Array::from({2, 1}, {.2, .7}, true);
              auto result = r.run(x, {2, 0}, h);
              auto y = result.sequence.values();
              near(y[2], 0);
              for (int i = 3; i < 6; ++i)
                  near(y[i], 0);
              near(result.hidden.values()[1], .7);
              sum(result.hidden).backward();
              auto g = x.grad();
              near(g[2], 0);
              for (int i = 3; i < 6; ++i)
                  near(g[i], 0);
              near(h.grad()[1], 1);
              expect_throw<std::invalid_argument>([&] { r.run(x, {4, 1}); });
          }},
         {"truncated BPTT cuts history without changing forward",
          []
          {
              nn::RNN r(1, 1);
              r.cell.weight_x->assign(Array::from({1, 1}, {.5}));
              r.cell.weight_h->assign(Array::from({1, 1}, {.7}));
              auto x = Array::from({1, 3, 1}, {.2, .3, .1}, true);
              auto full = r.run(x);
              auto truncated = r.run(x, {}, std::nullopt, 1);
              require(full.sequence.values() == truncated.sequence.values(),
                      "truncation changed values");
              sum(full.hidden).backward();
              require(std::abs(x.grad()[0]) > 1e-4, "full BPTT missing");
              sum(truncated.hidden).backward();
              near(x.grad()[0], 0);
              near(x.grad()[1], 0);
              require(std::abs(x.grad()[2]) > 0, "last step gradient missing");
          }},
         {"RNN learns a tiny sequence", []
          {
              nn::Sequential model;
              model.add(std::make_shared<nn::RNN>(1, 4));
              model.add(std::make_shared<nn::Linear>(4, 1));
              auto x = Array::from({1, 4, 1}, {-1, 1, -1, 1}),
                   target = Array::from({1, 4, 1}, {1, -1, 1, -1});
              OptimizerOptions o;
              o.kind = OptimizerKind::Adam;
              nn::Trainer t(model, o);
              for (int i = 0; i < 120; ++i)
              {
                  t.zero_grad();
                  mse(model.forward(x), target).backward();
                  t.step(.03, 1);
              }
              require(mse(model.forward(x), target).item() < .005, "sequence did not learn");
          }}});
}
