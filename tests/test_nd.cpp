#include "nd_test_support.h"
#include <filesystem>
#include <fstream>
using namespace ml;
using namespace ml::nd;
int main()
{
    return run_tests(
        {{"permute, broadcasting and shared branches",
          []
          {
              auto x = Array::from({2, 3}, {1, 2, 3, 4, 5, 6}, true);
              auto y = permute(x, {1, 0});
              require(y.values() == Vector({1, 4, 2, 5, 3, 6}), "permute");
              auto b = Array::from({3}, {2, 3, 4}, true);
              sum(multiply(x, b)).backward();
              require(x.grad() == Vector({2, 3, 4, 2, 3, 4}), "x grad");
              require(b.grad() == Vector({5, 7, 9}), "b grad");
              auto s = slice(x, 1, 1, 2);
              sum(concat({s, s}, 0)).backward();
              require(x.grad() == Vector({0, 2, 2, 0, 2, 2}), "shared accumulation");
              auto r = reduce_sum(x, 1);
              require(r.values() == Vector({6, 15}), "reduce");
              sum(r).backward();
              require(x.grad() == Vector(6, 1), "reduce grad");
          }},
         {"N-D composed numerical gradients",
          []
          {
              check_input_gradient({2, 3}, {.1, .3, .5, -.4, .2, .7},
                                   [](auto &x)
                                   {
                                       auto y = softmax(
                                           add(permute(x, {1, 0}), Array::from({2}, {.3, -.2})), 1);
                                       auto z = concat({slice(y, 0, 0, 2), slice(y, 0, 1, 2)}, 0);
                                       return mean(multiply(z, z));
                                   });
          }},
         {"stable masked CE",
          []
          {
              auto a = Array::from({2, 3}, {1000, 1001, 999, 3, 2, 1}, true);
              auto l = cross_entropy(a, {1, -1});
              l.backward();
              auto p = softmax(a).values();
              for (int i = 0; i < 3; ++i)
                  near(a.grad()[i], p[i] - (i == 1));
              for (int i = 3; i < 6; ++i)
                  near(a.grad()[i], 0);
              near(l.item(), std::log(1 + std::exp(-1) + std::exp(-2)));
              expect_throw<std::invalid_argument>([&] { cross_entropy(a, {-1, -1}); });
              check_input_gradient({2, 3}, {.1, .4, -.2, .3, .7, -.5},
                                   [](auto &x) { return cross_entropy(x, {2, 0}); });
          }},
         {"detach",
          []
          {
              auto x = Array::from({1}, {3}, true);
              sum(add(x, x.detach())).backward();
              near(x.grad()[0], 1);
          }},
         {"MLP training and topology",
          []
          {
              nn::Sequential model;
              auto l = std::make_shared<nn::Linear>(1, 1);
              model.add(l);
              OptimizerOptions o;
              o.kind = OptimizerKind::Adam;
              nn::Trainer t(model, o);
              auto x = Array::from({4, 1}, {-1, 0, 1, 2}),
                   target = Array::from({4, 1}, {-1, 1, 3, 5});
              double before = mse(model.forward(x), target).item();
              for (int i = 0; i < 300; ++i)
              {
                  t.zero_grad();
                  mse(model.forward(x), target).backward();
                  t.step(.05);
              }
              require(mse(model.forward(x), target).item() < before * 1e-5, "MLP convergence");
              model.add(l);
              require(nn::parameters(model).size() == 2, "dedup");
              model.eval();
              require(!l->training(), "eval");
              model.add(std::make_shared<nn::Linear>(1, 1));
              expect_throw<std::logic_error>([&] { t.zero_grad(); });
          }},
         {"clipping and linear gradients",
          []
          {
              nn::Linear m(2, 2);
              auto x = Array::from({2, 2}, {.3, .2, -.4, .7});
              check_parameter_gradients(m,
                                        [&]
                                        {
                                            auto y = m.forward(x);
                                            return mean(multiply(y, y));
                                        });
              nn::Trainer t(m);
              t.zero_grad();
              sum(m.forward(x)).backward();
              double n = t.grad_norm();
              auto w = m.weight->value().values();
              auto g = m.weight->value().grad();
              t.step(.1, .01);
              for (std::size_t i = 0; i < w.size(); ++i)
                  near(m.weight->value().values()[i], w[i] - .1 * g[i] * .01 / n);
          }},
         {"checkpoint Adam continuation and transactional rejection",
          []
          {
              nn::Linear a(1, 1), b(1, 1);
              OptimizerOptions o;
              o.kind = OptimizerKind::Adam;
              nn::Trainer ta(a, o), tb(b, o);
              auto x = Array::from({2, 1}, {-1, 2}), y = Array::from({2, 1}, {-2, 4});
              auto step = [&](auto &m, auto &t)
              {
                  t.zero_grad();
                  mse(m.forward(x), y).backward();
                  t.step(.01);
              };
              step(a, ta);
              auto path = (std::filesystem::temp_directory_path() / "clarni_nn_checkpoint_test.txt")
                              .string();
              nn::save_checkpoint(path, a, &ta);
              nn::load_checkpoint(path, b, &tb);
              step(a, ta);
              step(b, tb);
              require(a.forward(x).values() == b.forward(x).values(), "resume diverged");
              auto original = b.forward(x).values();
              {
                  std::ofstream out(path);
                  out << "CLARNI_NN 1\n2\n\"weight\" 2 1 1\n9\n";
              }
              expect_throw<std::invalid_argument>([&] { nn::load_checkpoint(path, b, &tb); });
              require(original == b.forward(x).values(), "partial load");
              std::filesystem::remove(path);
          }},
         {"invalid shapes", []
          {
              expect_throw<std::invalid_argument>([] { Array({0}); });
              expect_throw<std::invalid_argument>([] { add(Array({2}), Array({3})); });
              expect_throw<std::invalid_argument>([] { permute(Array({2, 2}), {0, 0}); });
              expect_throw<std::invalid_argument>([] { reshape(Array({2}), {3}); });
          }}});
}
