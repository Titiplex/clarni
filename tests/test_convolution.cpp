#include "convolution.h"
#include "nd_test_support.h"
using namespace ml;
using namespace ml::nd;
int main()
{
    return run_tests(
        {{"Conv1D hand values and gradients with stride padding dilation",
          []
          {
              nn::Conv1D m(1, 1, 2, 1, 0);
              m.weight->assign(Array::from({1, 1, 2}, {2, 3}));
              auto y = m.forward(Array::from({1, 1, 3}, {1, 2, 4}));
              require(y.values() == Vector({8, 16}), "convolution convention");
              nn::Conv1D a(2, 2, 2, 2, 1, 2);
              Vector v{.1, .2, .3, .4, .5, .7, -.2, .1, .8, .2};
              check_input_gradient({1, 2, 5}, v,
                                   [&](auto &x)
                                   {
                                       auto z = a.forward(x);
                                       return mean(multiply(z, z));
                                   });
              auto x = Array::from({1, 2, 5}, v);
              check_parameter_gradients(a,
                                        [&]
                                        {
                                            auto z = a.forward(x);
                                            return mean(multiply(z, z));
                                        });
          }},
         {"Conv2D gradients, rectangular kernels and batches",
          []
          {
              nn::Conv2D m(1, 2, {2, 2}, {1, 2}, {1, 0});
              auto x =
                  Array::from({2, 1, 2, 3}, {.1, .2, .5, -.2, .8, .3, .3, -.1, .7, .5, .2, .4});
              check_parameter_gradients(m,
                                        [&]
                                        {
                                            auto z = m.forward(x);
                                            return mean(multiply(z, z));
                                        });
              check_input_gradient(x.shape(), x.values(),
                                   [&](auto &a) { return sum(m.forward(a)); });
              nn::Conv2D rect(1, 1, {1, 2});
              require(rect.forward(x).shape() == Shape({2, 1, 2, 2}), "rectangular shape");
          }},
         {"pooling overlap and numerical gradients",
          []
          {
              check_input_gradient({1, 1, 4}, {.1, .7, -.3, .9},
                                   [](auto &a)
                                   {
                                       auto p = nn::pool(a, {3}, {1}, nn::PoolKind::Max);
                                       return sum(multiply(p, p));
                                   });
              check_input_gradient(
                  {1, 1, 3, 3}, {.1, .2, .3, .4, .5, .6, .7, .8, .9},
                  [](auto &a) { return sum(nn::pool(a, {2, 2}, {1, 1}, nn::PoolKind::Average)); });
              auto a = Array::from({1, 1, 3}, {2, 2, 1}, true);
              sum(nn::pool(a, {2}, {1})).backward();
              require(a.grad() == Vector({1, 1, 0}), "max ties choose first");
          }},
         {"CNN overfits tiny images",
          []
          {
              nn::Sequential m;
              m.add(std::make_shared<nn::Conv2D>(1, 2, Shape{2, 2}));
              m.add(std::make_shared<nn::ActivationLayer>(Activation::Tanh));
              m.add(std::make_shared<nn::Flatten>());
              m.add(std::make_shared<nn::Linear>(8, 2));
              OptimizerOptions o;
              o.kind = OptimizerKind::Adam;
              nn::Trainer t(m, o);
              auto x =
                  Array::from({2, 1, 3, 3}, {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1});
              for (int i = 0; i < 80; ++i)
              {
                  t.zero_grad();
                  cross_entropy(m.forward(x), {0, 1}).backward();
                  t.step(.03);
              }
              require(cross_entropy(m.forward(x), {0, 1}).item() < .02, "CNN failed to learn");
          }},
         {"invalid convolution options", []
          {
              expect_throw<std::invalid_argument>([] { nn::Conv1D(1, 1, 3, 0); });
              nn::Conv2D c(1, 1, {3, 3});
              expect_throw<std::invalid_argument>([&] { c.forward(Array({1, 1, 2, 2})); });
          }}});
}
