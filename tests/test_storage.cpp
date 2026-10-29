#include "nd_test_support.h"
using namespace ml;
using namespace ml::nd;
int main()
{
    return run_tests(
        {{"adopt storage and zero-copy immutable reshape",
          []
          {
              std::vector<double> data{1, 2, 3, 4, 5, 6};
              auto pointer = data.data();
              auto t = Tensor::from_values(2, 3, std::move(data));
              require(t.cpu_values().data() == pointer, "CPU storage not adopted");
              auto view = t.reshape(3, 2);
              require(view.cpu_values().data() == pointer, "reshape copied data");
              near(view.cpu_values()[4], 5);
              expect_throw<std::invalid_argument>([] { Tensor::from_values(2, 2, {1, 2, 3}); });
          }},
         {"same-shape broadcast returns same value storage",
          []
          {
              auto x = Array::from({2, 2}, {1, 2, 3, 4}, true);
              auto y = broadcast_to(x, {2, 2});
              require(x.variable().value().cpu_values().data() ==
                          y.variable().value().cpu_values().data(),
                      "identity broadcast copied");
              sum(y).backward();
              require(x.grad() == Vector(4, 1), "identity gradient");
          }},
         {"batched matmul broadcasting and numerical gradients", []
          {
              auto a = Array::from({2, 2, 2}, {.1, .2, .3, .4, .5, .6, .7, .8});
              auto b = Array::from({1, 2, 3}, {.1, .2, .3, .4, .5, .6});
              auto y = batched_matmul(a, b);
              require(y.shape() == Shape({2, 2, 3}), "output shape");
              for (int batch = 0; batch < 2; ++batch)
              {
                  auto ref = matmul(reshape(slice(a, 0, batch, 1), {2, 2}), reshape(b, {2, 3}));
                  auto got = reshape(slice(y, 0, batch, 1), {2, 3});
                  require(got.values() == ref.values(), "batched values");
              }
              check_input_gradient(a.shape(), a.values(),
                                   [&](auto &x)
                                   {
                                       auto p = batched_matmul(x, b);
                                       return sum(multiply(p, p));
                                   });
              check_input_gradient(b.shape(), b.values(),
                                   [&](auto &x) { return sum(batched_matmul(a, x)); });
              check_input_gradient(
                  {2, 1, 1, 2}, {.1, .2, .3, .4}, [](auto &x)
                  { return sum(batched_matmul(x, Array::from({3, 2, 1}, {1, 2, 3, 4, 5, 6}))); });
              expect_throw<std::invalid_argument>([&] { batched_matmul(a, Array({3, 2, 2})); });
          }}});
}
