#include "gated_recurrent.h"
#include "nd_test_support.h"
using namespace ml;
using namespace ml::nd;

int main()
{
    return run_tests(
        {
            {
                "GRU full BPTT gradients including reset gate",
                []
                {
                    nn::GRU m(1, 2);
                    const auto x = Array::from({1, 3, 1}, {.2, -.3, .4});
                    auto h = Array::from({1, 2}, {.3, -.4});
                    check_parameter_gradients(m,
                                              [&]
                                              {
                                                  auto y = m.run(x, {}, h).sequence;
                                                  return mean(multiply(y, y));
                                              });
                    check_input_gradient(x.shape(), x.values(),
                                         [&](auto& a) { return sum(m.run(a, {}, h).hidden); });
                }
            },
            {
                "LSTM full BPTT and initial cell gradients",
                []
                {
                    nn::LSTM m(1, 2);
                    const auto x = Array::from({1, 3, 1}, {.1, -.2, .5});
                    const auto h = Array::from({1, 2}, {.2, -.1});
                    const auto c = Array::from({1, 2}, {.4, .3});
                    check_parameter_gradients(m,
                                              [&]
                                              {
                                                  const auto y = m.run(x, {}, nn::LSTMState{h, c});
                                                  return add(sum(y.sequence), sum(y.cell));
                                              });
                    check_input_gradient(c.shape(), c.values(),
                                         [&](auto& a)
                                         {
                                             const auto y = m.run(x, {}, nn::LSTMState{h, a});
                                             return add(sum(y.hidden), sum(y.cell));
                                         });
                    check_input_gradient(x.shape(), x.values(),
                                         [&](auto& a) { return sum(m.run(a).sequence); });
                }
            },
            {
                "gated sequence masks and detachment",
                []
                {
                    auto x = Array::from({2, 3, 1}, {.2, .3, .1, .4, .2, .5}, true);
                    nn::GRU g(1, 2);
                    nn::LSTM l(1, 2);
                    auto a = g.run(x, {2, 0});
                    auto b = l.run(x, {2, 0});
                    for (int i = 4; i < 12; ++i)
                    {
                        near(a.sequence.values()[i], 0);
                        near(b.sequence.values()[i], 0);
                    }
                    auto full = l.run(x);
                    auto trunc = l.run(x, {}, std::nullopt, 1);
                    require(full.hidden.values() == trunc.hidden.values(), "LSTM truncation values");
                    sum(trunc.hidden).backward();
                    auto grad = x.grad();
                    near(grad[0], 0);
                    near(grad[1], 0);
                    near(grad[3], 0);
                    near(grad[4], 0);
                    auto gt = g.run(x, {}, std::nullopt, 1);
                    sum(gt.hidden).backward();
                    near(x.grad()[0], 0);
                    near(x.grad()[1], 0);
                    expect_throw<std::invalid_argument>([&] { l.run(x, {1, 4}); });
                }
            },
            {
                "GRU and LSTM overfit tiny sequences", []
                {
                    for (const bool lstm : {false, true})
                    {
                        nn::Sequential m;
                        if (lstm)
                            m.add(std::make_shared<nn::LSTM>(1, 3));
                        else
                            m.add(std::make_shared<nn::GRU>(1, 3));
                        m.add(std::make_shared<nn::Linear>(3, 1));
                        auto x = Array::from({1, 3, 1}, {.2, .8, -.3}),
                             y = Array::from({1, 3, 1}, {.8, -.3, .2});
                        OptimizerOptions o;
                        o.kind = OptimizerKind::Adam;
                        nn::Trainer t(m, o);
                        for (int i = 0; i < 180; ++i)
                        {
                            t.zero_grad();
                            mse(m.forward(x), y).backward();
                            t.step(.03, 1);
                        }
                        require(mse(m.forward(x), y).item() < .005,
                                lstm ? "LSTM convergence" : "GRU convergence");
                    }
                }
            }
        }
    );
}
