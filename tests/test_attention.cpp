#include "attention.h"
#include "nd_test_support.h"
using namespace ml;
using namespace ml::nd;

int main()
{
    return run_tests(
        {
            {
                "LayerNorm numerical gradients and constant inputs",
                []
                {
                    nn::LayerNorm n(3);
                    const auto x = Array::from({2, 3}, {.1, .4, -.7, .3, -.5, .2});
                    check_input_gradient(x.shape(), x.values(),
                                         [&](auto& a)
                                         {
                                             auto y = n.forward(a);
                                             return sum(multiply(y, Array::from({3}, {.2, .3, -.7})));
                                         });
                    check_parameter_gradients(n,
                                              [&]
                                              {
                                                  const auto y = n.forward(x);
                                                  return mean(multiply(y, y));
                                              });
                    auto y = n.forward(Array({2, 3}, 7));
                    for (const double v : y.values())
                        near(v, 0);
                }
            },
            {
                "attention numerical QKV gradients and all-masked rows",
                []
                {
                    auto q = Array::from({2, 2}, {.1, .3, .5, -.2});
                    auto k = Array::from({2, 2}, {-.3, .5, .2, .1});
                    auto v = Array::from({2, 2}, {.3, .6, -.4, .2});
                    std::vector mask{true, true, false, false};
                    check_input_gradient(
                        q.shape(), q.values(),
                        [&](auto& a) { return sum(nn::scaled_dot_product_attention(a, k, v, mask)); });
                    check_input_gradient(
                        k.shape(), k.values(),
                        [&](auto& a) { return sum(nn::scaled_dot_product_attention(q, a, v, mask)); });
                    check_input_gradient(
                        v.shape(), v.values(),
                        [&](auto& a) { return sum(nn::scaled_dot_product_attention(q, k, a, mask)); });
                    const auto y = nn::scaled_dot_product_attention(q, k, v, mask);
                    near(y.values()[2], 0);
                    near(y.values()[3], 0);
                }
            },
            {
                "multihead parameter gradients and masks",
                []
                {
                    nn::MultiHeadSelfAttention m(4, 2);
                    const auto x = Array::from(
                        {2, 2, 4}, {.1, .2, .3, .4, -.2, .1, .6, .3, .2, .4, .1, .3, .8, .7, .2, .5});
                    check_parameter_gradients(m,
                                              [&]
                                              {
                                                  const auto y = m.attend(x, {2, 0});
                                                  return mean(multiply(y, y));
                                              });
                    const auto y = m.attend(x, {2, 0});
                    for (int i = 8; i < 16; ++i)
                        near(y.values()[i], 0);
                    check_input_gradient(x.shape(), x.values(),
                                         [&](auto& a) { return sum(m.attend(a, {2, 1})); });
                }
            },
            {
                "Transformer block gradients and causal invariance",
                []
                {
                    nn::TransformerBlock m(2, 1, 3);
                    const auto x = Array::from({1, 3, 2}, {.2, -.3, .6, .1, -.2, .8});
                    check_parameter_gradients(
                        m,
                        [&]
                        {
                            const auto y = m.run(x);
                            return mean(multiply(y, y));
                        },
                        1e-4);
                    auto changed = x.values();
                    changed[4] = 40;
                    changed[5] = -70;
                    const auto a = m.forward(x).values();
                    const auto b = m.forward(Array::from(x.shape(), changed)).values();
                    for (int i = 0; i < 4; ++i)
                        near(a[i], b[i]);
                    nn::SinusoidalPosition positions(3, 4);
                    const auto p = positions.forward(Array({1, 2, 3})).values();
                    near(p[0], 0);
                    near(p[1], 1);
                    near(p[3], std::sin(1.0));
                    expect_throw<std::invalid_argument>([] { nn::MultiHeadSelfAttention(3, 2); });
                }
            },
            {
                "tiny Transformer language model learns next tokens", []
                {
                    nn::TransformerLM m(3, 4, 2, 8, 8);
                    const nn::TokenBatch batch{{0, 1, 2, 0, 1, 2}, {1, 6}};
                    const std::vector labels{1, 2, 0, 1, 2, 0};
                    OptimizerOptions o;
                    o.kind = OptimizerKind::Adam;
                    nn::Trainer t(m, o);
                    const double before = cross_entropy(m.logits(batch), labels).item();
                    for (int i = 0; i < 100; ++i)
                    {
                        t.zero_grad();
                        cross_entropy(m.logits(batch), labels).backward();
                        t.step(.02, 1);
                    }
                    require(cross_entropy(m.logits(batch), labels).item() < before * .05,
                            "Transformer failed to learn");
                    m.eval();
                    require(!m.block.attention.query.training(), "eval propagation");
                }
            }
        });
}
