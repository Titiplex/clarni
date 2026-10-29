#include "embedding.h"
#include "nd_test_support.h"
#include <filesystem>
using namespace ml;
using namespace ml::nd;

int main()
{
    return run_tests(
        {
            {
                "vocabulary roundtrip and unknown",
                []
                {
                    const nn::Vocabulary v({"chat", "chien", "chat"}, 2);
                    require(v.id("chien") == 1 && v.id("chat") == 2, "min count");
                    const auto p = (std::filesystem::temp_directory_path() / "clarni_vocab_test.txt").string();
                    v.save(p);
                    const auto w = nn::Vocabulary::load(p);
                    require(w.token(2) == "chat", "vocab roundtrip");
                    std::filesystem::remove(p);
                }
            },
            {
                "embedding repeated indices and frozen padding",
                []
                {
                    nn::Embedding e(4, 2, 0);
                    const auto original = e.weight->value().values();
                    const auto out = e.lookup({{1, 1, 0, 2}, {2, 2}});
                    require(out.shape() == Shape({2, 2, 2}), "shape");
                    sum(out).backward();
                    const auto g = e.weight->value().grad();
                    require(g == Vector({0, 0, 2, 2, 1, 1, 0, 0}), "scatter-add gradient");
                    nn::Trainer t(e);
                    t.step(.1);
                    const auto next = e.weight->value().values();
                    near(next[0], original[0]);
                    near(next[1], original[1]);
                    expect_throw<std::invalid_argument>([&] { e.lookup({{4}, {1}}); });
                    // Numerical check excludes padding, whose gradient is deliberately overridden.
                    check_parameter_gradients(e,
                                              [&]
                                              {
                                                  const auto a = e.lookup({{1, 2, 1}, {3}});
                                                  return mean(multiply(a, a));
                                              });
                }
            },
            {
                "SkipGram gradients and training", []
                {
                    nn::SkipGram m(4, 3);
                    const auto centers = std::vector{0, 1, 2, 3};
                    const auto labels = std::vector{1, 0, 3, 2};
                    check_parameter_gradients(m,
                                              [&] { return cross_entropy(m.logits(centers), labels); });
                    OptimizerOptions o;
                    o.kind = OptimizerKind::Adam;
                    nn::Trainer t(m, o);
                    const double before = cross_entropy(m.logits(centers), labels).item();
                    for (int i = 0; i < 100; ++i)
                    {
                        t.zero_grad();
                        cross_entropy(m.logits(centers), labels).backward();
                        t.step(.04);
                    }
                    require(cross_entropy(m.logits(centers), labels).item() < before * .05,
                            "SkipGram learning");
                    auto pairs = nn::skipgram_pairs({2, 3, 4}, 1);
                    require(pairs.centers == std::vector({2, 3, 3, 4}) &&
                            pairs.contexts == std::vector({3, 2, 4, 3}),
                            "context pairs");
                }
            }
        });
}
