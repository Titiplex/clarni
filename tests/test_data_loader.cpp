#include "data_loader.h"
#include "test_support.h"
#include <filesystem>
#include <fstream>
#include <set>
using namespace ml;
using namespace ml::data;
namespace
{
    std::shared_ptr<VectorDataset> dataset(int n = 5)
    {
        std::vector<Example> v;
        for (int i = 0; i < n; ++i)
            v.push_back({nd::Array::from({1}, {double(i)}), {i % 2}});
        return std::make_shared<VectorDataset>(v);
    }
    void write32(std::ostream &out, std::uint32_t x)
    {
        for (int shift = 24; shift >= 0; shift -= 8)
            out.put(static_cast<char>((x >> shift) & 255));
    }
} // namespace
int main()
{
    return run_tests(
        {{"stack batches, tail and drop_last",
          []
          {
              auto d = dataset();
              LoaderOptions o;
              o.batch_size = 2;
              o.shuffle = false;
              DataLoader loader(d, o);
              require(!loader.next(), "constructor should be exhausted");
              loader.reset_epoch();
              auto a = loader.next(), b = loader.next(), c = loader.next();
              require(a->indices == std::vector<std::size_t>({0, 1}) &&
                          b->indices == std::vector<std::size_t>({2, 3}) &&
                          c->indices == std::vector<std::size_t>({4}),
                      "batch order/tail");
              require(std::get<nd::Array>(c->input).shape() == nd::Shape({1, 1}),
                      "tail batch shape");
              require(!loader.next(), "extra batch");
              o.drop_last = true;
              DataLoader drop(d, o);
              drop.reset_epoch();
              require(drop.next().has_value() && drop.next().has_value() && !drop.next(),
                      "drop_last");
          }},
         {"token and feature padding, labels and lengths",
          []
          {
              auto tokens = make_collator({InputLayout::PadTokens, TargetLayout::Sequence, 0, -1});
              auto b =
                  tokens({{std::vector<int>{2, 3}, {3, 4}}, {std::vector<int>{5}, {6}}}, {7, 8});
              require(std::get<nn::TokenBatch>(b.input).ids == std::vector<int>({2, 3, 5, 0}),
                      "token padding");
              require(b.targets == std::vector<int>({3, 4, 6, -1}) &&
                          b.lengths == std::vector<int>({2, 1}),
                      "label masks");
              require(b.valid_targets() == 3, "valid tokens");
              auto features = make_collator({InputLayout::PadFeatures, TargetLayout::Class});
              auto f = features({{nd::Array::from({2, 2}, {1, 2, 3, 4}), {0}},
                                 {nd::Array::from({1, 2}, {5, 6}), {1}}},
                                {0, 1});
              require(std::get<nd::Array>(f.input).values() == nd::Vector({1, 2, 3, 4, 5, 6, 0, 0}),
                      "feature padding");
              expect_throw<std::invalid_argument>([&] { tokens({{std::vector<int>{}, {}}}, {0}); });
              expect_throw<std::invalid_argument>(
                  [&] { tokens({{std::vector<int>{2, 3}, {3}}}, {0}); });
          }},
         {"reproducible shuffling, exact mid-epoch resume and future epochs",
          []
          {
              auto d = dataset(17);
              LoaderOptions o;
              o.batch_size = 3;
              o.dataset_key = "dataset A";
              DataLoader a(d, o), b(d, o);
              a.reset_epoch();
              b.reset_epoch();
              require(a.next()->indices == b.next()->indices, "seed reproducibility");
              auto path =
                  (std::filesystem::temp_directory_path() / "clarni_loader_test.state").string();
              a.save_state(path);
              b.load_state(path);
              while (auto batch = a.next())
                  require(batch->indices == b.next()->indices, "resume batch mismatch");
              require(!b.next(), "resume tail");
              a.reset_epoch();
              b.reset_epoch();
              require(a.state().order == b.state().order, "future RNG mismatch");
              auto original = b.state(), bad = original;
              bad.order[1] = bad.order[0];
              expect_throw<std::invalid_argument>([&] { b.restore(bad); });
              require(b.state().rng == original.rng && b.state().order == original.order,
                      "failed restore mutated loader");
              bad = original;
              bad.cursor = 1;
              expect_throw<std::invalid_argument>([&] { b.restore(bad); });
              bad = original;
              bad.dataset_key = "other";
              expect_throw<std::invalid_argument>([&] { b.restore(bad); });
              std::filesystem::remove(path);
          }},
         {"collator failure does not consume samples",
          []
          {
              auto d = dataset();
              bool fail = true;
              auto base = make_collator();
              DataLoader l(d, {2, false, false, 42, ""},
                           [&](auto &samples, auto &indices)
                           {
                               if (fail)
                                   throw std::runtime_error("collator failed");
                               return base(samples, indices);
                           });
              l.reset_epoch();
              expect_throw<std::runtime_error>([&] { l.next(); });
              near(l.state().cursor, 0);
              fail = false;
              require(l.next()->indices == std::vector<std::size_t>({0, 1}), "samples lost");
          }},
         {"disjoint reproducible split and subset",
          []
          {
              auto a = random_split(20, 4, 3, 7), b = random_split(20, 4, 3, 7);
              require(a.training == b.training && a.validation == b.validation && a.test == b.test,
                      "split reproducibility");
              std::set<std::size_t> unique;
              for (auto *part : {&a.training, &a.validation, &a.test})
                  for (auto i : *part)
                      require(unique.insert(i).second, "overlapping split");
              require(unique.size() == 20 && a.training.size() == 13, "split coverage");
              Subset s(dataset(), {4, 1});
              near(std::get<nd::Array>(s.get(0).input).values()[0], 4);
              expect_throw<std::invalid_argument>([] { random_split(3, 2, 2); });
          }},
         {"raw IDX validation, limits and pixel normalization", []
          {
              auto dir = std::filesystem::temp_directory_path();
              auto images = (dir / "clarni_idx_images_test").string(),
                   labels = (dir / "clarni_idx_labels_test").string();
              {
                  std::ofstream out(images, std::ios::binary);
                  write32(out, 2051);
                  write32(out, 2);
                  write32(out, 2);
                  write32(out, 2);
                  for (int v : {0, 255, 127, 64, 1, 2, 3, 4})
                      out.put(static_cast<char>(v));
              }
              {
                  std::ofstream out(labels, std::ios::binary);
                  write32(out, 2049);
                  write32(out, 2);
                  out.put(3);
                  out.put(7);
              }
              MNIST d(images, labels);
              require(d.size() == 2 && d.rows() == 2 && d.cols() == 2, "IDX shape");
              auto sample = d.get(0);
              auto values = std::get<nd::Array>(sample.input).values();
              near(values[1], 1);
              near(values[2], 127.0 / 255);
              require(sample.targets == std::vector<int>({3}), "IDX label");
              MNIST limited(images, labels, 1);
              require(limited.size() == 1, "IDX limit");
              {
                  std::ofstream out(images, std::ios::binary | std::ios::app);
                  out.put(0);
              }
              expect_throw<std::invalid_argument>([&] { MNIST invalid(images, labels); });
              std::filesystem::remove(images);
              std::filesystem::remove(labels);
          }}});
}
