#include "attention.h"
#include "convolution.h"
#include "test_support.h"
#include "training.h"
#include <chrono>
#include <filesystem>
#include <limits>
using namespace ml;
using namespace ml::nd;
namespace
{
    struct TemporaryDirectory
    {
        std::filesystem::path path;
        TemporaryDirectory()
        {
            auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            for (int attempt = 0; attempt < 100; ++attempt)
            {
                path =
                    std::filesystem::temp_directory_path() /
                    ("clarni-cuda-test-" + std::to_string(nonce) + "-" + std::to_string(attempt));
                if (std::filesystem::create_directory(path))
                    return;
            }
            throw std::runtime_error("Cannot create isolated test directory");
        }
        ~TemporaryDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    };

    void compare(const Vector &a, const Vector &b, double tolerance = 1e-8)
    {
        require(a.size() == b.size(), "size mismatch");
        for (std::size_t i = 0; i < a.size(); ++i)
            near(a[i], b[i], tolerance);
    }
    Array on_gpu(const Array &a, bool gradient = false)
    {
        return Array::from(a.shape(), a.values(), gradient, Backend::CUDA);
    }
    void parameter_gradients(const nn::Module &a, const nn::Module &b)
    {
        auto pa = nn::parameters(a), pb = nn::parameters(b);
        require(pa.size() == pb.size(), "parameter count");
        for (std::size_t i = 0; i < pa.size(); ++i)
            compare(pa[i].parameter->value().grad(), pb[i].parameter->value().grad());
    }
    void weights(const nn::Module &a, const nn::Module &b)
    {
        auto pa = nn::parameters(a), pb = nn::parameters(b);
        for (std::size_t i = 0; i < pa.size(); ++i)
            compare(pa[i].parameter->value().values(), pb[i].parameter->value().values(), 2e-7);
    }
    std::unique_ptr<nn::Sequential> cnn(Backend backend)
    {
        auto m = std::make_unique<nn::Sequential>();
        m->add(std::make_shared<nn::Conv2D>(1, 2, Shape{2, 2}));
        m->add(std::make_shared<nn::ActivationLayer>(Activation::Tanh));
        m->add(std::make_shared<nn::Pool2D>(Shape{2, 2}, Shape{2, 2}));
        m->add(std::make_shared<nn::Flatten>());
        m->add(std::make_shared<nn::Linear>(8, 2));
        m->to(backend);
        return m;
    }
    Array images()
    {
        Vector v(4 * 25);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = .1 + double(i % 13) / 13;
        return Array::from({4, 1, 5, 5}, v);
    }
    void step(nn::Module &model, nn::Trainer &trainer, const Array &x,
              const std::vector<int> &labels)
    {
        trainer.zero_grad();
        cross_entropy(model.forward(x), labels).backward();
        trainer.step(.01, .5);
    }
} // namespace
int main()
{
#if defined(CLARNI_MOCK_GPU)
    std::cout << "CPU simulation of CUDA dispatch; per-output math is shared with nd.cu. No GPU "
                 "execution/performance validation.\n";
    initialize_compute(Backend::CUDA);
#else
    auto info = initialize_compute(Backend::Auto);
    if (info.selected != Backend::CUDA)
    {
        std::cout << "SKIP N-D CUDA hardware unavailable: " << info.diagnostic << '\n';
        return 77;
    }
#endif
    return run_tests(
        {{"CUDA storage transfers and cross-backend gradients",
          []
          {
              auto a = Array::from({2}, {.2, .4}, true);
              auto transferred = a.to(Backend::CUDA);
              sum(scale(transferred, 3)).backward();
              compare(a.grad(), {3, 3});
              require(transferred.backend() == Backend::CUDA, "transfer backend");
              auto x = on_gpu(Array::from({1, 3}, {.1, .4, .7}), true);
              auto cpu = Array::from({1, 3}, {.1, .4, .7}, true);
              auto y = softmax(broadcast_to(x, {2, 3}), 1),
                   ref = softmax(broadcast_to(cpu, {2, 3}), 1);
              compare(y.values(), ref.values());
              auto factor = Array::from({3}, {.2, -.3, .7});
              sum(multiply(y, on_gpu(factor))).backward();
              sum(multiply(ref, factor)).backward();
              compare(x.grad(), cpu.grad());
              expect_throw<std::invalid_argument>([&] { add(x, cpu); });
          }},
         {"CUDA permutation slicing and reductions",
          []
          {
              auto x = Array::from({2, 3, 2}, {.1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2},
                                   true),
                   device = on_gpu(x, true);
              auto transform = [](const Array &a)
              { return reduce_sum(slice(permute(a, {1, 0, 2}), 0, 1, 2), 1, true); };
              auto ref = transform(x), got = transform(device);
              compare(got.values(), ref.values());
              sum(multiply(ref, ref)).backward();
              sum(multiply(got, got)).backward();
              compare(device.grad(), x.grad());
              auto cpu = Array::from({2, 3}, {1, 2, 3, 4, 5, 6});
              compare(reduce_sum(on_gpu(cpu), 0).values(), reduce_sum(cpu, 0).values());
          }},
         {"CUDA stable sparse cross entropy and ignored labels",
          []
          {
              auto a = Array::from({2, 3}, {1000, 1001, 999, -1000, -999, -998}, true),
                   b = on_gpu(a, true);
              auto ca = cross_entropy(a, {1, -1}), cb = cross_entropy(b, {1, -1});
              near(ca.item(), cb.item());
              ca.backward();
              cb.backward();
              compare(a.grad(), b.grad());
              expect_throw<std::invalid_argument>([&] { cross_entropy(b, {-1, -1}); });
              expect_throw<std::invalid_argument>([&] { cross_entropy(b, {3, 0}); });
          }},
         {"CUDA Conv1D gradients stride padding dilation",
          []
          {
              nn::Conv1D cpu(2, 2, 2, 2, 1, 2), device(2, 2, 2, 2, 1, 2);
              device.to(Backend::CUDA);
              auto x = Array::from({2, 2, 5}, {.1, .2, .3, .4,  .5, .7, -.2, .1, .8,  .2,
                                               .2, .4, .1, -.1, .6, .3, .5,  .4, -.3, .1},
                                   true),
                   dx = on_gpu(x, true);
              auto y = cpu.forward(x), dy = device.forward(dx);
              compare(y.values(), dy.values());
              sum(multiply(y, y)).backward();
              sum(multiply(dy, dy)).backward();
              compare(x.grad(), dx.grad());
              parameter_gradients(cpu, device);
          }},
         {"CUDA Conv2D rectangular kernels and gradients",
          []
          {
              nn::Conv2D cpu(1, 2, {2, 3}, {1, 2}, {1, 1}, {2, 1}),
                  device(1, 2, {2, 3}, {1, 2}, {1, 1}, {2, 1});
              device.to(Backend::CUDA);
              Vector values(40);
              for (std::size_t i = 0; i < values.size(); ++i)
                  values[i] = double(i % 11) / 11 - .2;
              auto x = Array::from({2, 1, 4, 5}, values, true), dx = on_gpu(x, true);
              auto y = cpu.forward(x), dy = device.forward(dx);
              compare(y.values(), dy.values());
              mean(multiply(y, y)).backward();
              mean(multiply(dy, dy)).backward();
              compare(x.grad(), dx.grad());
              parameter_gradients(cpu, device);
          }},
         {"CUDA pooling overlap average max ties and negative infinity",
          []
          {
              for (auto kind : {nn::PoolKind::Max, nn::PoolKind::Average})
              {
                  auto x = Array::from({1, 1, 3, 4},
                                       {.2, .8, .1, .3, .7, .4, .9, -.1, .6, .5, .2, 1}, true),
                       device = on_gpu(x, true);
                  auto y = nn::pool(x, {2, 2}, {1, 1}, kind),
                       dy = nn::pool(device, {2, 2}, {1, 1}, kind);
                  compare(y.values(), dy.values());
                  sum(multiply(y, y)).backward();
                  sum(multiply(dy, dy)).backward();
                  compare(x.grad(), device.grad());
              }
              auto x = Array::from({1, 1, 3}, Vector(3, -std::numeric_limits<double>::infinity()),
                                   true),
                   device = on_gpu(x, true);
              sum(nn::pool(x, {2}, {1})).backward();
              sum(nn::pool(device, {2}, {1})).backward();
              compare(x.grad(), device.grad());
          }},
         {"CUDA norm clipping and resident training transfers",
          []
          {
              auto model = cnn(Backend::CUDA);
              OptimizerOptions options;
              options.kind = OptimizerKind::Adam;
              nn::Trainer trainer(*model, options);
              auto x = on_gpu(images());
              std::vector<int> labels{0, 1, 0, 1};
              gpu::resident::reset_statistics();
              step(*model, trainer, x, labels);
              auto stats = gpu::resident::statistics();
              require(stats.uploads == 1 && stats.uploaded_bytes == labels.size() * sizeof(double),
                      "parameters/intermediates uploaded again during an already-resident step");
              require(stats.downloaded_bytes < 256,
                      "bulk tensor download in forward/backward/Adam");
              for (auto &state : trainer.optimizer().states())
                  if (state.steps)
                  {
                      require(state.first.backend() == Backend::CUDA &&
                                  state.second.backend() == Backend::CUDA,
                              "Adam moments left VRAM");
                  }
              auto moment_state = trainer.optimizer().states();
              gpu::resident::reset_statistics();
              trainer.restore_optimizer(moment_state);
              require(gpu::resident::statistics().downloaded_bytes <= 128,
                      "optimizer-state validation downloaded full moments");
              auto logits = model->forward(x);
              double loss = cross_entropy(logits, labels).item();
              training::ClassificationMetrics metrics(2);
              gpu::resident::reset_statistics();
              metrics.add(logits, labels, loss);
              stats = gpu::resident::statistics();
              require(stats.uploads == 0 &&
                          stats.downloaded_bytes <= labels.size() * sizeof(double) + 16,
                      "metrics downloaded logits instead of compact predictions");
              require(metrics.result().count == 4, "metric count");
              require(on_gpu(Array::from({2}, {0, 1})).variable().value().all_nonnegative(),
                      "nonnegative GPU predicate");
              require(!on_gpu(Array::from({2}, {-1, 1})).variable().value().all_nonnegative(),
                      "negative GPU predicate");
              auto huge = Array::from({2}, {1e200, -1e200});
              near(on_gpu(huge).variable().value().l2_norm() / 1e200, std::sqrt(2.0));
          }},
         {"CUDA CNN Adam parity and CPU/GPU checkpoint portability",
          []
          {
              auto cpu = cnn(Backend::CPU), device = cnn(Backend::CUDA);
              OptimizerOptions options;
              options.kind = OptimizerKind::Adam;
              nn::Trainer ct(*cpu, options), dt(*device, options);
              auto x = images(), dx = on_gpu(x);
              std::vector<int> labels{0, 1, 0, 1};
              auto expected_inference = device->forward(dx).values();
              {
                  ad::NoGradGuard no_grad;
                  auto inference = device->forward(dx);
                  require(!inference.variable().requires_grad(), "CUDA inference recorded a graph");
                  compare(inference.values(), expected_inference);
              }
              double before = cross_entropy(device->forward(dx), labels).item();
              for (int i = 0; i < 20; ++i)
              {
                  step(*cpu, ct, x, labels);
                  step(*device, dt, dx, labels);
              }
              weights(*cpu, *device);
              require(cross_entropy(device->forward(dx), labels).item() < before,
                      "CUDA CNN loss did not decrease");
              TemporaryDirectory temporary;
              auto path = (temporary.path / "checkpoint.nn").string();
              nn::save_checkpoint(path, *device, &dt);
              auto loaded = cnn(Backend::CUDA), loaded_cpu = cnn(Backend::CPU);
              nn::Trainer lt(*loaded, options), lct(*loaded_cpu, options);
              nn::load_checkpoint(path, *loaded, &lt);
              nn::load_checkpoint(path, *loaded_cpu, &lct);
              weights(*device, *loaded);
              weights(*device, *loaded_cpu);
              require(loaded->backend() == Backend::CUDA &&
                          lt.optimizer().states()[0].first.backend() == Backend::CUDA,
                      "checkpoint changed target backend");
              step(*device, dt, dx, labels);
              step(*loaded, lt, dx, labels);
              step(*loaded_cpu, lct, x, labels);
              weights(*device, *loaded);
              weights(*device, *loaded_cpu);
              std::filesystem::remove(path);
          }},
         {"CUDA unsupported operations reject without fallback downloads", []
          {
              auto x = on_gpu(Array::from({2, 2}, {1, 2, 3, 4}));
              gpu::resident::reset_statistics();
              expect_throw<std::invalid_argument>([&] { concat({x, x}, 0); });
              expect_throw<std::invalid_argument>([&] { batched_matmul(x, x); });
              nn::LayerNorm norm(2);
              expect_throw<std::invalid_argument>([&] { norm.forward(x); });
              nn::RNN recurrent(2, 3);
              expect_throw<std::invalid_argument>([&] { recurrent.run(reshape(x, {1, 2, 2})); });
              require(gpu::resident::statistics().downloads == 0,
                      "unsupported operation silently copied data to CPU");
          }}});
}
