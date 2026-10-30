#include "convolution.h"
#include "training.h"
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

namespace
{
    struct Options
    {
        std::filesystem::path data, output = "runs/mnist";
        std::size_t train_limit = 2000, test_limit = 1000, validation = 200, epochs = 3, batch = 32;
        std::uint32_t seed = 42;
        double rate = .003;
        ml::Backend backend = ml::Backend::Auto;
    };

    std::size_t number(std::string_view value)
    {
        std::size_t n = 0;
        auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), n);
        if (ec != std::errc() || end != value.data() + value.size())
            throw std::invalid_argument("Invalid unsigned integer option");
        return n;
    }

    void usage()
    {
        std::cout << "MNIST CNN (raw IDX files)\n  --data DIRECTORY [--output DIRECTORY] [--epochs "
                     "3] [--batch-size 32]\n  [--train-limit 2000] [--test-limit 1000] "
                     "[--validation 200] [--seed 42]\n  [--learning-rate 0.003] [--backend "
                     "auto|cpu|cuda]\nLimits of 0 load "
                     "the full official split; defaults use a small CPU subset.\n";
    }

    Options parse(const int argc, char **argv)
    {
        Options o;
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view key = argv[i];
            if (key == "--help")
            {
                usage();
                std::exit(0);
            }
            if (i + 1 == argc)
                throw std::invalid_argument("Option requires a value");
            std::string value = argv[++i];
            if (key == "--data")
                o.data = value;
            else if (key == "--output")
                o.output = value;
            else if (key == "--backend")
            {
                if (value == "auto")
                    o.backend = ml::Backend::Auto;
                else if (value == "cpu")
                    o.backend = ml::Backend::CPU;
                else if (value == "cuda")
                    o.backend = ml::Backend::CUDA;
                else
                    throw std::invalid_argument("Backend must be auto, cpu or cuda");
            }
            else if (key == "--epochs")
                o.epochs = number(value);
            else if (key == "--batch-size")
                o.batch = number(value);
            else if (key == "--train-limit")
                o.train_limit = number(value);
            else if (key == "--test-limit")
                o.test_limit = number(value);
            else if (key == "--validation")
                o.validation = number(value);
            else if (key == "--seed")
            {
                const auto seed = number(value);
                if (seed > (std::numeric_limits<std::uint32_t>::max)())
                    throw std::invalid_argument("Seed out of range");
                o.seed = static_cast<std::uint32_t>(seed);
            }
            else if (key == "--learning-rate")
            {
                std::size_t end = 0;
                o.rate = std::stod(value, &end);
                if (end != value.size() || !std::isfinite(o.rate) || o.rate <= 0)
                    throw std::invalid_argument("Invalid learning rate");
            }
            else
                throw std::invalid_argument("Unknown option: " + std::string(key));
        }
        if (o.data.empty() || !o.epochs || !o.batch || !o.validation)
            throw std::invalid_argument(
                "Data directory, positive epochs/batch/validation are required");
        return o;
    }

    void indices(const std::filesystem::path &path, const std::vector<std::size_t> &ids)
    {
        std::ofstream out(path);
        out << "index\n";
        for (const auto i : ids)
            out << i << '\n';
        if (!out)
            throw std::runtime_error("Cannot write split indices");
    }
} // namespace
int main(int argc, char **argv)
{
    try
    {
        using namespace ml;
        const auto options = parse(argc, argv);
        const auto device = initialize_compute(options.backend);
        const char *backend = device.selected == Backend::CUDA ? "CUDA" : "CPU";
        std::cout << "Selected backend=" << backend << " device=" << device.name
                  << " diagnostic=" << device.diagnostic << '\n';

        auto full = std::make_shared<data::MNIST>(
            (options.data / "train-images-idx3-ubyte").string(),
            (options.data / "train-labels-idx1-ubyte").string(), options.train_limit);
        auto test_data = std::make_shared<data::MNIST>(
            (options.data / "t10k-images-idx3-ubyte").string(),
            (options.data / "t10k-labels-idx1-ubyte").string(), options.test_limit);
        if (full->rows() != 28 || full->cols() != 28 || test_data->rows() != 28 ||
            test_data->cols() != 28 || options.validation >= full->size())
            throw std::invalid_argument(
                "Expected 28x28 MNIST and validation smaller than loaded training split");
        auto split = data::random_split(full->size(), options.validation, 0, options.seed);
        auto train_data = std::make_shared<data::Subset>(full, split.training),
             valid_data = std::make_shared<data::Subset>(full, split.validation);
        std::string identity = "mnist:first=" + std::to_string(full->size()) +
                               ":validation=" + std::to_string(options.validation) +
                               ":seed=" + std::to_string(options.seed);
        data::DataLoader train(train_data,
                               {options.batch, true, false, options.seed, identity + ":train"}),
            valid(valid_data,
                  {options.batch, false, false, options.seed, identity + ":validation"}),
            test(test_data, {options.batch, false, false, options.seed, "mnist:official-test"});
        nn::Sequential model;
        model.add(std::make_shared<nn::Conv2D>(1, 4, nd::Shape{3, 3}, nd::Shape{1, 1},
                                               nd::Shape{0, 0}, nd::Shape{1, 1}, options.seed));
        model.add(std::make_shared<nn::ActivationLayer>(Activation::Relu));
        model.add(std::make_shared<nn::Pool2D>(nd::Shape{2, 2}, nd::Shape{2, 2}));
        model.add(std::make_shared<nn::Flatten>());
        model.add(std::make_shared<nn::Linear>(4 * 13 * 13, 10, options.seed + 1));
        OptimizerOptions optimizer;
        optimizer.kind = OptimizerKind::Adam;
        model.to(device.selected);
        if (model.backend() != device.selected)
            throw std::logic_error("Model migration did not match selected backend");
        training::SupervisedTrainer trainer(model, optimizer);
        auto forward = [&](const data::Batch &batch)
        { return model.forward(std::get<nd::Array>(batch.input).to(device.selected)); };
        std::filesystem::create_directories(options.output);
        std::ofstream csv(options.output / "epochs.csv"), config(options.output / "config.txt");
        if (!csv || !config)
            throw std::runtime_error("Cannot write run output files");
        csv << std::setprecision(17)
            << "epoch,learning_rate,train_loss,train_accuracy,validation_loss,validation_accuracy,"
               "validation_macro_f1,training_targets,validation_targets\n";
        config << std::setprecision(17)
               << "model=Conv2D(1,4,3)-ReLU-Pool2D(2,2)-Flatten-Linear(676,10)\nbackend=" << backend
               << "\ndevice=" << device.name << "\ncuda_compiled=" << device.cuda_compiled
               << "\nmatmul="
               << (device.selected == Backend::CUDA ? gpu::resident::matmul_implementation()
                                                    : "CPU reference matmul")
               << "\ndtype="
                  "float64\nseed="
               << options.seed << "\nloaded_train=" << full->size()
               << "\nvalidation=" << valid_data->size() << "\ntest=" << test_data->size()
               << "\nepochs=" << options.epochs << "\nbatch_size=" << options.batch
               << "\nlearning_rate=" << options.rate
               << "\noptimizer=Adam\nsource=" << options.data.string() << '\n';
        indices(options.output / "training_indices.csv", split.training);
        indices(options.output / "validation_indices.csv", split.validation);
        training::FitOptions fit;
        fit.epochs = options.epochs;
        fit.classes = 10;
        fit.learning_rate = options.rate;
        fit.max_grad_norm = 5;
        fit.patience = 3;
        fit.on_epoch = [&](const training::EpochResult &r)
        {
            csv << r.epoch << ',' << r.learning_rate << ',' << r.training.loss << ','
                << r.training.accuracy << ',' << r.validation.loss << ',' << r.validation.accuracy
                << ',' << r.validation.macro_f1 << ',' << r.training.count << ','
                << r.validation.count << '\n';
            csv.flush();
            if (!csv)
                throw std::runtime_error("Metrics write failed");
            std::cout << "epoch=" << r.epoch << " train_loss=" << r.training.loss
                      << " validation_loss=" << r.validation.loss
                      << " validation_accuracy=" << r.validation.accuracy << '\n'
                      << std::flush;
        };
        std::cout << backend << " MNIST: train=" << train_data->size()
                  << " validation=" << valid_data->size() << " test=" << test_data->size()
                  << " seed=" << options.seed << '\n';
        auto result = trainer.fit(train, valid, forward, fit);
        auto final_validation = trainer.evaluate(valid, forward, 10);
        auto final_test = trainer.evaluate(test, forward, 10);
        nn::save_checkpoint((options.output / "best.nn").string(), model, &trainer.optimizer());
        train.save_state((options.output / "training.loader").string());
        std::ofstream report(options.output / "metrics.txt");
        report << std::setprecision(17) << "best_epoch=" << result.best_epoch
               << "\nvalidation_loss=" << final_validation.loss
               << "\nvalidation_accuracy=" << final_validation.accuracy
               << "\ntest_loss=" << final_test.loss << "\ntest_accuracy=" << final_test.accuracy
               << "\ntest_macro_f1=" << final_test.macro_f1 << "\ntest_targets=" << final_test.count
               << '\n';
        config.flush();
        if (!config || !report)
            throw std::runtime_error("Run report write failed");
        std::cout << "best_epoch=" << result.best_epoch << " test_loss=" << final_test.loss
                  << " test_accuracy=" << final_test.accuracy
                  << " test_macro_f1=" << final_test.macro_f1 << "\nSaved run to "
                  << options.output.string() << '\n';
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "MNIST: " << e.what() << "\nUse --help for options.\n";
        return 1;
    }
}
