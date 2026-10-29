#include "network.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ml
{
    namespace
    {
        Tensor pack(const Dataset& data, const std::vector<std::size_t>& order, std::size_t begin, std::size_t count,
                    bool target)
        {
            const auto& first = data[order[begin]];
            const auto dim = target ? first.target.size() : first.input.size();
            if (dim > static_cast<std::size_t>((std::numeric_limits<int>::max)()) || count > static_cast<std::size_t>((
                std::numeric_limits<int>::max)()))
                throw std::length_error("Batch too large");
            math::Matrix matrix(static_cast<int>(dim), static_cast<int>(count));
            for (std::size_t j = 0; j < count; ++j)
            {
                const auto& values = target ? data[order[begin + j]].target : data[order[begin + j]].input;
                for (std::size_t i = 0; i < dim; ++i)matrix(static_cast<int>(i), static_cast<int>(j)) = values[i];
            }
            return Tensor::from_matrix(matrix, compute_device().selected);
        }
    }

    NeuralNetwork::NeuralNetwork(std::size_t maximum, double rate, LearningType type, Error error, std::uint32_t seed)
        : layer_nb(maximum), learning_rate(rate), learning_type(type), error_type_(error), generator(seed)
    {
        if (!std::isfinite(rate) || rate <= 0)throw std::invalid_argument("Invalid learning rate");
        if (type != BATCH && type != ONLINE && type != MINI_BATCH && type != AUTO)
            throw std::invalid_argument(
                "Unknown learning type");
        if (error != SIMPLE && error != BINARY_CROSS_ENTROPY && error != CROSS_ENTROPY)
            throw std::invalid_argument(
                "Unknown loss");
        layers.reserve(layer_nb);
    }

    void NeuralNetwork::add_layer(int inputs, int outputs, Layer::Activation type)
    {
        if (layers.size() >= layer_nb)throw std::invalid_argument("Declared layer number too small");
        if (!layers.empty() && layers.back().output_size() != inputs)throw std::invalid_argument("Incompatible layers");
        Layer layer(inputs, outputs, type);
        layer.init_random(generator);
        layers.push_back(std::move(layer));
        observed_revisions_.push_back(layers.back().revision());
    }

    Layer& NeuralNetwork::get(std::size_t i) { return layers.at(i); }
    const Layer& NeuralNetwork::get(std::size_t i) const { return layers.at(i); }
    void NeuralNetwork::set_optimizer(OptimizerOptions options) { optimizer_ = Optimizer(options); }

    LossKind NeuralNetwork::loss_kind() const
    {
        return error_type_ == SIMPLE
                   ? LossKind::Squared
                   : error_type_ == BINARY_CROSS_ENTROPY
                   ? LossKind::BinaryCrossEntropy
                   : LossKind::CrossEntropy;
    }

    void NeuralNetwork::validate_topology() const
    {
        if (layers.empty())throw std::logic_error("Network has no layers");
        for (std::size_t i = 1; i < layers.size(); ++i)
            if (layers[i - 1].output_size() != layers[i].input_size())
                throw
                    std::invalid_argument("Incompatible layers");
        if (error_type_ == BINARY_CROSS_ENTROPY && layers.back().activation_type() != Layer::SIGMOID)
            throw
                std::invalid_argument("BCE requires sigmoid output");
        if (error_type_ == CROSS_ENTROPY && layers.back().activation_type() != Layer::SOFTMAX)
            throw
                std::invalid_argument("Cross entropy requires softmax output");
    }

    void NeuralNetwork::validate_dataset(const Dataset& dataset) const
    {
        validate_topology();
        if (dataset.empty())throw std::invalid_argument("Empty dataset");
        for (const auto& sample : dataset)
        {
            if (sample.input.size() != static_cast<std::size_t>(layers.front().input_size()) || sample.target.size() !=
                static_cast<std::size_t>(layers.back().output_size()))
                throw std::invalid_argument("Dataset dimensions");
            for (double x : sample.input)if (!std::isfinite(x))throw std::invalid_argument("Non-finite input");
            double sum = 0;
            for (double y : sample.target)
            {
                if (!std::isfinite(y) || (error_type_ != SIMPLE && (y < 0 || y > 1)))
                    throw std::invalid_argument(
                        "Invalid target");
                sum += y;
            }
            if (error_type_ == CROSS_ENTROPY && std::abs(sum - 1) > 1e-8)
                throw std::invalid_argument(
                    "Class targets must sum to one");
        }
    }

    Tensor NeuralNetwork::batch_loss_input(const Tensor& input)
    {
        Tensor value = input;
        for (std::size_t i = 0; i < layers.size(); ++i)
        {
            value = layers[i].forward_batch(value);
            if (i + 1 == layers.size() && error_type_ != SIMPLE)return layers[i].cached_logits();
        }
        return value;
    }

    std::vector<double> NeuralNetwork::learn(const Dataset& dataset, std::size_t epochs, std::size_t batch_size,
                                             bool shuffle)
    {
        validate_dataset(dataset);
        switch (learning_type)
        {
        case ONLINE: batch_size = 1;
            break;
        case BATCH: batch_size = dataset.size();
            break;
        case AUTO: if (!batch_size)batch_size = 32;
            batch_size = std::min(batch_size, dataset.size());
            break;
        case MINI_BATCH: if (!batch_size)throw std::invalid_argument("Mini-batch must be positive");
            batch_size = std::min(batch_size, dataset.size());
            break;
        }
        std::vector<std::size_t> order(dataset.size());
        std::iota(order.begin(), order.end(), 0);
        std::vector<double> losses;
        losses.reserve(epochs);
        for (std::size_t epoch = 0; epoch < epochs; ++epoch)
        {
            // Canonical order each epoch: RNG state alone suffices to resume exactly.
            std::iota(order.begin(),order.end(),0);
            if (shuffle)std::shuffle(order.begin(), order.end(), generator);
            for (std::size_t begin = 0; begin < dataset.size();)
            {
                const auto count = std::min(batch_size, dataset.size() - begin);
                for (auto& layer : layers)layer.zero_grad();
                auto prediction = batch_loss_input(pack(dataset, order, begin, count, false));
                auto loss = prediction.loss(pack(dataset, order, begin, count, true), loss_kind());
                auto gradient = loss.gradient;
                for (std::size_t i = layers.size(); i-- > 0;)
                    gradient = layers[i].backward_batch(
                        gradient, i + 1 == layers.size() && error_type_ != SIMPLE);
                for (std::size_t i = 0; i < layers.size(); ++i)
                {
                    if (observed_revisions_[i] != layers[i].revision())
                    {
                        optimizer_.reset(i);
                        observed_revisions_[i] = layers[i].revision();
                    }
                    layers[i].apply(optimizer_, i, learning_rate, count, l2_);
                }
                begin += count;
            }
            losses.push_back(mean_loss(dataset));
        }
        return losses;
    }

    Tensor NeuralNetwork::output_batch(const Tensor& input) const
    {
        validate_topology();
        Tensor result = input;
        for (const auto& layer : layers)result = layer.output_batch(result);
        return result;
    }

    math::Vector NeuralNetwork::output(const std::vector<double>& input)const{
        validate_topology();if(input.size()!=static_cast<std::size_t>(layers.front().input_size()))throw std::invalid_argument("Input dimension");
        math::Matrix matrix(static_cast<int>(input.size()),1);for(std::size_t i=0;i<input.size();++i)matrix(static_cast<int>(i),0)=input[i];
        return math::Vector::convert(output_batch(Tensor::from_matrix(matrix,compute_device().selected)).to_matrix());
    }

    double NeuralNetwork::mean_loss(const Dataset& dataset) const
    {
        validate_dataset(dataset);
        std::vector<std::size_t> order(dataset.size());
        std::iota(order.begin(), order.end(), 0);
        double total = 0;
        for (std::size_t begin = 0; begin < dataset.size();)
        {
            const auto count = std::min(std::size_t{256}, dataset.size() - begin);
            Tensor result = pack(dataset, order, begin, count, false);
            for (std::size_t i = 0; i < layers.size(); ++i)
            {
                if (i + 1 == layers.size() && error_type_ != SIMPLE)result = layers[i].preactivation(result);
                else result = layers[i].output_batch(result);
            }
            total += result.loss(pack(dataset, order, begin, count, true), loss_kind()).values.sum();
            begin += count;
        }
        if (!std::isfinite(total))throw std::overflow_error("Non-finite loss");
        return total / static_cast<double>(dataset.size());
    }

    void NeuralNetwork::set_l2(double coefficient){if(!std::isfinite(coefficient)||coefficient<0)throw std::invalid_argument("Invalid L2 coefficient");l2_=coefficient;}
    double NeuralNetwork::objective_loss(const Dataset& data)const{
        double value=mean_loss(data);if(l2_)for(const auto& layer:layers)value+=0.5*l2_*layer.tensor_weights().zero_first_column().unary(Unary::Square).sum();return value;
    }
    FitResult NeuralNetwork::fit(const Dataset& training,const Dataset& validation,const FitOptions& options){
        validate_dataset(training);validate_dataset(validation);
        if(!std::isfinite(options.min_delta)||options.min_delta<0)throw std::invalid_argument("Invalid early stopping delta");
        set_l2(options.l2);FitResult result;
        double best_value=mean_loss(validation),patience_reference=best_value;std::size_t stale=0;
        NeuralNetwork best=*this;
        for(std::size_t epoch=0;epoch<options.epochs;++epoch){
            const auto loss=learn(training,1,options.batch_size,options.shuffle).back();const double val=mean_loss(validation);
            result.training_loss.push_back(loss);result.validation_loss.push_back(val);
            if(val<best_value){best_value=val;best=*this;result.best_epoch=epoch+1;}
            if(val<patience_reference-options.min_delta){patience_reference=val;stale=0;}else ++stale;
            if(options.patience&&stale>=options.patience)break;
        }
        *this=std::move(best);return result;
    }
}
