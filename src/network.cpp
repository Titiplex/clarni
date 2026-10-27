#include "network.h"
#include <cmath>
#include <numeric>
#include <random>
#include <ranges>

using namespace ml;
using namespace std;

vector<double> NeuralNetwork::learn(const Dataset& dataset, const size_t max_epochs, size_t batch_size, bool shuffle)
{
    if (layers.empty())
        throw logic_error("Le reseau ne contient aucune couche");

    if (dataset.empty())
        throw invalid_argument("Le jeu de donnees est vide");

    if (batch_size <= 0)
        throw invalid_argument("La taille du groupe doit etre positive");

    batch_size =
        learning_type == ONLINE ? 1 : learning_type == BATCH ? dataset.size() : min(batch_size, dataset.size());

    for (auto& layer : layers)
        layer.zero_grad();

    for (const auto& [input, target] : dataset)
    {
        if (input.size() != layers.front().input_size())
            throw invalid_argument("Incorrect entry size");

        if (target.size() != layers.back().output_size())
            throw invalid_argument("Incorrect target size");
    }

    vector<size_t> order(dataset.size());
    iota(order.begin(), order.end(), size_t{0});

    vector<double> losses;
    losses.reserve(max_epochs);

    random_device rd; // obtain a random number from hardware
    mt19937 eng(rd()); // seed the generator

    // Iterating over epochs
    for (size_t epoch = 0; epoch < max_epochs; ++epoch)
    {
        if (shuffle)
        {
            std::shuffle(order.begin(), order.end(), eng);
        }
        for (size_t begin = 0; begin < dataset.size();)
        {
            // Le dernier groupe peut être plus petit.
            const std::size_t count =
                std::min(batch_size, dataset.size() - begin);

            const std::size_t end = begin + count;

            for (auto& layer : layers)
                layer.zero_grad();

            for (std::size_t p = begin; p < end; ++p)
            {
                const Sample& sample = dataset[order[p]];

                // ---- Propagation avant ----

                math::Vector prediction = math::Vector(sample.input);

                for (auto& layer : layers)
                    prediction = layer.forward(prediction);

                // ---- Dérivée de la perte ----
                //
                // L = 1/2 * somme_k (prediction[k] - target[k])²
                // dL/dprediction[k] = prediction[k] - target[k]

                math::Vector gradient = error(prediction, math::Vector(sample.target));

                // ---- Rétropropagation ----

                for (auto& layer : std::views::reverse(layers)) gradient = layer.backward(gradient);
            }
            // Tous les exemples du groupe ont utilisé les mêmes poids.
            // step() divise les gradients accumulés par count.
            for (auto& layer : layers)
                layer.step(learning_rate, count);

            begin = end;
        }
        // Perte mesurée avec les poids obtenus à la fin de l'époque.
        losses.push_back(mean_loss(dataset));
    }
    return losses;
}

double NeuralNetwork::mean_loss(
    const Dataset& dataset
) const
{
    if (layers.empty())
        throw std::logic_error("Le reseau ne contient aucune couche");

    if (dataset.empty())
        throw std::invalid_argument("Le dataset est vide");

    double total_loss = 0.0;

    for (const auto& [input, target] : dataset)
    {
        // Propagation avec les poids actuels, sans modifier les caches.
        const math::Vector prediction = output(input);

        total_loss += error(
            prediction,
            math::Vector(target)
        ).norm();
    }

    return total_loss / static_cast<double>(dataset.size());
}

math::Vector NeuralNetwork::output(const vector<double>& input) const
{
    auto result = math::Vector(input);

    for (auto& layer : layers)
    {
        result = layer.output(result);
    }

    return result;
}
