#pragma once
#include <vector>

namespace ml
{
    struct Sample
    {
        std::vector<double> input, target;
    };

    class Dataset
    {
        std::vector<Sample> data;

    public:
        explicit Dataset(const std::vector<Sample>& data) : data(data)
        {
        }

        [[nodiscard]] std::size_t size() const
        {
            return data.size();
        }

        [[nodiscard]] bool empty() const
        {
            return data.empty();
        }

        auto begin() { return data.begin(); }
        auto end() { return data.end(); }
        [[nodiscard]] auto cbegin() const { return data.cbegin(); }
        [[nodiscard]] auto cend() const { return data.cend(); }
        [[nodiscard]] auto begin() const { return data.begin(); }
        [[nodiscard]] auto end() const { return data.end(); }

        const Sample& operator[](const std::size_t i) const
        {
            return data.at(i);
        }
    };
}
