#pragma once
#include "dataset.h"
#include <cstdint>
#include <string>
#include <vector>
namespace ml {
struct DatasetSplit {Dataset training,validation,test;};
DatasetSplit split_dataset(const Dataset& data,double validation_fraction,double test_fraction,
                           std::uint32_t seed=42,bool shuffle=true);
class StandardScaler {
    std::vector<double> mean_,scale_;
public:
    void fit(const Dataset& training);
    Dataset transform(const Dataset& data)const;
    std::vector<double> transform(const std::vector<double>& values)const;
    const std::vector<double>& mean()const{return mean_;}
    const std::vector<double>& scale()const{return scale_;}
    void save(const std::string& path)const;
    static StandardScaler load(const std::string& path);
};
}
