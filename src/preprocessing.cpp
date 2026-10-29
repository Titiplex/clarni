#include "preprocessing.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ml {
DatasetSplit split_dataset(const Dataset& data,double validation,double test,std::uint32_t seed,bool shuffle){
    if(data.empty()||!std::isfinite(validation)||!std::isfinite(test)||validation<0||test<0||validation+test>=1)throw std::invalid_argument("Invalid split fractions");
    const auto nval=static_cast<std::size_t>(validation*data.size()),ntest=static_cast<std::size_t>(test*data.size());
    const auto ntrain=data.size()-nval-ntest;
    std::vector<std::size_t> order(data.size());std::iota(order.begin(),order.end(),0);
    if(shuffle){std::mt19937 generator(seed);std::shuffle(order.begin(),order.end(),generator);}
    std::vector<Sample> train_samples,val_samples,test_samples;
    for(std::size_t i=0;i<data.size();++i){if(i<ntrain)train_samples.push_back(data[order[i]]);else if(i<ntrain+nval)val_samples.push_back(data[order[i]]);else test_samples.push_back(data[order[i]]);}
    return {Dataset(train_samples),Dataset(val_samples),Dataset(test_samples)};
}
void StandardScaler::fit(const Dataset& training){
    if(training.empty()||training[0].input.empty())throw std::invalid_argument("Scaler requires training samples/features");
    const auto dim=training[0].input.size();std::vector<double> mean(dim,0),m2(dim,0);std::size_t count=0;
    for(const auto& sample:training){if(sample.input.size()!=dim)throw std::invalid_argument("Scaler dimensions");++count;
        for(std::size_t i=0;i<dim;++i){double x=sample.input[i];if(!std::isfinite(x))throw std::invalid_argument("Non-finite scaler input");
            double delta=x-mean[i];mean[i]+=delta/static_cast<double>(count);m2[i]+=delta*(x-mean[i]);}}
    std::vector<double> scale(dim);for(std::size_t i=0;i<dim;++i){scale[i]=std::sqrt(std::max(0.0,m2[i]/static_cast<double>(count)));
        if(!std::isfinite(mean[i])||!std::isfinite(scale[i]))throw std::overflow_error("Scaler statistics overflow");
        if(scale[i]<1e-12)scale[i]=1;}
    mean_=std::move(mean);scale_=std::move(scale);
}
std::vector<double> StandardScaler::transform(const std::vector<double>& input)const{
    if(mean_.empty()||input.size()!=mean_.size())throw std::invalid_argument("Unfitted scaler or invalid dimension");
    auto result=input;for(std::size_t i=0;i<result.size();++i){if(!std::isfinite(input[i]))throw std::invalid_argument("Non-finite scaler input");result[i]=(input[i]-mean_[i])/scale_[i];if(!std::isfinite(result[i]))throw std::overflow_error("Scaled value overflow");}return result;
}
Dataset StandardScaler::transform(const Dataset& data)const{
    std::vector<Sample> result;result.reserve(data.size());for(const auto& sample:data)result.push_back({transform(sample.input),sample.target});return Dataset(result);
}
void StandardScaler::save(const std::string& path)const{
    if(mean_.empty())throw std::logic_error("Cannot save unfitted scaler");
    std::ofstream file(path);if(!file)throw std::runtime_error("Cannot open scaler file");
    file<<std::setprecision(17)<<"CLARNI_SCALER 1\n"<<mean_.size()<<'\n';for(std::size_t i=0;i<mean_.size();++i)file<<mean_[i]<<' '<<scale_[i]<<'\n';if(!file)throw std::runtime_error("Scaler write failed");
}
StandardScaler StandardScaler::load(const std::string& path){
    std::ifstream file(path);std::string tag;int version=0;std::size_t count=0;StandardScaler out;
    if(!(file>>tag>>version>>count)||tag!="CLARNI_SCALER"||version!=1||count==0||count>1000000)throw std::runtime_error("Invalid scaler file");
    out.mean_.resize(count);out.scale_.resize(count);for(std::size_t i=0;i<count;++i)if(!(file>>out.mean_[i]>>out.scale_[i])||!std::isfinite(out.mean_[i])||!std::isfinite(out.scale_[i])||out.scale_[i]<=0)throw std::runtime_error("Invalid scaler statistics");
    file>>std::ws;if(!file.eof())throw std::runtime_error("Trailing scaler data");return out;
}
}
