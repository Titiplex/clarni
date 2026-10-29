#include "network.h"
#include "preprocessing.h"
#include <exception>
#include <iostream>

int main(int argc,char** argv){try{
    ml::initialize_compute(ml::Backend::Auto);
    std::vector<ml::Sample> samples;
    for(int x=-3;x<=3;++x)for(int y=-3;y<=3;++y)samples.push_back({{double(x),double(y)},{2.0*x-3.0*y+0.7}});
    const auto split=ml::split_dataset(ml::Dataset(samples),0.2,0.2,42,true);
    ml::StandardScaler scaler;scaler.fit(split.training);
    auto training=scaler.transform(split.training),validation=scaler.transform(split.validation),test=scaler.transform(split.test);
    ml::NeuralNetwork net(1,0.05,ml::NeuralNetwork::MINI_BATCH);net.add_layer(2,1,ml::Layer::IDENTITY);
    ml::OptimizerOptions optimizer;optimizer.kind=ml::OptimizerKind::Adam;net.set_optimizer(optimizer);
    ml::FitOptions fit;fit.epochs=500;fit.batch_size=16;fit.patience=40;fit.min_delta=1e-9;
    auto history=net.fit(training,validation,fit);
    std::cout<<"Epochs "<<history.training_loss.size()<<", best "<<history.best_epoch<<", held-out loss "<<net.mean_loss(test)<<'\n';
    if(argc>1){net.save(argv[1]);scaler.save(std::string(argv[1])+".scaler");auto restored=ml::NeuralNetwork::load(argv[1]);auto preprocessing=ml::StandardScaler::load(std::string(argv[1])+".scaler");
        std::cout<<"Prediction reloaded: "<<restored.output(preprocessing.transform(std::vector<double>{0.3,-0.4}))[0]<<" (expected 2.5)\n";}
    return net.mean_loss(test)<1e-3?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
