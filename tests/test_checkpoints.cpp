#include "test_support.h"
#include "network.h"
#include "preprocessing.h"
#include <chrono>
#include <filesystem>
#include <fstream>
using namespace ml;
struct TemporaryFile{
    std::string path;
    TemporaryFile(const char* name):path((std::filesystem::temp_directory_path()/(std::string(name)+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string()){}
    ~TemporaryFile(){std::error_code error;std::filesystem::remove(path,error);}
};
int main(){initialize_compute(Backend::CPU);return run_tests({
    {"checkpoint_preserves_predictions_optimizer_and_rng",[]{
        NeuralNetwork net(2,0.02,NeuralNetwork::MINI_BATCH);net.add_layer(2,3,Layer::TANH);net.add_layer(3,1,Layer::IDENTITY);
        OptimizerOptions option;option.kind=OptimizerKind::Adam;net.set_optimizer(option);
        Dataset data(std::vector<Sample>{{{0,0},{0}},{{0,1},{1}},{{1,0},{2}},{{1,1},{3}},{{2,1},{5}}});auto uninterrupted=net;uninterrupted.learn(data,17,2,true);net.learn(data,10,2,true);
        TemporaryFile file("clarni_model");net.save(file.path);auto restored=NeuralNetwork::load(file.path);
        near(restored.output({0.3,0.7})[0],net.output({0.3,0.7})[0]);
        net.learn(data,7,2,true);restored.learn(data,7,2,true);
        for(std::size_t i=0;i<2;++i){const auto& a=net[i].get_weights();const auto& b=restored[i].get_weights();const auto& continuous=uninterrupted[i].get_weights();for(int r=0;r<a.get_x();++r)for(int c=0;c<a.get_y();++c){near(a(r,c),b(r,c),1e-13);near(a(r,c),continuous(r,c),1e-13);}}
    }},
    {"checkpoint_resets_stale_moments_after_external_weights_change",[]{
        NeuralNetwork net(1,0.03,NeuralNetwork::ONLINE);net.add_layer(1,1,Layer::IDENTITY);OptimizerOptions option;option.kind=OptimizerKind::Adam;net.set_optimizer(option);
        Dataset data(std::vector<Sample>{{{1},{2}}});net.learn(data,3);net[0].set_weights(math::Matrix(1,2,0));TemporaryFile file("clarni_stale");net.save(file.path);auto restored=NeuralNetwork::load(file.path);
        net.learn(data,1);restored.learn(data,1);near(net.output({1})[0],restored.output({1})[0],1e-13);
    }},
    {"scaler_checkpoint_and_corrupt_model_rejected",[]{
        StandardScaler scaler;scaler.fit(Dataset(std::vector<Sample>{{{1,5},{0}},{{3,5},{1}}}));TemporaryFile file("clarni_scaler");scaler.save(file.path);
        auto loaded=StandardScaler::load(file.path);near(loaded.transform(std::vector<double>{4,5})[0],2);
        TemporaryFile corrupt("clarni_corrupt");{std::ofstream out(corrupt.path);out<<"CLARNI_MODEL 1\n999999999999999 1 0.1 0 0 0\n";}
        expect_throw<std::runtime_error>([&]{(void)NeuralNetwork::load(corrupt.path);});
    }}
});}
