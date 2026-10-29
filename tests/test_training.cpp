#include "test_support.h"
#include "network.h"
#include "preprocessing.h"
using namespace ml;
int main(){initialize_compute(Backend::CPU);return run_tests({
    {"scaler_fitted_only_on_training",[]{Dataset training(std::vector<Sample>{{{1,5},{0}},{{3,5},{1}}});StandardScaler scaler;scaler.fit(training);
        auto val=scaler.transform(std::vector<double>{100,5});near(scaler.mean()[0],2);near(scaler.scale()[0],1);near(val[0],98);near(val[1],0);
        auto data=scaler.transform(training);near(data[0].input[0],-1);near(data[1].input[0],1);near(data[1].target[0],1);}},
    {"splits_are_disjoint_and_temporal_order_preserved",[]{std::vector<Sample> samples;for(int i=0;i<10;++i)samples.push_back({{double(i)},{0}});
        auto split=split_dataset(Dataset(samples),0.2,0.2,42,false);require(split.training.size()==6&&split.validation.size()==2&&split.test.size()==2,"split sizes");near(split.training[5].input[0],5);near(split.validation[0].input[0],6);near(split.test[0].input[0],8);}},
    {"l2_excludes_bias_and_is_not_scaled_twice",[]{NeuralNetwork net(1,0.1,NeuralNetwork::BATCH);net.add_layer(1,1,Layer::IDENTITY);
        math::Matrix w(1,2);w(0,0)=3;w(0,1)=2;net[0].set_weights(w);net.set_l2(0.5);Dataset data(std::vector<Sample>{{{0},{3}},{{0},{3}}});
        near(net.objective_loss(data),1);net.learn(data,1);near(net[0].get_weights()(0,0),3);near(net[0].get_weights()(0,1),1.9);}},
    {"early_stopping_restores_best_complete_state",[]{NeuralNetwork net(1,0.1,NeuralNetwork::BATCH);net.add_layer(1,1,Layer::IDENTITY);net[0].set_weights(math::Matrix(1,2,0));
        Dataset train(std::vector<Sample>{{{1},{1}}}),val(std::vector<Sample>{{{1},{-1}}});FitOptions option;option.epochs=20;option.patience=2;option.shuffle=false;
        auto history=net.fit(train,val,option);require(history.training_loss.size()==2&&history.best_epoch==0,"patience/best epoch");near(net.output({1})[0],0);require(net.optimizer().states().empty(),"best optimizer state not restored");}}
});}
