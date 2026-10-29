#include "network.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace ml {
namespace {
void write_tensor(std::ostream& file,const Tensor& tensor){
    const auto matrix=tensor.to_matrix();file<<matrix.get_x()<<' '<<matrix.get_y()<<'\n';
    for(int i=0;i<matrix.get_x();++i){for(int j=0;j<matrix.get_y();++j)file<<matrix(i,j)<<' ';file<<'\n';}
}
Tensor read_tensor(std::istream& file,std::size_t& budget){
    int rows=0,cols=0;if(!(file>>rows>>cols)||rows<0||cols<0)throw std::runtime_error("Invalid checkpoint tensor shape");
    const auto r=static_cast<std::size_t>(rows),c=static_cast<std::size_t>(cols);
    if(c&&r>budget/c)throw std::runtime_error("Checkpoint tensor budget exceeded");
    budget-=r*c;
    math::Matrix matrix(rows,cols);for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)
        if(!(file>>matrix(i,j))||!std::isfinite(matrix(i,j)))throw std::runtime_error("Invalid checkpoint scalar");
    return Tensor::from_matrix(matrix);
}
}
void NeuralNetwork::save(const std::string& path)const{
    validate_topology();std::ofstream file(path);if(!file)throw std::runtime_error("Cannot open model checkpoint");
    Optimizer effective=optimizer_;
    for(std::size_t i=0;i<layers.size();++i)if(observed_revisions_[i]!=layers[i].revision())effective.reset(i);
    file<<std::setprecision(std::numeric_limits<double>::max_digits10)<<"CLARNI_MODEL 1\n"
        <<layer_nb<<' '<<layers.size()<<' '<<learning_rate<<' '<<learning_type<<' '<<error_type_<<' '<<l2_<<'\n';
    const auto& options=effective.options();file<<static_cast<int>(options.kind)<<' '<<options.momentum<<' '<<options.beta1<<' '<<options.beta2<<' '<<options.epsilon<<'\n';
    for(const auto& layer:layers){file<<layer.input_size()<<' '<<layer.output_size()<<' '<<layer.activation_type()<<'\n';write_tensor(file,layer.tensor_weights());}
    file<<effective.states().size()<<'\n';
    for(const auto& state:effective.states()){
        file<<state.steps<<' '<<state.rows<<' '<<state.cols<<'\n';write_tensor(file,state.first);write_tensor(file,state.second);
    }
    file<<generator<<"\nEND\n";if(!file)throw std::runtime_error("Model checkpoint write failed");
}
NeuralNetwork NeuralNetwork::load(const std::string& path){
    std::ifstream file(path);std::string tag;int version=0,type=0,error=0,kind=0;std::size_t maximum=0,count=0;double rate=0,l2=0;
    if(!(file>>tag>>version)||tag!="CLARNI_MODEL"||version!=1)throw std::runtime_error("Unsupported model checkpoint");
    if(!(file>>maximum>>count>>rate>>type>>error>>l2)||maximum==0||maximum>1024||count==0||count>maximum)throw std::runtime_error("Invalid model architecture");
    OptimizerOptions options;if(!(file>>kind>>options.momentum>>options.beta1>>options.beta2>>options.epsilon))throw std::runtime_error("Truncated optimizer options");
    options.kind=static_cast<OptimizerKind>(kind);
    NeuralNetwork result(maximum,rate,static_cast<LearningType>(type),static_cast<Error>(error));result.set_optimizer(options);result.set_l2(l2);
    std::size_t budget=30000000;
    for(std::size_t i=0;i<count;++i){int inputs=0,outputs=0,activation=0;
        if(!(file>>inputs>>outputs>>activation)||inputs<=0||outputs<=0||inputs==(std::numeric_limits<int>::max)()
           ||static_cast<std::size_t>(outputs)>budget/(static_cast<std::size_t>(inputs)+1))throw std::runtime_error("Invalid layer dimensions");
        auto weights=read_tensor(file,budget);if(weights.rows()!=outputs||weights.cols()!=inputs+1)throw std::runtime_error("Checkpoint weight shape mismatch");
        result.add_layer(inputs,outputs,static_cast<Layer::Activation>(activation));result[i].set_weights(weights.to_matrix());result.observed_revisions_[i]=result[i].revision();
    }
    std::size_t states=0;if(!(file>>states)||states>count)throw std::runtime_error("Invalid optimizer state count");
    std::vector<Optimizer::State> restored(states);
    for(std::size_t i=0;i<states;++i){auto& state=restored[i];if(!(file>>state.steps>>state.rows>>state.cols))throw std::runtime_error("Truncated optimizer state");
        state.first=read_tensor(file,budget);state.second=read_tensor(file,budget);
        if(state.steps==0){if(state.rows||state.cols||state.first.size()||state.second.size())throw std::runtime_error("Invalid empty optimizer state");}
        else {
            if(state.rows!=result[i].output_size()||state.cols!=result[i].input_size()+1)throw std::runtime_error("Optimizer parameter shape mismatch");
            if(options.kind==OptimizerKind::SGD&&(state.first.size()||state.second.size()))throw std::runtime_error("Unexpected SGD moments");
            if(options.kind==OptimizerKind::Momentum&&state.second.size())throw std::runtime_error("Unexpected momentum second moment");
        }
    }
    result.optimizer_.restore_states(restored);
    if(!(file>>result.generator>>tag)||tag!="END")throw std::runtime_error("Invalid checkpoint RNG/terminator");
    file>>std::ws;if(!file.eof())throw std::runtime_error("Trailing model checkpoint data");result.validate_topology();return result;
}
}
