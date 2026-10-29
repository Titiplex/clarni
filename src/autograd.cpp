#include "autograd.h"
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ml::ad {
struct Node {
    Tensor value,gradient;
    bool requires_gradient=false,has_gradient=false;
    using Edge=std::pair<std::shared_ptr<Node>,std::function<Tensor(const Tensor&)>>;
    std::vector<Edge> parents;
    Node(Tensor v,bool needs_gradient):value(std::move(v)),requires_gradient(needs_gradient){}
};
namespace {
std::shared_ptr<Node> node(Tensor value,std::vector<Node::Edge> edges){
    auto result=std::make_shared<Node>(std::move(value),false);
    for(auto& edge:edges)if(edge.first->requires_gradient){result->parents.push_back(std::move(edge));result->requires_gradient=true;}
    return result;
}
}
Variable::Variable(Tensor value,bool requires_gradient):node_(std::make_shared<Node>(std::move(value),requires_gradient)){}
Variable::Variable(std::shared_ptr<Node> value):node_(std::move(value)){}
const Tensor& Variable::value()const{return node_->value;}
const Tensor& Variable::grad()const{if(!node_->has_gradient)throw std::logic_error("backward has not reached variable");return node_->gradient;}
void Variable::backward(){if(value().size()!=1)throw std::invalid_argument("Non-scalar backward requires seed");backward(Tensor(value().rows(),value().cols(),1,value().backend()));}
void Variable::backward(const Tensor& seed){
    if(!node_->requires_gradient||seed.rows()!=value().rows()||seed.cols()!=value().cols()||seed.backend()!=value().backend()||!seed.all_finite())throw std::invalid_argument("Invalid backward seed/root");
    std::vector<std::shared_ptr<Node>> order;std::unordered_set<Node*> visited;
    std::function<void(const std::shared_ptr<Node>&)> visit=[&](const auto& current){if(!visited.insert(current.get()).second)return;for(const auto& edge:current->parents)visit(edge.first);order.push_back(current);};
    visit(node_);
    for(auto& current:order){current->gradient=Tensor(current->value.rows(),current->value.cols(),0,current->value.backend());current->has_gradient=true;}
    node_->gradient=seed;
    for(auto it=order.rbegin();it!=order.rend();++it)for(const auto& [parent,pullback]:(*it)->parents)parent->gradient=parent->gradient+pullback((*it)->gradient);
}
Variable add(const Variable& a,const Variable& b){return Variable(node(a.value()+b.value(),{{a.node_,[](const Tensor& g){return g;}},{b.node_,[](const Tensor& g){return g;}}}));}
Variable affine(const Variable& a,double scale,double offset){return Variable(node(a.value().affine(scale,offset),{{a.node_,[scale](const Tensor& g){return g.affine(scale);}}}));}
Variable transpose(const Variable& a){return Variable(node(a.value().transpose(),{{a.node_,[](const Tensor& g){return g.transpose();}}}));}
Variable matmul(const Variable& a,const Variable& b){
    return Variable(node(a.value().matmul(b.value()),{{a.node_,[v=b.value()](const Tensor& g){return g.matmul(v.transpose());}},{b.node_,[v=a.value()](const Tensor& g){return v.transpose().matmul(g);}}}));
}
Variable hadamard(const Variable& a,const Variable& b){return Variable(node(a.value().hadamard(b.value()),{{a.node_,[v=b.value()](const Tensor& g){return g.hadamard(v);}},{b.node_,[v=a.value()](const Tensor& g){return g.hadamard(v);}}}));}
Variable broadcast_columns(const Variable& a,int count){return Variable(node(a.value().broadcast_columns(count),{{a.node_,[](const Tensor& g){return g.sum_columns();}}}));}
Variable sum(const Variable& a){
    return Variable(node(a.value().sum_columns().sum_rows(),{{a.node_,[rows=a.value().rows(),cols=a.value().cols()](const Tensor& g){return g.broadcast_rows(rows).broadcast_columns(cols);}}}));
}
Variable mean(const Variable& a){if(a.value().size()==0)throw std::invalid_argument("Empty mean");return affine(sum(a),1.0/static_cast<double>(a.value().size()));}
Variable square(const Variable& a){return hadamard(a,a);}
Variable activate(const Variable& a,Activation activation){
    auto output=a.value().activate(activation);
    return Variable(node(output,{{a.node_,[input=a.value(),output,activation](const Tensor& g){return input.activation_backward(output,g,activation);}}}));
}
Variable loss(const Variable& a,const Tensor& target,LossKind kind){
    auto result=a.value().loss(target,kind);const double factor=1.0/result.values.cols();
    return Variable(node(result.values.sum_columns().affine(factor),{{a.node_,[gradient=result.gradient,factor](const Tensor& g){return gradient.affine(factor).hadamard(g.broadcast_rows(gradient.rows()).broadcast_columns(gradient.cols()));}}}));
}
}
