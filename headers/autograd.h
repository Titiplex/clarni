#pragma once
#include "tensor.h"
#include <memory>

namespace ml::ad {
struct Node;
class Variable {
    std::shared_ptr<Node> node_;
    explicit Variable(std::shared_ptr<Node> node);
    friend Variable add(const Variable&,const Variable&);
    friend Variable matmul(const Variable&,const Variable&);
    friend Variable hadamard(const Variable&,const Variable&);
    friend Variable affine(const Variable&,double,double);
    friend Variable transpose(const Variable&);
    friend Variable broadcast_columns(const Variable&,int);
    friend Variable sum(const Variable&);
    friend Variable activate(const Variable&,Activation);
    friend Variable loss(const Variable&,const Tensor&,LossKind);
public:
    explicit Variable(Tensor value,bool requires_gradient=true);
    const Tensor& value()const;
    const Tensor& grad()const;
    void backward();
    void backward(const Tensor& seed);
};
Variable add(const Variable&,const Variable&);
Variable matmul(const Variable&,const Variable&);
Variable hadamard(const Variable&,const Variable&);
Variable affine(const Variable&,double scale,double offset=0);
Variable transpose(const Variable&);
Variable broadcast_columns(const Variable&,int count);
Variable sum(const Variable&);
Variable mean(const Variable&);
Variable square(const Variable&);
Variable activate(const Variable&,Activation);
Variable loss(const Variable&,const Tensor& target,LossKind);
}
