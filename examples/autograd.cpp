#include "autograd.h"
#include "optimizer.h"
#include <iostream>

int main()
{
    const ml::ad::Variable x(ml::Tensor(1, 1, 3));
    auto objective = ml::ad::add(ml::ad::square(x), x);
    objective.backward();
    std::cout << "x^2+x at x=3: value=" << objective.value().item() << ", derivative=" << x.grad().item() << '\n';
    return std::abs(x.grad().item() - 7) < 1e-12 ? 0 : 1;
}
