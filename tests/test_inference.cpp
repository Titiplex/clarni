#include "module.h"
#include "test_support.h"
#include <thread>
using namespace ml;
int main()
{
    return run_tests({{"NoGrad nested, exceptions, thread isolation and eval independence",
                       []
                       {
                           nn::Linear m(1, 1);
                           m.eval();
                           auto x = nd::Array::from({1, 1}, {2}, true);
                           require(m.forward(x).variable().requires_grad(),
                                   "eval disabled autograd");
                           std::vector<double> expected = m.forward(x).values();
                           {
                               ad::NoGradGuard guard;
                               require(!ad::grad_enabled(), "guard inactive");
                               require(!m.forward(x).variable().requires_grad(), "graph recorded");
                               require(m.forward(x).values() == expected, "prediction changed");
                               {
                                   ad::NoGradGuard nested;
                                   require(!ad::grad_enabled(), "nested guard");
                               }
                               require(!ad::grad_enabled(), "nested restoration");
                               bool thread_recording = false;
                               std::thread worker([&] { thread_recording = ad::grad_enabled(); });
                               worker.join();
                               require(thread_recording, "not thread local");
                           }
                           require(ad::grad_enabled(), "not restored");
                           try
                           {
                               ad::NoGradGuard guard;
                               throw std::runtime_error("test");
                           }
                           catch (const std::runtime_error &)
                           {
                           }
                           require(ad::grad_enabled(), "exception restoration");
                       }},
                      {"iterative backward and destruction of a deep graph",
                       []
                       {
                           ad::Variable x(Tensor(1, 1, 2));
                           auto result = x;
                           for (int i = 0; i < 40000; ++i)
                               result = ad::affine(result, 1, .00001);
                           result.backward();
                           near(x.grad().item(), 1);
                           near(result.value().item(), 2.4, 1e-8);
                           result.backward();
                           near(x.grad().item(), 1);
                       }},
                      {"shared branch and first contribution semantics", []
                       {
                           ad::Variable x(Tensor(1, 1, 3));
                           auto y = ad::square(x);
                           ad::add(y, y).backward();
                           near(x.grad().item(), 12);
                       }}});
}
