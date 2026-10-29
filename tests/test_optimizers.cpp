#include "test_support.h"
#include "optimizer.h"
using namespace ml;

int main()
{
    return run_tests({
        {
            "sgd_and_momentum_hand_steps", []
            {
                Tensor p(1, 1, 1), g(1, 1, 1);
                Optimizer sgd;
                near(sgd.update(0, p, g, 0.1).item(), 0.9);
                OptimizerOptions option;
                option.kind = OptimizerKind::Momentum;
                Optimizer momentum(option);
                p = momentum.update(0, p, g, 0.1);
                near(p.item(), 0.9);
                p = momentum.update(0, p, g, 0.1);
                near(p.item(), 0.71);
            }
        },
        {
            "adam_bias_correction_and_state_keys", []
            {
                OptimizerOptions option;
                option.kind = OptimizerKind::Adam;
                Optimizer adam(option);
                Tensor p(1, 1, 1), g(1, 1, 2);
                p = adam.update(0, p, g, 0.1);
                near(p.item(), 1 - 0.1 * 2 / (2 + 1e-8));
                p = adam.update(0, p, g, 0.1);
                near(p.item(), 1 - 0.2 * 2 / (2 + 1e-8));
                (void)adam.update(1, Tensor(1, 1, 1), g, 0.1);
                require(adam.states()[0].steps == 2 && adam.states()[1].steps == 1,
                        "step counted per epoch/key incorrectly");
                adam.reset(0);
                require(adam.states()[0].steps == 0, "state reset");
            }
        }
    });
}
