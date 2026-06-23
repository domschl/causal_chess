#include "model.hpp"
#include <cmath>

namespace causal_chess {

ValueNetworkImpl::ValueNetworkImpl(int in_channels) {
    // Define Conv layers
    conv_layers = register_module("conv_layers", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::AdaptiveAvgPool2d(1) // Output shape: (batch, 128, 1, 1)
    ));

    // Define Fully Connected layers
    fc_layers = register_module("fc_layers", torch::nn::Sequential(
        torch::nn::Flatten(),
        //torch::nn::Functional(torch::sigmoid),
        torch::nn::Linear(128, 64),
        //torch::nn::Linear(64, 64),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(64, 1),
        torch::nn::Functional(torch::sigmoid) // Output range [0, 1]
    ));

    // Initialize Conv2d and Linear layers with Orthogonal weights
    {
        torch::NoGradGuard no_grad;
        for (auto& module : modules(/*include_self=*/false)) {
            if (auto* conv = module->as<torch::nn::Conv2dImpl>()) {
                torch::nn::init::orthogonal_(conv->weight, std::sqrt(2.0));
                if (conv->options.bias()) {
                    torch::nn::init::constant_(conv->bias, 0.0);
                }
            } else if (auto* linear = module->as<torch::nn::LinearImpl>()) {
                double gain = (linear->weight.size(0) == 1) ? 1.0 : std::sqrt(2.0);
                torch::nn::init::orthogonal_(linear->weight, gain);
                if (linear->options.bias()) {
                    torch::nn::init::constant_(linear->bias, 0.0);
                }
            }
        }
    }
}

torch::Tensor ValueNetworkImpl::forward(torch::Tensor x) {
    x = conv_layers->forward(x);
    return fc_layers->forward(x);
}

int64_t ValueNetworkImpl::param_count() {
    int64_t count = 0;
    for (const auto& p : parameters()) {
        if (p.requires_grad()) {
            count += p.numel();
        }
    }
    return count;
}

} // namespace causal_chess
