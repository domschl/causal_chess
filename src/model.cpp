#include "model.hpp"

namespace causal_chess {

ValueNetworkImpl::ValueNetworkImpl(int in_channels) {
    // Define Conv layers (with GroupNorm for stable online updates)
    conv_layers = register_module("conv_layers", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 32, 3).padding(1)),
        torch::nn::GroupNorm(torch::nn::GroupNormOptions(8, 32)),
        torch::nn::Functional(torch::relu),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
        torch::nn::GroupNorm(torch::nn::GroupNormOptions(8, 128)),
        torch::nn::Functional(torch::relu),
        torch::nn::AdaptiveAvgPool2d(1) // Output shape: (batch, 128, 1, 1)
    ));

    // Define Fully Connected layers
    fc_layers = register_module("fc_layers", torch::nn::Sequential(
        torch::nn::Flatten(),
        torch::nn::Linear(128, 64),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(64, 1),
        torch::nn::Functional(torch::sigmoid) // Output range [0, 1]
    ));
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
