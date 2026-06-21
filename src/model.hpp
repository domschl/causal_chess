#ifndef CAUSAL_CHESS_MODEL_HPP
#define CAUSAL_CHESS_MODEL_HPP

#include <torch/torch.h>

namespace causal_chess {

class ValueNetworkImpl : public torch::nn::Module {
public:
    ValueNetworkImpl(int in_channels = 15);

    torch::Tensor forward(torch::Tensor x);

    int64_t param_count();

private:
    torch::nn::Sequential conv_layers;
    torch::nn::Sequential fc_layers;
};

TORCH_MODULE(ValueNetwork);

} // namespace causal_chess

#endif // CAUSAL_CHESS_MODEL_HPP
