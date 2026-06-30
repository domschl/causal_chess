#ifndef CAUSAL_CHESS_MODEL_HPP
#define CAUSAL_CHESS_MODEL_HPP

#include <torch/torch.h>

namespace causal_chess {

class TransformerBlockImpl : public torch::nn::Module {
public:
    TransformerBlockImpl(int embed_dim, int num_heads, int ff_dim);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, out_proj{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Sequential ffn{nullptr};
    int num_heads;
    int embed_dim;
    int head_dim;
};

TORCH_MODULE(TransformerBlock);

class ValueNetworkImpl : public torch::nn::Module {
public:
    ValueNetworkImpl(int in_channels = 15);

    torch::Tensor forward(torch::Tensor x);

    int64_t param_count();

private:
    torch::nn::Sequential conv_layers;
    torch::nn::LayerNorm norm0{nullptr};
    torch::Tensor pos_embedding;
    torch::nn::Sequential transformer_blocks;
    torch::nn::Sequential fc_layers;
};

TORCH_MODULE(ValueNetwork);

} // namespace causal_chess

#endif // CAUSAL_CHESS_MODEL_HPP
