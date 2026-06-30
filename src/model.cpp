#include "model.hpp"
#include <cmath>

namespace causal_chess {

TransformerBlockImpl::TransformerBlockImpl(int embed_dim, int num_heads, int ff_dim) {
    q_proj = register_module("q_proj", torch::nn::Linear(embed_dim, embed_dim));
    k_proj = register_module("k_proj", torch::nn::Linear(embed_dim, embed_dim));
    v_proj = register_module("v_proj", torch::nn::Linear(embed_dim, embed_dim));
    out_proj = register_module("out_proj", torch::nn::Linear(embed_dim, embed_dim));

    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));

    ffn = register_module("ffn", torch::nn::Sequential(
        torch::nn::Linear(embed_dim, ff_dim),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(ff_dim, embed_dim)
    ));

    this->num_heads = num_heads;
    this->embed_dim = embed_dim;
    this->head_dim = embed_dim / num_heads;
}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x) {
    int64_t batch = x.size(0);
    int64_t seq_len = x.size(1);

    // 1. Attention Block with residual connection & LayerNorm
    torch::Tensor norm_x = norm1->forward(x);

    torch::Tensor q = q_proj->forward(norm_x).reshape({batch, seq_len, num_heads, head_dim}).transpose(1, 2);
    torch::Tensor k = k_proj->forward(norm_x).reshape({batch, seq_len, num_heads, head_dim}).transpose(1, 2);
    torch::Tensor v = v_proj->forward(norm_x).reshape({batch, seq_len, num_heads, head_dim}).transpose(1, 2);

    torch::Tensor scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(head_dim);
    torch::Tensor attn = torch::softmax(scores, -1);
    torch::Tensor context = torch::matmul(attn, v).transpose(1, 2).reshape({batch, seq_len, embed_dim});

    torch::Tensor attn_out = out_proj->forward(context);
    x = x + attn_out;

    // 2. Feed-Forward Block with residual connection & LayerNorm
    x = x + ffn->forward(norm2->forward(x));

    return x;
}

ValueNetworkImpl::ValueNetworkImpl(int in_channels) {
    // 1. CNN Projection to 128 channels
    conv_layers = register_module("conv_layers", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
        torch::nn::Functional(torch::relu)
    ));

    // 2. Positional Embedding
    norm0 = register_module("norm0", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
    pos_embedding = register_parameter("pos_embedding", torch::zeros({1, 64, 128}));

    // 3. Transformer Blocks (N layers)
    transformer_blocks = register_module("transformer_blocks", torch::nn::Sequential(
        TransformerBlock(128, 4, 256),
        TransformerBlock(128, 4, 256),
        TransformerBlock(128, 4, 256),
        TransformerBlock(128, 4, 256)
    ));

    // 4. Value Head (FC layers)
    fc_layers = register_module("fc_layers", torch::nn::Sequential(
        torch::nn::Linear(128, 64),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(64, 1),
        torch::nn::Functional(torch::sigmoid)
    ));

    // Initialize weights
    {
        torch::NoGradGuard no_grad;
        torch::nn::init::normal_(pos_embedding, 0.0, 0.02);
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
    x = x.flatten(2).transpose(1, 2); // shape: (batch, 64, 128)
    x = norm0->forward(x); // add extra normalization
    x = x + pos_embedding;
    x = transformer_blocks->forward(x);
    x = x.mean(1); // shape: (batch, 128)
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
