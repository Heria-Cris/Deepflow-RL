# core/model_spec.py
import json


class TransformerLayerSpec:
    def __init__(self, layer_id, config):
        self.layer_id = layer_id
        self.h = config["hidden_size"]
        self.i = config["intermediate_size"]
        self.num_heads = config["num_heads"]
        self.bytes_per_param = config["bytes_per_param"]
        self.seq_cap = config["max_position_embeddings"]

    def get_parameter_count(self):
        """
        精准计算单层参数量。
        1. Attention: 4个矩阵 (W_q, W_k, W_v, W_o) -> 4 * h^2
        2. MLP (LLaMA SwiGLU结构): 3个矩阵 (Gate, Up, Down) -> 3 * h * i
        3. LayerNorms (2个): 2 * h (可忽略不计，但为了严谨保留)
        """
        attn_params = 4 * (self.h ** 2)
        mlp_params = 3 * (self.h * self.i)
        ln_params = 2 * self.h
        return attn_params + mlp_params + ln_params

    def get_flops(self, batch_size, seq_len, is_decoding=False):
        """
        计算前向传播 FLOPs (GFLOPs)。

        论文引用: "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"

        公式:
        - GEMM FLOPs = 2 * Parameters * Tokens
        - Attention Score FLOPs = 4 * B * S^2 * H (Prefill阶段) 或 4 * B * S * H (Decode阶段)
        """
        # 1. 线性层 (Linear Projections + MLP)
        # FLOPs = 2 * N_params * (Batch * Seq)
        # 注意：这里我们用具体的矩阵维度计算，不依赖 get_parameter_count 以示严谨

        # Attn Linear: 4 * [B, S, H] * [H, H] -> 4 * 2 * B * S * H^2
        # MLP Linear:  3 * [B, S, H] * [H, I] -> 3 * 2 * B * S * H * I
        gemm_flops = 2 * batch_size * seq_len * (4 * self.h ** 2 + 3 * self.h * self.i)

        # 2. Attention Mechanism (Q * K^T and Score * V)
        # Prefill (S x S): 4 * B * S^2 * H
        # Decode  (1 x S): 4 * B * 1 * S_past * H  (S_past 约等于 seq_len)
        if is_decoding:
            # Decode 阶段，seq_len 通常指当前的长度，但查询的是 Kv Cache
            attn_ops_flops = 4 * batch_size * 1 * seq_len * self.h
        else:
            # Prefill 阶段，全量计算
            attn_ops_flops = 4 * batch_size * (seq_len ** 2) * self.h

        total_flops = gemm_flops + attn_ops_flops
        return total_flops / 1e9

    def get_activation_memory_mb(self, batch_size, seq_len):
        """
        计算层间传输的激活值大小。
        Output: [Batch, Seq, Hidden]
        """
        elements = batch_size * seq_len * self.h
        return (elements * self.bytes_per_param) / (1024 ** 2)

    def get_kv_cache_increment_mb(self, batch_size):
        """
        Decode 阶段，每生成 1 个 Token，KV Cache 增加的大小。
        Size = 2 (K+V) * Batch * 1 * H * Bytes
        """
        elements = 2 * batch_size * self.h
        return (elements * self.bytes_per_param) / (1024 ** 2)


class LLaMAModel:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.name = self.config.get("model_name", "Unknown")
        self.layers = [TransformerLayerSpec(i, self.config) for i in range(self.config["num_layers"])]

        # 词表层 (Logits)
        self.vocab_size = self.config["vocab_size"]
        self.hidden_size = self.config["hidden_size"]

    def get_token_id_size_mb(self, batch_size, seq_len):
        """
        投机推理传输 Token ID 的大小 (v3.0 核心)
        Int32 = 4 Bytes
        """
        return (batch_size * seq_len * 4) / (1024 ** 2)