import torch
import torch.nn as nn
import math

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 预计算旋转角度，基于 Qwen2 的 RoPE 实现
        theta = 10000 ** (-2 * torch.arange(0, d_model//2, dtype=torch.float) / d_model)
        self.register_buffer('theta', theta)
        self.d_model = d_model

    def forward(self, q, k, position_ids):
        # q, k 形状: (batch_size, num_heads, seq_len, d_k)
        # position_ids 形状: (batch_size, seq_len)
        angles = position_ids.unsqueeze(-1) * self.theta  # (batch_size, seq_len, d_model//2)
        rot_sin = torch.sin(angles)
        rot_cos = torch.cos(angles)
        
        # 对 q 和 k 应用旋转
        q_rot = torch.zeros_like(q)
        k_rot = torch.zeros_like(k)
        # 每两个维度一组进行旋转
        q_rot[..., 0::2] = q[..., 0::2] * rot_cos - q[..., 1::2] * rot_sin
        q_rot[..., 1::2] = q[..., 0::2] * rot_sin + q[..., 1::2] * rot_cos
        k_rot[..., 0::2] = k[..., 0::2] * rot_cos - k[..., 1::2] * rot_sin
        k_rot[..., 1::2] = k[..., 0::2] * rot_sin + k[..., 1::2] * rot_cos
        
        return q_rot, k_rot

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for query, key, value, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # RoPE 模块
        self.rotary = RotaryPositionalEncoding(d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        batch_size = Q.size(0)
        
        # Calculate attention scores: (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for padding or causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, Q, K, V, position_ids, mask=None):
        batch_size = Q.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用 RoPE
        Q, K = self.rotary(Q, K, position_ids)
        
        # Apply scaled dot-product attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, position_ids, mask=None):
        # Self-attention with RoPE
        attn_output, _ = self.self_attention(x, x, x, position_ids, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, position_ids, tgt_mask=None):
        # Target embedding
        tgt = self.embedding(tgt)
        tgt = self.dropout(tgt)
        
        # Decoder
        for layer in self.decoder_layers:
            tgt = layer(tgt, position_ids, tgt_mask)
        
        # Final linear layer
        output = self.fc_out(tgt)
        return output

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    batch_size = 32
    seq_len = 10
    
    # Create model
    model = TransformerDecoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # Dummy input
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(0).expand(batch_size, seq_len)
    
    # Causal mask for autoregressive decoding
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    # Forward pass
    output = model(tgt, position_ids, tgt_mask)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, vocab_size)
