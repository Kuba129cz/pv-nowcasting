import torch
import math

class CrossAttention(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads!"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_projection = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.key_projection = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.value_projection = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.out_projection = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t_q, _ = query.shape
        _, t_kv, _ = key_value.shape

        query = self.query_projection(query) # [B, T_q, D]
        key = self.key_projection(key_value) # [B, T_kv, D]
        value = self.value_projection(key_value) # [B, T_kv, D]

        query = query.view(b, t_q, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(b, t_kv, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(b, t_kv, self.num_heads, self.head_dim).transpose(1, 2)

        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att_weights = torch.softmax(score, dim=-1)

        att_weights_dropped = self.dropout(att_weights)
        out = torch.matmul(att_weights_dropped, value)
        out = out.transpose(1, 2) # [B, num_heads, T_q, head_dim] -> [B, T_q, num_heads, head_dim]
        out = out.contiguous().view(b, t_q, self.embedding_dim) # [B, T_q, embed_dim]

        output = self.out_projection(out)

        return output, att_weights




