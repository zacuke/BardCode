import torch

class ExtendedSelfAttention2(torch.nn.Module):
    def __init__(self, d_model, heads=8, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout

        # Create the linear layers for the query, key, and value vectors.
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)

        # Create the scaled dot-product attention mechanism.
        self.attention = torch.nn.MultiheadAttention(d_model, heads, dropout)

        # Create the residual connection.
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        # Compute the query, key, and value vectors.
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Compute the attention weights.
        attn = self.attention(q, k, v)

        # Mask the attention weights.
        # This is necessary for masked self-attention, where you cannot attend to future positions.
        mask = torch.tril(torch.ones(x.size(0), x.size(0)), diagonal=-1).to(x.device)
        attn = attn.masked_fill(~mask, -float('inf'))

        # Compute the attention output.
        x = attn @ v

        # Apply a residual connection.
        x = self.residual(x) + x

        return x
    

class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, heads=8, dropout=0.1):
        super().__init__()

        self.self_attention = ExtendedSelfAttention2(d_model, heads, dropout)

        # Other layers in the encoder, such as a feed-forward layer.

    def forward(self, x):
        x = self.self_attention(x)

        # Other operations in the encoder.

        return x
