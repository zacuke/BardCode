import torch

class ExtendedSelfAttention(torch.nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(ExtendedSelfAttention, self).__init__()

        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout

        self.q_linear = torch.nn.Linear(d_model, heads * d_model)
        self.k_linear = torch.nn.Linear(d_model, heads * d_model)
        self.v_linear = torch.nn.Linear(d_model, heads * d_model)

        self.attention = torch.nn.MultiheadAttention(
            heads, d_model, dropout=dropout
        )

    def forward(self, query, key, value):
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = q.view(-1, self.heads, self.d_model)
        k = k.view(-1, self.heads, self.d_model)
        v = v.view(-1, self.heads, self.d_model)

        output, attention = self.attention(q, k, v)

        output = output.view(-1, self.d_model)

        return output, attention

# Example usage:

model = ExtendedSelfAttention(512, 8)

query = torch.randn(10, 512)
key = torch.randn(10, 512)
value = torch.randn(10, 512)

output, attention = model(query, key, value)

print(output.shape)  # torch.Size([10, 512])