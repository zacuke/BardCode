from bardapi import Bard
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer

load_dotenv()

class ExtendedSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, heads=8, dropout=0.1):
        super(ExtendedSelfAttention, self).__init__()

        self.d_model = embed_dim
        self.heads = heads
        self.dropout = dropout

        self.q_linear = nn.Linear(embed_dim, heads * embed_dim)
        self.k_linear = nn.Linear(embed_dim, heads * embed_dim)
        self.v_linear = nn.Linear(embed_dim, heads * embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, bias=False
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

#create a .env file with this in it
#token=cookie: __Secure-1PSID
token = os.getenv("token")
 
# Create a new Bard client.
bard_client = Bard(token=token)

initPrompt = "Your response to this will be sent back to you exactly as you sent it to me."

print ("---- " + initPrompt +" ----")

# Use the Bard client to call the Bard API to generate text.
initResponse = bard_client.get_answer(initPrompt)
initResponseContent = initResponse["content"]
print (initResponseContent)


print ("--------------------------")
queryResponse = bard_client.get_answer(initResponseContent)
queryResponseContent = queryResponse["content"]
print (queryResponseContent)

# Convert the text to a list of integers.
tokens = AutoTokenizer.from_pretrained("bert-base-uncased")(queryResponseContent) # n/a: google/bard-base 

# Convert the list of integers to a Tensor. 
tensor = torch.tensor(tokens["input_ids"])

# Pass the tensor to the ExtendedSelfAttention model.
model = ExtendedSelfAttention()
output, attention = model(tensor, tensor, tensor)

# Print the output.
print(output)



