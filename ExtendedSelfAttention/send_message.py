
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

