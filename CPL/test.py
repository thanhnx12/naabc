# with open('/media/data/thanhnb/Bi/naabc/CPL/data/CFRLFewRel/relation_description.txt', 'r',encoding = 'utf-8') as f:
#     for line in f:
#         line = line.split('\t')
#         print(line[1])
#         print(line[2])


from transformers import BertTokenizer, BertModel
import pickle
import torch
import numpy as np

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = BertModel.from_pretrained('google-bert/bert-base-uncased')

# Specify the path to your .pkl file
file_path = '/media/data/thanhnb/Bi/naabc/CPL/data/CFRLTacred/tacred-rich_context_description_d1.pkl'

# Open the file in read-binary mode and load the contents
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Initialize the dictionary to save embeddings
dict_save_to_pkl = {}

# Loop through the data and process each item
with open('/media/data/thanhnb/Bi/naabc/CPL/data/CFRLTacred/relation_description.txt', 'r',encoding = 'utf-8') as f:
    for line in f:
        line = line.split('\t')
        key = line[1].strip()
        value = line[1].strip()
        
        encoding = tokenizer.batch_encode_plus(
            [value],                   # List of input texts
            padding=True,          # Pad to the maximum sequence length
            truncation=True,       # Truncate to the maximum sequence length if necessary
            return_tensors='pt',   # Return PyTorch tensors
            add_special_tokens=True # Add special tokens CLS and SEP
        )

        input_ids = encoding['input_ids']          # Token IDs
        attention_mask = encoding['attention_mask'] # Attention mask

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state  # This contains the embeddings
            mean_embedding = word_embeddings.mean(dim=1) # Mean pooling of embeddings
            mean_embedding_np = mean_embedding.numpy()   # Convert tensor to NumPy array
            dict_save_to_pkl[key.strip()] = mean_embedding_np

# Specify the path to save the dictionary as a .pkl file
save_path = '/media/data/thanhnb/Bi/naabc/CPL/data/CFRLTacred/tacred_embeddings_only_label.pkl'

# Save the dictionary to a .pkl file
with open(save_path, 'wb') as save_file:
    pickle.dump(dict_save_to_pkl, save_file)

print(f"Embeddings saved to {save_path}")

