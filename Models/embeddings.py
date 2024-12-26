# from transformers import BertModel, BertTokenizer
# import torch
# import numpy as np

# class TextEmbedder:
#     def __init__(self, model_name='bert-base-uncased'):
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)
        
#     def get_embeddings(self, text):
#         inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         outputs = self.model(**inputs)
#         embeddings = outputs.last_hidden_state
#         sentence_embedding = torch.mean(embeddings, dim=1)
#         return sentence_embedding.detach()
from transformers import BertModel, BertTokenizer
import torch

class TextEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode
        
    def get_embeddings(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        # Get the model outputs
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model(**inputs)
        
        # Get the embeddings and calculate the sentence embedding
        embeddings = outputs.last_hidden_state
        sentence_embedding = torch.mean(embeddings, dim=1)
        return sentence_embedding.detach() 




    



