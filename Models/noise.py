# from transformers import BertModel, BertTokenizer
# import torch
# import torch.nn as nn
# import numpy as np
# from embeddings import TextEmbedder
# class NoiseGenerator(nn.Module):
#     def __init__(self, embedding_size, target_shape):
#         super(NoiseGenerator, self).__init__()
#         self.embedding_size = embedding_size
#         self.target_shape = target_shape
#         self.fc = nn.Linear(embedding_size, torch.prod(torch.tensor(target_shape)))
#         self.activation = nn.Tanh()  
#     def generate_noise(self, text_embedding):
#         text_embedding = text_embedding.view(1, self.embedding_size)
#         noise = self.fc(text_embedding)
#         noise = self.activation(noise)  
#         noise = noise.view(*self.target_shape)
#         return noise

# embedding_size = 768
# target_shape = (16, 16, 3)
# model = NoiseGenerator(embedding_size, target_shape)

# embedder=TextEmbedder()
# text_embedding=embedder.get_embeddings('Hello how are you')
# noise = model.generate_noise(text_embedding)
# print(noise)
import torch
import torch.nn as nn

class NoiseGenerator(nn.Module):
    def __init__(self, embedding_size, target_shape):
        super(NoiseGenerator, self).__init__()
        self.embedding_size = embedding_size
        self.target_shape = target_shape
        self.fc = nn.Linear(embedding_size, torch.prod(torch.tensor(target_shape)))
        self.activation = nn.Tanh()

    def generate_noise(self, text_embedding):
        # Use the actual batch size dynamically
        batch_size = text_embedding.size(0)  # Get the actual batch size from the input
        noise = self.fc(text_embedding)
        noise = self.activation(noise)
        noise = noise.view(batch_size, *self.target_shape)  # Reshape using the actual batch size
        return noise

    
