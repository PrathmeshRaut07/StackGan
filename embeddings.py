from transformers import BertModel, BertTokenizer
import torch
from sklearn.decomposition import PCA
import numpy as np

class TextEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        sentence_embedding = torch.mean(embeddings, dim=1)
        return sentence_embedding.detach()

if __name__ == "__main__":
    # Initialize components
    text = "This is an example sentence to generate embeddings."
    noise_shape = (60, 60)

    # Create instances of classes
    embedder = TextEmbedder()
    
    # Generate text embeddings
    text_embeddings = embedder.get_embeddings(text)
    
    # Ensure tensor is contiguous and reshape
    text_embeddings = text_embeddings.view(1, 768)  # Ensure it's the correct shape (1, 768)
    image_rgb = text_embeddings.view(16, 16, 3)     # Now reshape to (16, 16, 3)
    
    print(image_rgb.shape)

    # Further usage in GAN
    # gan_model.train(gan_input)
