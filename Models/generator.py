# from noise import NoiseGenerator
# from embeddings import TextEmbedder
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# class Generator(nn.Module):
#     def __init__(self, noise_shape=(16, 16, 3), target_shape=(224, 224, 3)):
#         super(Generator, self).__init__()
#         self.noise_shape = noise_shape
#         self.target_shape = target_shape
        
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(3, 128, kernel_size=4, stride=2, padding=1),  # Output: (32, 32, 128)
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 64, 64)
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (128, 128, 32)
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: (256, 256, 16)
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),  # Output: (224, 224, 3)
#             nn.Tanh()
#         )

#     def forward(self, noise):
#         noise = noise.unsqueeze(0)  # Add batch dimension (1, 16, 16, 3)
#         noise = noise.permute(0, 3, 1, 2)  # Change shape from (1, 16, 16, 3) to (1, 3, 16, 16)
#         output = self.model(noise)
#         return output

# if __name__ == "__main__":
#      embedder = TextEmbedder()
#      embedding = embedder.get_embeddings('Hello How are you Paritosh')
#      print(embedding.shape)
#      embedding_size = 768
#      target_shape=(16,16,3)
#      noise_generator = NoiseGenerator(embedding.shape[1],target_shape)
#      noise = noise_generator.generate_noise(embedding)
#      print(noise.shape)
    
#      generator = Generator(noise_shape=target_shape)
#      upsampled_image = generator(noise)
#      print("Generated image shape:", upsampled_image.shape)
#      image_np = upsampled_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#      image_np = (image_np + 1) / 2
#      #plt.imshow(image_np)
#      plt.imshow(image_np)
#      plt.axis('off')  # Hide the axis
#      plt.show()

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_shape=(16, 16, 3), target_shape=(224, 224, 3)):
        super(Generator, self).__init__()
        self.noise_shape = noise_shape
        self.target_shape = target_shape
        
        # Update channel numbers and dimensions correctly
        self.model = nn.Sequential(
            # Assuming the input noise has shape [batch_size, 3, 16, 16]
            nn.ConvTranspose2d(3, 128, kernel_size=4, stride=2, padding=1),  # Corrected to take 3 channels
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),  # Assumed 3 for RGB output
            nn.Tanh()
        )

    def forward(self, noise):
        # Ensure noise is in the correct shape [batch_size, channels, height, width]
        noise = noise.permute(0, 3, 1, 2)  # Change shape to [batch_size, 3, 16, 16] if needed
        output = self.model(noise)
        return output
