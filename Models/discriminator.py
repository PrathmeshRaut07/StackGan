import torch
import torch.nn as nn
from noise import NoiseGenerator
from generator import Generator
from embeddings import TextEmbedder
class Discriminator(nn.Module):
    def __init__(self, image_shape=(3, 256, 256)):
        super(Discriminator, self).__init__()
        self.image_shape = image_shape
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 128, 128)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # Output: (1024, 8, 8)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),  # Output: (2048, 4, 4)
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2048, 1, kernel_size=4, stride=1, padding=0),  # Output: (1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, image):
        validity = self.model(image)
        return validity.view(-1, 1).squeeze(1)

# if __name__ == "__main__":
#      image_shape = (3, 256, 256)
#      embedder = TextEmbedder()
#      embedding = embedder.get_embeddings('Hello How are you Paritosh')
#      print(embedding.shape)
#      target_shape=(16,16,3)
#      noise_generator = NoiseGenerator(embedding.shape[1],target_shape)
#      noise = noise_generator.generate_noise(embedding)
#      print(noise.shape)
    
#      generator = Generator(noise_shape=(16, 16, 3))
#      upsampled_image = generator(noise)

#      discriminator = Discriminator(image_shape=upsampled_image.shape)
#      validity = discriminator(upsampled_image)
#      print("Validity score:", validity)
