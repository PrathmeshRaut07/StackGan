import torch
import torch.nn as nn
import torch.optim as optim
from noise import NoiseGenerator
from generator import Generator
from discriminator import Discriminator
from embeddings import TextEmbedder

class GANTrainer:
    def __init__(self, generator, discriminator, noise_generator, embedder, device):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_generator = noise_generator
        self.embedder = embedder
        self.device = device

        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, epochs, batch_size):

        for epoch in range(epochs):
            for _ in range(batch_size):
                self.optimizer_D.zero_grad()

            # Get embeddings and ensure they are on the correct device
                embedding = self.embedder.get_embeddings('Hello How are you Paritosh').to(self.device)

            # Ensure noise generation also respects device allocation
                noise = self.noise_generator.generate_noise(embedding).to(self.device)

                fake_images = self.generator(noise)

            # Ensure real images tensor is on the correct device
                real_images = torch.randn((1, 3, 256, 256), device=self.device)

                valid = torch.ones((1,), device=self.device, dtype=torch.float)
                fake = torch.zeros((1,), device=self.device, dtype=torch.float)

                validity_real = self.discriminator(real_images)
                d_real_loss = self.criterion(validity_real, valid)

                validity_fake = self.discriminator(fake_images.detach())
                d_fake_loss = self.criterion(validity_fake, fake)

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                validity_fake = self.discriminator(fake_images)
                g_loss = self.criterion(validity_fake, valid)

                g_loss.backward()
                self.optimizer_G.step()
        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(noise_shape=(16, 16, 3)).to(device)
    discriminator = Discriminator(image_shape=(3, 256, 256)).to(device)
    noise_generator = NoiseGenerator(embedding_size=768,target_shape=(16,16,3)).to(device)
    embedder = TextEmbedder()

    trainer = GANTrainer(generator, discriminator, noise_generator, embedder, device)
    trainer.train(epochs=5, batch_size=1)
