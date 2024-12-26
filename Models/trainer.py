# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from PIL import Image
# from torchvision import transforms
# from noise import NoiseGenerator
# from generator import Generator
# from discriminator import Discriminator
# from embeddings import TextEmbedder

# class GANTrainer:
#     def __init__(self, generator, discriminator, noise_generator, embedder, device, dataset):
#         self.generator = generator
#         self.discriminator = discriminator
#         self.noise_generator = noise_generator
#         self.embedder = embedder
#         self.device = device
#         self.dataset = dataset

#         self.criterion = nn.BCELoss()
#         self.optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#         self.optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#         self.transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def train(self, epochs, batch_size):
#         for epoch in range(epochs):
#             for idx in range(0, len(self.dataset), batch_size):
#                 batch_data = self.dataset[idx:idx+batch_size]
#                 self.optimizer_D.zero_grad()

#                 real_images = []
#                 embeddings = []

#                 for data in batch_data:
#                     image_path, caption = data['image'], data['caption']
#                     image = Image.open(image_path).convert('RGB')
#                     image = self.transform(image).to(self.device)
#                     real_images.append(image)
#                     embedding = self.embedder.get_embeddings(caption).to(self.device)
#                     embeddings.append(embedding)

#                 real_images = torch.stack(real_images)
#                 embeddings = torch.stack(embeddings)

#                 noise = self.noise_generator.generate_noise(embeddings).to(self.device)
#                 fake_images = self.generator(noise)

#                 valid = torch.ones((batch_size,), device=self.device, dtype=torch.float)
#                 fake = torch.zeros((batch_size,), device=self.device, dtype=torch.float)

#                 validity_real = self.discriminator(real_images)
#                 d_real_loss = self.criterion(validity_real, valid)

#                 validity_fake = self.discriminator(fake_images.detach())
#                 d_fake_loss = self.criterion(validity_fake, fake)

#                 d_loss = (d_real_loss + d_fake_loss) / 2
#                 d_loss.backward()
#                 self.optimizer_D.step()

#                 self.optimizer_G.zero_grad()

#                 validity_fake = self.discriminator(fake_images)
#                 g_loss = self.criterion(validity_fake, valid)

#                 g_loss.backward()
#                 self.optimizer_G.step()

#                 print(f"Epoch [{epoch+1}/{epochs}], Batch [{idx//batch_size+1}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_path = r'archive (1)\flickr30k_images\Images.csv'
#     df = pd.read_csv(data_path, dtype={'comment_number': str}, low_memory=False)

#     df.columns = df.columns.str.strip()

# # Print column names to verify
#     print("Columns:", df.columns)

    
#     dataset = [{'image': row['image_name'], 'caption': row['comment']} for index, row in df.iterrows()]

#     generator = Generator(noise_shape=(16, 16, 3)).to(device)
#     discriminator = Discriminator(image_shape=(3, 256, 256)).to(device)
#     noise_generator = NoiseGenerator(embedding_size=768, target_shape=(16,16,3)).to(device)
#     embedder = TextEmbedder()

#     trainer = GANTrainer(generator, discriminator, noise_generator, embedder, device, dataset)
#     trainer.train(epochs=100, batch_size=1)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torchvision import transforms
from noise import NoiseGenerator
from generator import Generator
from discriminator import Discriminator
from embeddings import TextEmbedder

class GANTrainer:
    def __init__(self, generator, discriminator, noise_generator, embedder, device, dataset):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_generator = noise_generator
        self.embedder = embedder
        self.device = device
        self.dataset = dataset

        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to the required dimensions
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def train(self, epochs, batch_size):
        for epoch in range(epochs):
            for idx in range(0, len(self.dataset), batch_size):
                batch_data = self.dataset[idx:idx+batch_size]
                self.optimizer_D.zero_grad()

                real_images = []
                embeddings = []

                for data in batch_data:
                    image_path, caption = data['image'], data['caption']
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image).to(self.device)
                    real_images.append(image)
                    embedding = self.embedder.get_embeddings(caption).to(self.device)
                    embeddings.append(embedding)

                real_images = torch.stack(real_images)
                embeddings = torch.cat(embeddings, dim=0)

                noise = self.noise_generator.generate_noise(embeddings)
                fake_images = self.generator(noise)

                valid = torch.ones((len(batch_data),), device=self.device, dtype=torch.float)
                fake = torch.zeros((len(batch_data),), device=self.device, dtype=torch.float)

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

                print(f"Epoch [{epoch+1}/{epochs}], Batch [{idx//batch_size+1}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = r'archive (1)\flickr30k_images\Images.csv'
    df = pd.read_csv(data_path, dtype={'comment_number': str}, low_memory=False)
    df.columns = df.columns.str.strip()

    print("Columns:", df.columns)
    dataset = [{'image': row['image_name'], 'caption': row['comment']} for index, row in df.iterrows()]

    generator = Generator(noise_shape=(16, 16, 3)).to(device)
    discriminator = Discriminator(image_shape=(3, 256, 256)).to(device)
    noise_generator = NoiseGenerator(embedding_size=768, target_shape=(16, 16, 3)).to(device)
    embedder = TextEmbedder()

    trainer = GANTrainer(generator, discriminator, noise_generator, embedder, device, dataset)
    trainer.train(epochs=100, batch_size=20)
