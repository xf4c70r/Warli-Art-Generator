from torchvision.utils import make_grid
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

class utils:
    def __init__(self, data_dir, device=None):
        self.data_dir = data_dir
        self.device = device

    def print_examples(self):
        for category in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, category)):
                category_dir = os.path.join(self.data_dir, category)
                image_files = [f for f in os.listdir(category_dir) if f.endswith(('.jpg'))]
                num_images = len(image_files)

                print(f"Category: {category}")
                print(f"Number of images: {num_images}")

                if num_images > 0:
                    sample_image_path = os.path.join(category_dir, image_files[0])
                    sample_image = Image.open(sample_image_path)
                    image_size = sample_image.size
                    # num_channels = len(sample_image.mode)
                    num_channels = len(sample_image.getbands())

                    print(f"Image size: {image_size}")
                    print(f"Number of channels: {num_channels}")

                print()

    def generate_art(self, gen):
        # Set the model to evaluation mode
        self.gen.eval()

        num_images = 16  # Number of images to generate
        latent_dim = 100  # Dimension of the latent vector
        noise = torch.randn(num_images, latent_dim, 1, 1)  # Generate random noise

        # Generate images from the noise
        # Ensure that noise is on the same device as the model
        noise = noise.to(next(gen.parameters()).self.device)  # Move noise to the device of the model
        fake_images = gen(noise)

        # Convert images to a suitable format for displaying
        fake_images = (fake_images + 1) / 2  # Adjust from [-1, 1] to [0, 1]
        grid = make_grid(fake_images, nrow=4)  # Create a grid of images

        # Plot the images
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())  # Convert to numpy and plot
        plt.axis('off')
        plt.show()