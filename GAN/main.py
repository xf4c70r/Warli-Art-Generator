import torch
from dataloader import DataLoader
from models import Generator, Discriminator
from train import Training
import utils

def main(device):
    dir = 'Warli Art Dataset/Warli Art Dataset'
    gen = Generator()
    disc = Discriminator()
    utility = utils(dir, device)
    dataset = DataLoader.dataloader(dir)
    utility.print_examples()
    training = Training(dataset, disc, gen, device)
    generartor = training.train()
    utility.generate_art(generartor)

if __name__ == '__main__':
    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    main(device)