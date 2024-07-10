from torchvision.datasets import ImageFolder
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class DataLoader:
    def __init__(self, batch_size=64):
        self.transform = transforms.Compose([
            torchvision.transforms.CenterCrop((600, 600)),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        self.batch_size = batch_size
    
    def dataloader(self, directory):
        dataset = ImageFolder(directory, transform=self.transform)
        # total_samples = len(dataset)
        # train_samples = int(0.5 * total_samples)  # 50% of the total samples
        # val_samples = total_samples - train_samples

        # Split the dataset into training and validation sets
        # train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return dataloader