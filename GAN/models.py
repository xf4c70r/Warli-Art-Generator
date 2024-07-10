import torch.nn as nn

# Residual Block

class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
        nn.Conv2d(in_feat, out_feat, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_feat),
        nn.ReLU(inplace=True), 
        nn.Conv2d(out_feat, out_feat, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_feat),
        nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return x + self.block(x)

epsilon=0.00005

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input is the latent vector Z.
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024, eps=epsilon),
            nn.ReLU(True),
            nn.Dropout(0.4),

#             # Additional Conv layer
#             nn.ConvTranspose2d(1024, 1024, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(1024, eps=epsilon),
#             nn.ReLU(True),
#             nn.Dropout(0.4),
            
            ResBlock(1024,1024),

            # State size. (1024x4x4)
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512, eps=epsilon),
            nn.ReLU(True),
            nn.Dropout(0.4),

#             # Additional Conv layer
#             nn.ConvTranspose2d(512, 512, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(512, eps=epsilon),
#             nn.ReLU(True),
#             nn.Dropout(0.4),
            
            ResBlock(512,512),

            # State size. (512x8x8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, eps=epsilon),
            nn.ReLU(True),
            nn.Dropout(0.4),

            # State size. (256x16x16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, eps=epsilon),
            nn.ReLU(True),
            nn.Dropout(0.4),
            
            ResBlock(128,128),

            # State size. (128x32x32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, eps=epsilon),
            nn.ReLU(True),
            nn.Dropout(0.4),

            # State size. (64x64x64)
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()  
            # Final state size. (3x128x128)
        )

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)  # Reshape input noise vector into a batch of inputs for ConvTranspose2d
        output = self.model(x)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(

            # Input size. (3x128x128)
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            # State size. (64x64x64)
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
#             ResBlock(128,128),

            # State size. (128x32x32)
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
#             ResBlock(256,256),

            # State size. (256x16x16)
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
#             ResBlock(512,512),

            # State size. (512x8x8)
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
            # State size. (1024x4x4)
            nn.Conv2d(1024, 1, 4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Sigmoid()  # Output a single scalar per image indicating real or fake
        )

    def forward(self, x):
        output = self.model(x)
        return output.view(-1, 1)  # Flatten to [batch_size, 1]