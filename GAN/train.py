import time
import torch
import torch.nn.functional as F
import torch.optim as optim

class Training:
    def __init__(self, dataloader, dis, gen, n_epochs=100, optimizer_dis=None, optimizer_gen=None, device=None):
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.optimizer_dis = optimizer_dis
        self.optimizer_gen = optimizer_gen
        self.dis = dis
        self.gen = gen
        self.device = device

        if self.optimizer_dis == None or self.optimizer_gen == None:
            self.optimizer_dis = optim.RMSprop(dis.parameters(), lr=0.00002)
            self.optimizer_gen = optim.Adagrad(gen.parameters(), lr=0.02)
        
        if self.device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              
    def train(self):
        print(f"Starting Training for {self.n_epochs} Epochs")
        for epoch in range(self.n_epochs):
                start = time.time()

                # Initialize losses
                d_loss = 0.0
                g_loss = 0.0
                num_batch = 0

                for _, (real_image, _) in enumerate(self.dataloader):
                    real_image = real_image.to(self.device)
                    batch_size = real_image.size(0)
                    # Add noise to the inputs
                    noise_real = torch.randn_like(real_image) * 0.1
                    noise_fake = torch.randn_like(real_image) * 0.1

                    # Train Discriminator
                    self.optimizer_dis.zero_grad()
                    real_image_noisy = real_image + noise_real
                    # Real Data
                    real_pred = self.dis(real_image_noisy)
                    # real_label = torch.ones_like(real_pred, device=device, dtype=torch.float32)
                    real_label = torch.full_like(real_pred, 0.9, device=self.device, dtype=torch.float32)  # Smoothed label
                    real_loss =  F.binary_cross_entropy(real_pred, real_label, reduction='mean')

                    # Fake Data
                    noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
                    gen_out = self.gen(noise)
                    fake_image_noisy = gen_out + noise_fake
                    fake_pred = self.dis(fake_image_noisy)
                    # fake_label = torch.zeros_like(fake_pred, device=device, dtype=torch.float32)
                    fake_label = torch.full_like(fake_pred, 0.1, device=self.device, dtype=torch.float32)  # Smoothed label
                    fake_loss =  F.binary_cross_entropy(fake_pred, fake_label, reduction='mean')

                    dis_loss = (real_loss+fake_loss)/2
                    dis_loss.backward()
                    self.optimizer_dis.step()

                    # Train Generator
                    self.optimizer_gen.zero_grad()

                    noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
                    gen_out = self.gen(noise)
                    dis_out = self.dis(gen_out)
                    # label = real_label = torch.ones_like(dis_out, device=device, dtype=torch.float32)
                    gen_loss =  F.binary_cross_entropy(dis_out, real_label, reduction='mean')

                    gen_loss.backward()
                    self.optimizer_gen.step()

                    # Accumulate losses
                    d_loss += dis_loss.item()
                    g_loss += gen_loss.item()
                    num_batch += 1


                # Print losses
                end = time.time()
                # if i % 10 == 0:
                avg_d_loss = d_loss/num_batch
                avg_g_loss = g_loss/num_batch
                print(f"Epoch [{epoch + 1}] Loss D: {avg_d_loss:.4f} Loss G: {avg_g_loss:.4f} Time: {end-start:.2f} sec")

        return self.gen