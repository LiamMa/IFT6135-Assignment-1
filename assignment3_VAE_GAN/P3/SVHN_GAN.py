"""
The architecture is adapted from DCGAN
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inp):
        return inp.view((inp.shape[0], -1))


class Reshape(nn.Module):

    def __init__(self, tgt_shape):
        super(Reshape, self).__init__()
        self.tgt_shape = tgt_shape

    def forward(self, inp):
        return inp.view([inp.shape[0], *self.tgt_shape])


class GAN(nn.Module):

    def __init__(self, n_latent):
        super(GAN, self).__init__()

        self.n_latent = n_latent
        self.nc = 32

        # generator input should be: B x n_latent
        self.generator = nn.Sequential(
            # n_latent
            Reshape((self.n_latent, 1, 1)),
            nn.ConvTranspose2d(self.n_latent, self.nc * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.nc * 4),
            nn.ELU(),
            # 128 x 4 x 4
            nn.ConvTranspose2d(self.nc*4, self.nc*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc * 2),
            nn.ELU(),
            # 64 x 8 x 8
            nn.ConvTranspose2d(self.nc*2, self.nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc),
            nn.ELU(),
            # 32 x 16 x 16
            nn.ConvTranspose2d(self.nc, 3, 4, 2, 1, bias=False),
            # 3 x 32 x 32
            nn.Tanh()
        )
        # discriminator input should be: B x 3 x 32 x 32
        self.discriminator = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, self.nc, 4, 2, 1),
            nn.ELU(),
            # 32 x 16 x16
            nn.Conv2d(self.nc, self.nc*2, 4, 2, 1),
            nn.ELU(),
            # 64 x 8 x 8
            nn.Conv2d(self.nc*2, self.nc*4, 4, 2, 1),
            nn.ELU(),
            # 128 x 4 x 4
            nn.Conv2d(self.nc*4, 1, 4, 1, 0),
            Flatten(),
            # nn.Sigmoid()
            # 1
        )

        for nm, w in self.generator.named_parameters():
            if 'bias' not in nm:
                nn.init.normal_(w, 0, 0.02)
        for nm, w in self.discriminator.named_parameters():
            if 'bias' not in nm:
                nn.init.normal_(w, 0, 0.02)

    def forward(self, rand_x):
        pass

    def generate(self, rand_x):
        return self.generator(rand_x)

    def discriminate(self, x):
        return self.discriminator(x)

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.shape[0]

        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        x_hat = eps * real_data + (1-eps) * fake_data
        x_hat.requires_grad = True

        res_D = self.discriminator(x_hat)
        grad_x_hat = torch.autograd.grad(res_D, x_hat, grad_outputs=torch.ones(res_D.shape, device=real_data.device),
                                         create_graph=True, retain_graph=True, only_inputs=True)[0].view(res_D.shape[0], -1)

        grad_loss = torch.pow((grad_x_hat.norm(2, dim=1) - 1), 2).mean()

        return grad_loss


if __name__ == '__main__':
    GAN(100)