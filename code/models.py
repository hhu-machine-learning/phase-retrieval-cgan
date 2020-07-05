import numpy as np
import torch
import torch.nn as nn
import math

# Helper module to print layer output sizes
class Debug(nn.Module):
       
    def forward(self, x):
        print(x.shape)
        return x

# Helper module to flatten vectors in nn.Sequential
class Flatten(nn.Module):
    
    def flatten(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
    def forward(self, x):
        return self.flatten(x)


# Fully connected image network
class FCNet(nn.Module):

    __constants__ = ['mean', 'std']

    def __init__(self, imsize=(1, 28, 28), outsize=None, h=2048, mean=None, std=None):
        super(FCNet, self).__init__()
        print("Version 0.4")
        self.imsize = imsize
        if outsize is None:
            self.outsize = imsize
        else:
            self.outsize = outsize
        
        if mean is None:
            self.register_buffer('mean', torch.zeros(imsize))
        else:
            self.register_buffer('mean', torch.Tensor(mean))

        if std is None:
            self.register_buffer('std', torch.ones(imsize))
        else:
            self.register_buffer('std', torch.Tensor(std))
        
        self.layers = nn.Sequential(
            nn.Linear(imsize[0] * imsize[1] * imsize[2], h),
            nn.ReLU(),
            nn.Linear(h,h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h,h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h,h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, self.outsize[0] * self.outsize[1] * self.outsize[2]),
            nn.Sigmoid()
        )
        
    def forward(self, x):
    
        N = x.shape[0]

        x_norm = (x - self.mean) / self.std
    
        out = self.layers(x_norm.view(N, -1))
        return out.view(N, *self.outsize)


#CNN Image network with FC layers in between

class KernModule(nn.Module):
    
    def __init__(self, h):
        super(KernModule, self).__init__()
        self.h = h
        self.layers = nn.Sequential(
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
        )
        
    def forward(self, x):
        shp = x.shape
        return self.layers(x.view(shp[0], self.h)).view(*shp)

class ConvNet(nn.Module):

    __constants__ = ['mean', 'std']

    def __init__(self, imsize=(1, 28, 28), outsize=None, s=32, mean=None, std=None):
        super(ConvNet, self).__init__()
        print("Version 0.6")
        pow_pad = (2 ** (int(np.ceil(np.log2(imsize[-2])))) - imsize[-2],
                   2 ** (int(np.ceil(np.log2(imsize[-1])))) - imsize[-1])
        kern_size = 4 * ((imsize[1] + pow_pad[0]) // 16) * ((imsize[2] + pow_pad[1]) // 16) * s
        print("Additional padding to fit 2 exp:", pow_pad)
        print("Kern size:", kern_size)
        self.imsize = imsize
        if outsize is None:
            self.outsize = imsize
        else:
            self.outsize = outsize
        
        if mean is None:
            self.register_buffer('mean', torch.zeros(imsize))
        else:
            self.register_buffer('mean', torch.Tensor(mean))

        if std is None:
            self.register_buffer('std', torch.ones(imsize))
        else:
            self.register_buffer('std', torch.Tensor(std))
            
        self.layers = nn.Sequential(
            nn.Conv2d(imsize[0], imsize[0], kernel_size=1, padding=pow_pad), #32x32x1 = 1024
            nn.BatchNorm2d(imsize[0]),
            nn.ReLU(),
            nn.Conv2d(imsize[0], 1*s, kernel_size=5, padding=2), #32x32x32 = 32768
            nn.BatchNorm2d(1*s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #16x16x32 = 8192
            nn.Conv2d(1*s, 2*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8x8x64 = 4096
            nn.Conv2d(2*s, 4*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4x4x128 = 2048
            nn.Conv2d(4*s, 4*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2x2x128 = 512
            KernModule(h=kern_size),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(4*s, 4*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*s),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(4*s, 2*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*s),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(2*s, 1*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(1*s),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(1*s, self.outsize[0], kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        N = x.shape[0]
        x_norm = (x - self.mean) / self.std
        out = self.layers(x_norm)[..., :self.outsize[-2], :self.outsize[-1]]
        return out
        
        
# Encoder for real valued signals
class SignalEncoder(nn.Module):
    
    def __init__(self, shape=(1,28,28), latent_dim=20, hidden_dim=500, mean=None, std=None):
        super(SignalEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.in_dim = np.prod(np.array(shape))
        self.hidden_dim = hidden_dim
        
        if mean is None:
            self.register_buffer('mean', torch.zeros(shape))
        else:
            self.register_buffer('mean', torch.Tensor(mean))

        if std is None:
            self.register_buffer('std', torch.ones(shape))
        else:
            self.register_buffer('std', torch.Tensor(std))
        
        self.fc_hidden = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(self.hidden_dim, latent_dim)

    def forward(self, x):
        N = x.shape[0]

        x_norm = (x - self.mean) / self.std

        hidden_state = self.fc_hidden(x_norm.view(N, self.in_dim))
        mu , sigma = self.fc_mu(hidden_state), torch.exp(0.5 * self.fc_sigma(hidden_state))
        
        return mu, sigma


# Decoder for real valued signals        
class SignalDecoder(nn.Module):
    
    def __init__(self, shape=(1,28,28), latent_dim=20, hidden_dim=500):
        super(SignalDecoder, self).__init__()
        
        self.shape = shape
        self.out_dim = np.prod(np.array(shape))
        self.hidden_dim = hidden_dim
        
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        N = z.shape[0]
        return self.layers(z).view(N, *self.shape)


# VAE combining SignalEncoder and SignalDecoder that samples latent vector z from N(mu, sigma) in training
class SignalVAE(nn.Module):
    
    def __init__(self, shape=(1,28,28), latent_dim=392, hidden_dim=500, mean=None, std=None):
        super(SignalVAE, self).__init__()
        
        print("Version 4")
        
        self.encoder = SignalEncoder(shape=shape, latent_dim=latent_dim, hidden_dim=hidden_dim, mean=None, std=None)
        self.decoder = SignalDecoder(shape=shape, latent_dim=latent_dim, hidden_dim=hidden_dim)
        
    def sample(self, mu, sigma):
        noise = torch.randn_like(sigma)
        
        return mu + noise * sigma
        
    def forward(self, x):
        z_mu, z_sigma = self.encoder(x)
        
        if self.training:
            z = self.sample(z_mu, z_sigma)
        else:
            z = z_mu
        
        return self.decoder(z), z_mu, z_sigma
        

class SignalDCVAE(nn.Module):
    def __init__(self, z_size=500, ndf=64):
        super(VAE, self).__init__()
        self.ndf = ndf
        self.z_size = z_size

        self.enc = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 1),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2),
        nn.Conv2d(ndf, ndf*2, 4, 2, 1),
        nn.BatchNorm2d(ndf*2),
        nn.LeakyReLU(0.2),
        nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
        nn.BatchNorm2d(ndf*4),
        nn.LeakyReLU(0.2),
        nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
        nn.BatchNorm2d(ndf*8),
        nn.LeakyReLU(0.2),
        nn.Conv2d(ndf*8, ndf*8, 4, 2, 1),
        nn.BatchNorm2d(ndf*8),
        nn.LeakyReLU(0.2))

        self.dec = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReplicationPad2d(1),
        nn.Conv2d(ndf*8, ndf*8, 3, 1),
        nn.BatchNorm2d(ndf*8, 1.e-3),
        nn.LeakyReLU(0.2),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReplicationPad2d(1),
        nn.Conv2d(ndf*8, ndf*4, 3, 1),
        nn.BatchNorm2d(ndf*4, 1.e-3),
        nn.LeakyReLU(0.2),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReplicationPad2d(1),
        nn.Conv2d(ndf*4, ndf*2, 3, 1),
        nn.BatchNorm2d(ndf*2, 1.e-3),
        nn.LeakyReLU(0.2),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReplicationPad2d(1),
        nn.Conv2d(ndf*2, ndf, 3, 1),
        nn.BatchNorm2d(ndf, 1.e-3),
        nn.LeakyReLU(0.2),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReplicationPad2d(1),
        nn.Conv2d(ndf, 3, 3, 1),
        nn.Sigmoid()
        )
        
        self.mu_head = nn.Linear(8*4*ndf, z_size)
        self.logvar_head = nn.Linear(8*4*ndf, z_size)
        self.fc = nn.Linear(z_size, ndf*8*4)

    def encode(self, x):
        h1 = self.enc(x).view(-1, 8*self.ndf*4)
        return self.mu_head(h1), self.logvar_head(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h2 = F.relu(self.fc(z).view(-1, self.ndf*8, 2, 2))
        return self.dec(h2)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        BCE = nn.BCELoss(size_average=False)
        BCE = BCE(recon_x, x)
    
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        return BCE + KLD

        
class DCGenerator(nn.Module):
    def __init__(self, ngpu, ngf, nz, nc):
        super(DCGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, unit_interval=False):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            
        if unit_interval:
            return output / 2.0 + 0.5
        else:
            return output
        

# CNN Discriminator Network

class ConvDiscriminator(nn.Module):
    def __init__(self, imsize=(3, 64, 64), s=64):
        super(ConvDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(imsize[0], s, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (s) x 32 x 32
            nn.Conv2d(s, s * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(s * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (s*2) x 16 x 16
            nn.Conv2d(s * 2, s * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(s * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (s*4) x 8 x 8
            nn.Conv2d(s * 4, s * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(s * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (s*8) x 4 x 4
            nn.Conv2d(s * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)

    
# CNN Discriminator Network

class ConvDiscriminatorSmall(nn.Module):
    def __init__(self, imsize=(1, 28, 28), s=64):
        super(ConvDiscriminatorSmall, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(imsize[0], s, 4, stride=2, padding=1+2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (s) x 16 x 16
            nn.Conv2d(s, s * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(s * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (s*2) x 8 x 8
            nn.Conv2d(s * 2, s * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(s * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (s*4) x 4 x 4
            nn.Conv2d(s * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)
    
    
class FCDiscriminator(nn.Module):

    def __init__(self, imsize=(1, 28, 28), h=2048):
        super(FCDiscriminator, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(imsize[0] * imsize[1] * 2 * imsize[2], h),
            nn.BatchNorm1d(h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
    
        N = x.shape[0]
    
        out = self.layers(x.view(N, -1))
        return out
        
#############################
###        ARCHIVE        ###
#############################
