"""NICE model
"""

import torch
import torch.nn as nn


m =torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))


class Model(nn.Module):
    def __init__(self, latent_dim,device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28 maybe mistake
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim) # B, latent_dim
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim) # B, latent_dim

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )


    def sample(self,sample_size,mu=None,logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu==None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)

        samples_normal = torch.reshape(m.sample(torch.tensor([sample_size, self.latent_dim])),
                                       [sample_size, self.latent_dim]).to(self.device)  # sample_size, latent_dim
        out = self.upsample(samples_normal)
        out = torch.reshape(out, [sample_size, 64, 7, 7])
        sample = self.decoder(out) # sample_size, 1, 28, 28
        return sample


        #TODO


    def z_sample(self, mu, logvar):
        B, latent_dim = mu.size()
        samples_normal = torch.reshape(m.sample(torch.tensor([B, latent_dim]))
                                       , [B, latent_dim]).to(self.device) # B, latent_dim
        z = mu + torch.sqrt(torch.exp(logvar))*samples_normal #maybe exp on logvar
        return z
        #TODO
        pass

    def loss(self, x, recon, mu, logvar):
        KL = torch.sum(0.5*(1+ 2*logvar-torch.square(mu)-torch.exp(logvar)), dim=1)
        Bernoulli = torch.sum(x*torch.log(recon) + (1 - x)*torch.log(1 - recon), dim=1)
        return KL + Bernoulli
        #TODO
        pass

    def forward(self, x):
        B, _, W, H = x.size()
        out = self.encoder(x)
        out = torch.reshape(out, [B, -1])
        mu = self.mu(out)
        logvar = self.logvar(out)
        samples_posterior = self.z_sample(mu=mu, logvar=logvar)
        out = self.upsample(samples_posterior)
        out = torch.reshape(out, [B, 64, 7, 7])
        decode = self.decoder(out)
        recon = torch.reshape(decode,[B, 28*28])
        x = torch.reshape(x,[B, 28*28])
        Elbo = self.loss(x,recon,mu,logvar)
        return Elbo

        #TODO
        pass