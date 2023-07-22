import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

class DualVAE(BaseVAE):
    def __init__(self,
                 latent_dim: int,
                 hidden_dims = None,
                 **kwargs) -> None:
        super(DualVAE, self).__init__()
        self.latent_dim = latent_dim
        modules_rgb = []
        modules_depth = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        # Build Encoder
        in_channels_rgb = 3
        in_channels_depth = 1
        for h_dim in hidden_dims:
            modules_rgb.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_rgb, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            modules_depth.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_depth, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels_rgb = h_dim
            in_channels_depth = h_dim
        modules_rgb.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()))
        modules_depth.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()))
        self.encoder_rgb = nn.Sequential(*modules_rgb)
        self.encoder_depth = nn.Sequential(*modules_depth)
        self.fc_mu = nn.Linear(hidden_dims[-1]*2, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*2, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*12)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 4,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())


    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        rgb = input[:, :3]
        depth = input[:, 3]
        depth.unsqueeze_(1)
        result_rgb = self.encoder_rgb(rgb)
        result_depth = self.encoder_depth(depth)
        self.result_shape = result_rgb.shape[-1]
        result = torch.cat([result_rgb, result_depth], dim=-1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.Tensor) [B x D]
        :return: (torch.Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view((-1, 256, 3, 4))
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, rgb, depth) -> np.array:
        rgb = Image.fromarray(rgb)
        rgb = ToTensor()(rgb)
        depth = Image.fromarray(depth)
        depth = ToTensor()(depth)
        input = torch.cat((rgb, depth), dim=0)
        input.unsqueeze_(0)
        input = input.to(device="cuda:0")
        with torch.no_grad():
            mu, log_var = self.encode(input)
            z = self.reparameterize(mu, log_var)
            z.squeeze_(0)
        # return  [self.decode(z), input, mu, log_var]
        return  z.cpu().numpy()

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)
        # recons_loss =F.binary_cross_entropy(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]