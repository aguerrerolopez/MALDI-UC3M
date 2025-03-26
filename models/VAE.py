import torch
import torch.nn as nn
from pytorch_model_summary import summary

from utils.probability_distributions import log_normal_diag, log_standard_normal, log_bernoulli, log_categorical
from utils.losses import RE_log_prob, KL_divergence, calculate_ELBO

class Encoder(nn.Module):
    """Encoder network for the Variational Autoencoder (VAE).
    This class defines the encoder part of a VAE, which is responsible for encoding
    the input data into a la tent space representation. It includes methods for 
    reparameterization, encoding, sampling, and calculating the log probability.
    Attributes:
        encoder (nn.Module): The neural network used as the encoder.
    Methods:
        reparameterization(mu, log_var):
            Applies the reparameterization trick to sample from a Gaussian distribution.
        encode(x):
            Encodes the input data into the parameters of a Gaussian distribution (mean and log-variance).
        sample(x=None, mu_e=None, log_var_e=None):
            Samples from the Gaussian distribution using the reparameterization trick.
        log_prob(x=None, mu_e=None, log_var_e=None, z=None):
            Calculates the log probability of the input data, used for calculating the Evidence Lower Bound (ELBO).
        forward(x, type='log_prob'):
            Defines the forward pass of the encoder, which can either return the log probability or a sample.
    """

    def __init__(self, encoder_net):
        super(Encoder, self).__init__()

        # The init of the encoder network
        self.encoder = encoder_net

    @staticmethod
    def reparameterization(mu, log_var):
        """
        Reparametrization trick for Gaussians. The formula is the following:
        z = mu + std * eps, where eps ~ N(0, 1)
        """
        # First , we need to get std from log−variance .
        std = torch.exp(0.5*log_var)
        # Then we get the noise from the standard normal distribution (mean=0, std=1)
        eps = torch.randn_like(std)

        return mu + std * eps

    def encode(self, x):
        """This function implements the output of the encoder network (i.e., parameters of a Gaussian)."""
        # First, we calculate the output of the encoder network of size 2*latent_dim
        h_e = self.encoder(x)
        # Then we split the output into two parts: mu and log_var
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        """This is the sampling procedure from the Gaussian distribution."""
        #If we don ’t provide a mean and a log−variance , we must first calculate it:
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        else:
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log-var can`t be None!')
            
        # Apply the reparameterization trick
        z = self.reparameterization(mu_e, log_var_e)
        return z

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        """This function calculates the log probability of the input x, which is later used for caluclating the ELBO."""
        # If x is provided, we need to calculate a corresponding sample (get mu, log-var and z)
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        # Otherwise , we should provide mu , log−var and z!
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError('mu, log-var and z can`t be None!')

        return log_normal_diag(z, mu_e, log_var_e)

    def forward(self, x, type='log_prob'):
        """Forward pass: it is either log-probability (by default) or sampling."""
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)
        
class Decoder(nn.Module):
    """Decoder network for the Variational Autoencoder (VAE).
    This class defines the decoder part of a VAE, which is responsible for decoding
    the latent space representation into the output data. It includes methods for
    decoding, sampling, and calculating the log probability.
    Attributes:
        decoder (nn.Module): The neural network used as the decoder.
        distribution (str): The distribution used for the decoder (categorical, Bernoulli, or gaussian).
        num_vals (int): The number of values for the categorical distribution.
    Methods:
        decode(z):
            Calculates the parameters of the likelihood function p(x|z).
        sample(z):
            Samples from the decoder (likelihood function p(x|z)).
        log_prob(x, z):
            Calculates the conditional log-likelihood function p(x|z).
        forward(z, x=None, type='log_prob'):
            Defines the forward pass of the decoder, which can either return the log probability or a sample.
    """

    def __init__(self, decoder_net, distribution='categorical'):
        super(Decoder, self).__init__()

        # The decoder network
        self.decoder = decoder_net
        # The distribution used for the decoder (categotical by default)
        self.distribution = distribution

    def decode(self, z):
        """This function calculates parameters of the likelihood function p(x|z)"""
        # First, we apply the decoder network
        h_d = self.decoder(z)

        # In the case of Bernoulli distribution
        if self.distribution == 'bernoulli':
            # In the Bernoulli case , we have x_d \in {0 ,1}. 
            # Therefore, it is enough to output a single probability,
            # because p(x_d =1|z) = \theta and p(x_d =0|z) = 1− \theta
            mu_d = torch.sigmoid(h_d)
            return [mu_d]
        
        # In the case of Gaussian distribution
        elif self.distribution == 'gaussian':
            # The decoder outputs just the mean, std is fixed
            mu_d = h_d
            log_var_d = torch.log(torch.tensor(0.1))  # Fixed log variance (log(0.1) or log(0.2), etc.)
            return [mu_d, log_var_d]
        
        else:
            raise ValueError('Either `bernoulli` or `gaussian`')

    def sample(self, z):
        """This function samples from the decoder (likelihood function p(x|z))."""
        outs = self.decode(z)

        if self.distribution == 'bernoulli':
            # In the case of Benoulli, we don't need reshaping
            mu_d = outs[0]
            # and we can use the built-in PyTorch function for Bernoulli sampling
            x_new = torch.bernoulli(mu_d)

        elif self.distribution == 'gaussian':
            mu_d = outs[0]
            log_var_d = outs[1]
            # We sample from the Gaussian distribution
            std_d = torch.exp(0.5 * log_var_d)  # Compute standard deviation from log variance
            eps = torch.randn_like(mu_d)  # Sample from standard normal distribution
            x_new = mu_d + eps * std_d  # Reparameterization trick
            
        else:
            raise ValueError('Either `bernoulli` or `gaussian`')

        return x_new
    
    def log_prob(self, x, z):
        """This function calculates the conditional log−likelihood function p(x|z)"""
        outs = self.decode(z)

        if self.distribution == 'bernoulli':
            mu_d = outs[0]
            log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        elif self.distribution == 'gaussian':
            mu_d = outs[0]
            log_var_d = outs[1]  # log_var_d is the second output from the decoder
            log_p = log_normal_diag(x, mu_d, log_var_d, reduction='sum')
            
        else:
            raise ValueError('Only `bernoulli` and `gaussian` distributions are supported')

        return log_p

    def forward(self, z, x=None, type='log_prob'):
        """Forward pass: it is either log-probability (by default) or sampling."""
        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(z)
        
class Prior(nn.Module):
    """
    The current implementation of the prior is very simple, namely, it is a standard Gaussian.

    We could have used a built-in PyTorch distribution. However, we didn't do that for two reasons:
    (i) It is important to think of the prior as a crucial component in VAEs.
    (ii) We can implement a learnable prior (e.g., a flow-based prior, VampPrior, a mixture of distributions).

    Args:
        L (int): Dimensionality of the latent space.
    Methods:
        sample(batch_size): Samples from the prior distribution.
                batch_size (int): Number of samples to generate.
            Returns: torch.Tensor: Samples from the prior distribution.
        log_prob(z): Computes the log probability of the given samples under the prior distribution.
                z (torch.Tensor): Samples for which to compute the log probability.
            Returns: torch.Tensor: Log probability of the samples.
    """

    def __init__(self, L):
        super(Prior, self).__init__()
        self.L = L # Dimensionality of the latent space

    def sample(self, batch_size):
        """Samples from the prior distribution."""
        z = torch.randn((batch_size, self.L))
        return z

    def log_prob(self, z):
        """Computes the log probability of the given samples under the prior distribution."""
        return log_standard_normal(z)
    
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.
    
    Args:
        likelihood_type (str): Likelihood function used for the decoder (gaussian or Bernoulli).
        D (int): Input dimension.
        L (int): Latent dimension.
    Methods:
        forward(x, reduction='avg'): Forward pass of the VAE.
                x (torch.Tensor): Input data.
                reduction (str): Reduction type ('avg' or 'sum').
            Returns: torch.Tensor: ELBO, RE, KL.
        sample(batch_size): Samples from the VAE.
                batch_size (int): Number of samples to generate.
            Returns: torch.Tensor: Samples from the VAE.
    """

    def __init__(self, likelihood_type='bernoulli', D=256, L=32):
        super(VAE, self).__init__()

        encoder_net = nn.Sequential(nn.Linear(D, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(),
                                    nn.Linear(64, 2 * L))  # outputs mu and log_var
        
        decoder_net = nn.Sequential(nn.Linear(L, 64), nn.ReLU(),
                                    nn.Linear(64, 128), nn.ReLU(),
                                    nn.Linear(128, D))
        
        # Print model summary
        print("VAE ENCODER:\n", summary(encoder_net, torch.zeros(1, D), show_input=False, show_hierarchical=False))
        print("\n VAE DECODER:\n", summary(decoder_net, torch.zeros(1, L), show_input=False, show_hierarchical=False))


        self.encoder = Encoder(encoder_net=encoder_net)
        self.decoder = Decoder(distribution=likelihood_type, decoder_net=decoder_net)
        self.prior = Prior(L=L)

        self.likelihood_type = likelihood_type

    def forward(self, x, reduction='avg'):
        # 1) Encode
        mu_e, log_var_e = self.encoder.encode(x)
        # 2) Sample z
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        # 3) compute KL
        KL = KL_divergence(self.prior, self.encoder, mu_e, log_var_e, z)

        # Older approaches:
        # RE = self.decoder.log_prob(x, z) # Reconstruction error
        # KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1) # KL divergence

        # RE = RE_log_prob(x, z, self.decoder)
        # KL = KL_divergence(self.prior, self.encoder, mu_e, log_var_e, z)
        # ELBO, RE, KL = calculate_ELBO(x, z, self.encoder, self.prior, self.decoder, reduction=reduction)

        return z, KL, mu_e, log_var_e

    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)