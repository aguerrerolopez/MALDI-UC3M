from utils.probability_distributions import log_bernoulli, log_normal_diag
import torch
import torch.nn.functional as F
import numpy as np

def RE_log_prob(x, x_recon, distribution='bernoulli', reduction='avg'):

    # In the case of Bernoulli distribution
    if distribution == 'bernoulli':
        # In the Bernoulli case , we have x_d \in {0 ,1}. 
        # Therefore, it is enough to output a single probability,
        # because p(x_d =1|z) = \theta and p(x_d =0|z) = 1− \theta
        mu_d = torch.sigmoid(x_recon) # why exactly sigmoid?
        log_p = log_bernoulli(x, mu_d, reduction=reduction, dim=-1)

        # How to choose the reduction
        # Si la entrada es muy pequeña y la avg da casi cero porque de media las muestras son 0 entonces es mejor sum
        # Si la entrada y salida son fotos por ejemplo y están en valores grandes pues avg
    
    # In the case of Gaussian distribution
    elif distribution == 'gaussian':
        # The decoder outputs just the mean, std is fixed
        mu_d = x_recon
        log_var_d = torch.log(torch.tensor(0.1))  # Fixed log variance (log(0.1) or log(0.2), etc.) # 0.1 es mucho para los maldis, 10% de la media de mis datos
        log_p = log_normal_diag(x, mu_d, log_var_d, reduction=reduction)
    
    else:
        raise ValueError('Either `bernoulli` or `gaussian`')
    
    if reduction == 'avg':
        return log_p.mean()
    elif reduction == 'sum':
        return log_p.sum()
    else:
        return log_p  # shape [batch]

def kl_divergence(mu_q, log_var_q, reduction='sum'):
    """
    Compute KL divergence KL(q(z|x) || p(z)) between two distributions:
    q(z|x) = N(mu_q, std_q)
    p(z) = N(mu_p, std_p)

    KL(q(z|x) || p(z)) = 0.5 * (logvar_p - logvar_q + (std_q^2 + (mu_q - mu_p)^2) / std_p^2 - 1)
    where mu_p = 0, std_p = 1

    Args:
        mu_q: Mean of the posterior distribution q(z|x)
        log_var_q: Log variance of the posterior distribution q(z|x)
        reduction: Reduction method ('avg' or 'sum')
    Returns:
        KL divergence value
    """

    # Define mu and std of the prior distribution.
    mu_p = torch.zeros_like(mu_q)
    logvar_p = torch.zeros_like(log_var_q)

    # Get the std
    std_q = torch.exp(0.5 * log_var_q)
    std_p = torch.exp(0.5 * logvar_p)

    # Create the posterior and prior distributions
    q = torch.distributions.Normal(mu_q, std_q)
    p = torch.distributions.Normal(mu_p, std_p)

    # Calculate KL(q||p)
    kl = torch.distributions.kl_divergence(q, p)  # shape [batch, dim]

    if reduction == 'avg':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl

def RE(recon_x, x, loss_mode, reduction='sum'):

    # Choose the reconstruction loss based on the keyword.
    if loss_mode == 'bce':
        RE = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    elif loss_mode == 'mse':
        RE = F.mse_loss(recon_x, x, reduction=reduction)
    elif loss_mode == 'gaussian':
        # Assuming unit variance, the negative log-likelihood is proportional to MSE.
        RE = 0.5 * F.mse_loss(recon_x, x, reduction=reduction)
    else:
        raise ValueError("Unsupported loss mode. Use 'bce', 'mse' or 'gaussian'.")
    
    return RE
