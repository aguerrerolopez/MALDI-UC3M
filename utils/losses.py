from utils.probability_distributions import log_bernoulli, log_normal_diag
import torch

def RE_log_prob(x, x_recon, distribution='bernoulli', reduction='avg'):

    # In the case of Bernoulli distribution
    if distribution == 'bernoulli':
        # In the Bernoulli case , we have x_d \in {0 ,1}. 
        # Therefore, it is enough to output a single probability,
        # because p(x_d =1|z) = \theta and p(x_d =0|z) = 1− \theta
        mu_d = torch.sigmoid(x_recon)
        log_p = log_bernoulli(x, mu_d, reduction='avg', dim=-1)

        # How to choose the reduction
        # Si la entrada es muy pequeña y la avg da casi cero porque de media las muestras son 0 entonces es mejor sum
        # Si la entrada y salida son fotos por ejemplo y están en valores grandes pues avg
    
    # In the case of Gaussian distribution
    elif distribution == 'gaussian':
        # The decoder outputs just the mean, std is fixed
        mu_d = x_recon
        log_var_d = torch.log(torch.tensor(0.1))  # Fixed log variance (log(0.1) or log(0.2), etc.) # 0.1 es mucho para los maldis, 10% de la media de mis datos
        log_p = log_normal_diag(x, mu_d, log_var_d, reduction='avg')
    
    else:
        raise ValueError('Either `bernoulli` or `gaussian`')
    
    print(f"RE_log_prob mean: {log_p.mean()}")

    RE = abs(log_p.sum()) if reduction == 'sum' else abs(log_p.mean())

    return log_p, RE


def KL_divergence(log_p_z, log_q_z, reduction='avg'):
    """
    Computes KL(q(z|x) || p(z)) by comparing the log-probs of z under
    the prior vs. the encoder (approx posterior).
    Returns a per-sample KL, summed along sum_dim if desired.

    Args:
        log_p_z (Tensor): log p(z) from the prior.
        log_q_z (Tensor): log q(z|x) from the encoder.
    Returns:
        KL (Tensor): shape [batch_size] if sum_dim != None, or shape [batch_size, dimension].
    """
    # log p(z)
    # log q(z|x)
    KL = (log_p_z - log_q_z).sum(dim=-1)    # KL(q || p)
    KL = abs(KL.sum()) if reduction == 'sum' else abs(KL.mean())
    return KL

def calculate_ELBO(x, z, encoder, prior, decoder, reduction='avg'):
    """
    ELBO = - (RE(x, x_recon) + KL(q(z|x) || p(z)))
    where RE is the reconstruction error (negative log-likelihood).
    Returns a tensor [batch_size].
    """
    mu_e, log_var_e = encoder.encode(x)

    RE = RE_log_prob(x, z, decoder)
    KL = KL_divergence(prior, encoder, mu_e, log_var_e, z)
    ELBO = - (RE + KL)

    if reduction == 'sum':
            ELBO = -(RE + KL).sum()
            RE = abs(RE.sum())
            KL = abs(KL.sum())
    else:
        ELBO = -(RE + KL).mean()
        RE = abs(RE.mean())
        KL = abs(KL.mean())

    return ELBO, RE, KL