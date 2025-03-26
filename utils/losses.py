

def RE_log_prob(x, z, decoder):
    """
    Compute the conditional log-likelihood log p(x|z).
    Args:
        x (torch.Tensor): Original input (batch_size x n_features).
        z (torch.Tensor): Latent sample (batch_size x n_latents).
        decoder: your decoder object, with a .log_prob(x, z) method
    Returns:
        log_p (torch.Tensor): log-likelihood for each sample (batch_size).
    """
    return decoder.log_prob(x, z)


def KL_divergence(prior, encoder, mu_e, log_var_e, z, sum_dim=-1):
    """
    Computes KL(q(z|x) || p(z)) by comparing the log-probs of z under
    the prior vs. the encoder (approx posterior).
    Returns a per-sample KL, summed along sum_dim if desired.

    Args:
        prior: your prior object, with a .log_prob(z) method
        encoder: your encoder object, with a .log_prob(z=..., mu_e=..., log_var_e=...) method
        mu_e (Tensor): mean of the approximate posterior
        log_var_e (Tensor): log-variance of the approximate posterior
        z (Tensor): latent sample
        sum_dim (int): dimension along which to sum. Typically -1 for the features dimension.
    Returns:
        KL (Tensor): shape [batch_size] if sum_dim != None, or shape [batch_size, dimension].
    """
    log_p_z = prior.log_prob(z)                                         # log p(z)
    log_q_z = encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)     # log q(z|x)
    KL = (log_p_z - log_q_z).sum(dim=sum_dim)                           # KL(q || p)
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