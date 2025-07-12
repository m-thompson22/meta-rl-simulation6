import torch

def compute_regularization_terms(logits, temp, episode, config):
    logits = logits - logits.max(dim=-1, keepdim=True)[0]
    probs = torch.softmax(logits / temp, dim=-1)

    if episode < 20000:
        lambda_sym = config.symmetry_lambda_start / (1.0 + config.lambda_decay_rate * episode)
        lambda_kl = config.kl_lambda_start / (1.0 + config.lambda_decay_rate * episode)
    else:
        lambda_sym = 0.0
        lambda_kl = 0.0

    logit_sym_penalty = torch.mean((logits[:, 0] - logits[:, 1]) ** 2)
    uniform_dist = torch.full_like(probs, fill_value=1.0 / probs.shape[-1])
    kl_div = torch.sum(probs * (probs.log() - uniform_dist.log()), dim=-1).mean()

    return lambda_sym * logit_sym_penalty + lambda_kl * kl_div
