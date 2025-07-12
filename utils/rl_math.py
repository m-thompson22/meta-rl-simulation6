import torch

def compute_returns(rewards, values, gamma, device):
    """
    Bootstrapped n-step return:
    R_t = r_t + γ r_{t+1} + ... + γ^k V(s_{t+k})
    """
    R = values[-1].detach()  # Use last critic value as bootstrap
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.stack(returns)

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

def compute_advantage(returns, values):
    adv = returns - values.detach()
    return adv

def compute_losses(log_probs, advantage, returns, values, entropy, beta_e, beta_v):
    policy_loss = -(log_probs * advantage.detach()).mean()
    value_loss = (returns - values).pow(2).mean()
    entropy_loss = -entropy.mean()

    actor_loss = policy_loss + beta_e * entropy_loss
    critic_loss = beta_v * value_loss

    return actor_loss, critic_loss