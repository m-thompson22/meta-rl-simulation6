import numpy as np

def get_beta_e(episode, config):
    """Exponentially decaying beta_e with floor."""
    if episode < config.beta_decay_start:
        return config.beta_start
    decayed = config.beta_start * (config.beta_decay_rate ** (episode - config.beta_decay_start))
    return max(config.beta_min, decayed)

def detach_hidden_state(hx):
    return tuple(h.detach() for h in hx) if hx is not None else None


def apply_dopamine_manipulation(rpe, block_type, is_risky_arm, is_safe_arm, reward):
    """
    Modifies RPE based on dopamine manipulation condition.
    
    """
    if block_type == "block_risky_reward" and is_risky_arm and reward > 0:
        rpe -= 4
    elif block_type == "block_safe_reward" and is_safe_arm:
        rpe -= 1
    elif block_type == "block_risky_loss" and is_risky_arm and reward == 0:
        rpe += 1
    return rpe
