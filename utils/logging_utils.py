import numpy as np
import torch

class BufferedLogger:
    def __init__(self, episode, condition, print_every=500, trial_every=5, to_file=False):
        self.episode = episode
        self.condition = condition
        self.print_every = print_every
        self.trial_every = trial_every
        self.to_file = to_file
        self.logs = []

    def log(self, trial, p_risky, policy_p_risky, action, entropy):
        if self.episode % self.print_every == 0 and trial % self.trial_every == 0:
            log_str = (f"[Ep {self.episode} | Trial {trial}] "
                       f"p_risky: {p_risky:.3f} | policy_p_risky: {policy_p_risky:.3f} | "
                       f"action: {action} | entropy: {entropy:.3f}")
            self.logs.append(log_str)

    def flush(self):
        for line in self.logs:
            print(line)
        self.logs = [] 

def print_episode_diagnostics(episode, trial, scaled_logits, hx_actor, entropies, actor_grad_post, critic_grad_post, diagnostic_log, raw_rpes):
    h, c = hx_actor
    avg_entropy = torch.stack(entropies).mean().detach().cpu().item() if entropies else None
    print(f"\n=== Episode {episode} | Trial {trial} Diagnostics ===")
    print(f"hx_actor h norm: {h.norm().detach().cpu().item():.4f} | c norm: {c.norm().detach().cpu().item():.4f}")
    print("Actor logits:", scaled_logits.detach().cpu().numpy())
    print(f"Avg Entropy: {avg_entropy:.4f}")
    print(f"Actor grad norm: {actor_grad_post:.4f} | Critic grad norm: {critic_grad_post:.4f}")

    risky_choices_by_pr = {0.125: [], 0.5: []}
    for arm_label, pr in zip(diagnostic_log["chosen_arm"][10:], diagnostic_log["true_p_risky"][10:]):
        risky_choices_by_pr[pr].append(1 if arm_label == "risky" else 0)

    abs_rpe_by_pr = {pr: [] for pr in set(diagnostic_log["true_p_risky"])}
    for pr, rpe_val in zip(diagnostic_log["true_p_risky"], raw_rpes):
        abs_rpe_by_pr[pr].append(abs(rpe_val))

    print(f"\n[Ep {episode:>5}] Risky Choice Proportion by p_risky:")
    for pr, choices in risky_choices_by_pr.items():
        if choices:
            avg_abs_rpe = np.mean(abs_rpe_by_pr[pr]) if abs_rpe_by_pr[pr] else float('nan')
            print(f"  p_risky = {pr:.3f} | risky choice prop = {np.mean(choices):.3f} | n = {len(choices)} | avg |RPE| = {avg_abs_rpe:.4f}")
        else:
            print(f"  p_risky = {pr:.3f} | no free choice trials")
    print("=" * 50)

def initialize_test_results():
    return {
        'rewards': [],
        'actions': [],
        'coefs': [],
        'p_riskys': [],
        'block_types': [],
        'arms': [],
        'entropies': [],
        'rpes': []
    }