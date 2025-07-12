import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
from torch.distributions import Categorical
import pickle 

from utils.analysis import extract_features, compute_logistic_coefficients
from envs.risky_safe import RiskySafeTask, get_arm_mapping, get_block_schedule_by_episode
from core.buffers import ActorInputBuffer,CriticInputBuffer

@torch.no_grad()
def lesion_diagnostic_sessions(actor, critic, config, episode=0, block_type="control", custom_block_schedule=None):
    n_trials = 200
    diagnostic_temp = 0.1

    actor_buffer = ActorInputBuffer(seq_len=config.seq_len)
    critic_buffer = CriticInputBuffer(seq_len=config.seq_len)

    arm_map, safe_action, risky_action = get_arm_mapping()
    forced_actions = [safe_action] * 5 + [risky_action] * 5
    np.random.shuffle(forced_actions)

    if custom_block_schedule is not None:
        block_schedule = custom_block_schedule
        switch_trials = [t for (t, _) in block_schedule[1:]]
    else:
        block_schedule, switch_trials = get_block_schedule_by_episode(episode)

    env = RiskySafeTask(block_schedule=block_schedule, arm_map=arm_map, block_type=block_type)
    hx_actor, hx_critic = None, None

    actions, arms, rewards, rpes = [], [], [], []
    p_riskys, entropies, policy_p_risky_log = [], [], []

    if episode % 50 == 0:
        print(f"Arm mapping: {arm_map}")

    for trial in range(n_trials):
        env.update_probability(trial)
        p_riskys.append(env.current_p_risky)
        is_forced = trial < 10

        if is_forced:
            current_action = forced_actions[trial]
            entropies.append(float('nan'))
            policy_p_risky_log.append(float('nan'))
        else:
            actor_input = actor_buffer.get_tensor().float().to(config.device)
            logits, hx_actor, _ = actor(actor_input, None)
            probs = torch.softmax(logits / diagnostic_temp, dim=-1)

            dist = Categorical(probs)
            current_action = dist.sample().item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            entropies.append(entropy.item())

            policy_p_risky = probs[0, risky_action].item()
            policy_p_risky_log.append(policy_p_risky)

            if episode % 50 == 0 and trial % 5 == 0:
                print(f"[Test | Ep {episode} | Trial {trial}] "
                      f"p_risky: {env.current_p_risky:.3f} | "
                      f"policy[p_risky]: {policy_p_risky:.3f} | "
                      f"action: {current_action}")

        reward, arm = env.get_reward(current_action)
        is_risky_arm = int(arm == "risky")
        is_safe_arm = int(arm == "safe")

        critic_input = critic_buffer.get_tensor().float().to(config.device)
        value, hx_critic = critic(critic_input, hx_critic)
        value = value.view(-1).item()

        next_value = 0.0
        if trial < n_trials - 1:
            try:
                next_input = critic_buffer.peek_next(reward, current_action).float().to(config.device)
                next_value, _ = critic(next_input, hx_critic)
                next_value = next_value.view(-1).item()
            except Exception as e:
                print(f"[Warning | Trial {trial}] next_value failed: {e}")

        rpe = reward + config.gamma * next_value - value

        if block_type == "block_risky_reward" and is_risky_arm and reward > 0:
            rpe -= 4
        elif block_type == "block_safe_reward" and is_safe_arm:
            rpe -= 1
        elif block_type == "block_risky_loss" and is_risky_arm and reward == 0:
            rpe += 1

        scaled_rpe = np.tanh(rpe)
        actor_buffer.update(current_action, scaled_rpe)
        critic_buffer.update(reward, current_action)

        actions.append(current_action)
        arms.append(arm)
        rewards.append(reward)
        if not is_forced:
            rpes.append(rpe)

    return {
        'actions': actions,
        'arms': arms,
        'rewards': rewards,
        'p_riskys': p_riskys,
        'block_types': [block_type] * n_trials,
        'switch_trials': switch_trials,
        'policy_p_risky': policy_p_risky_log,
        'arm_map': arm_map,
        'safe_action': safe_action,
        'risky_action': risky_action,
        'entropies': entropies,
        'rpes': rpes,
    }

def run_meta_diagnostics(actor, critic, config, save_path=None, custom_block_schedule=None, seq_lens=[1, 5, 10, 20]):
    conditions = ["control", "block_risky_reward", "block_safe_reward", "block_risky_loss"]
    full_results = {}

    for seq_len in seq_lens:
        print(f"\n=== Running diagnostics with seq_len = {seq_len} ===")
        config.seq_len = seq_len
        results_by_condition = {}

        for condition in conditions:
            print(f"\n[Condition: {condition}]")
            condition_results = defaultdict(list)

            for episode in range(100):
                trial_data = lesion_diagnostic_sessions(
                    actor=actor,
                    critic=critic,
                    config=config,
                    episode=episode,
                    block_type=condition,
                    custom_block_schedule=custom_block_schedule,
                )

                try:
                    feature_dim = 3  # RPE, one-hot action
                    default_size = seq_len * feature_dim + 5
                    X, y = extract_features(trial_data, n_lags=seq_len)
                    coef = compute_logistic_coefficients(X, y, default_size=default_size)
                except Exception as e:
                    print(f"[Episode {episode}] Logistic regression failed: {e}")
                    coef = np.full(5, np.nan)

                for key, val in trial_data.items():
                    condition_results[key].append(val)
                condition_results["logistic_weights"].append(coef)

            results_by_condition[condition] = dict(condition_results)

        full_results[f"seq_len_{seq_len}"] = results_by_condition

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(full_results, f)
        print(f"\n[Saved] Diagnostic results written to: {save_path}")

    return full_results
