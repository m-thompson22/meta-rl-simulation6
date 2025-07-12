import torch
import numpy as np

from config import Config
from core.buffers import ActorInputBuffer, CriticInputBuffer
from envs.risky_safe import RiskySafeTask
from envs.env_utils import get_arm_mapping
from envs.curriculum import generate_test_blocks
from utils.logging_utils import initialize_test_results
from utils.analysis import compute_logistic_coefficients, extract_features
from .utils import apply_dopamine_manipulation

@torch.no_grad()
def unrolled_session_without_learning(actor, critic, config, episode=0, block_type="control"):
    n_trials = config.n_trials_test
    test_temp = config.test_temp

    actor_buffer = ActorInputBuffer(seq_len=config.seq_len)
    critic_buffer = CriticInputBuffer(seq_len=config.seq_len)

    # === Arm mapping setup ===
    arm_map, safe_action, risky_action = get_arm_mapping()
    forced_actions = [safe_action] * 5 + [risky_action] * 5
    np.random.shuffle(forced_actions)

    block_schedule, switch_trials = generate_test_blocks(n_trials=n_trials)
    env = RiskySafeTask(block_schedule=block_schedule, arm_map=arm_map, block_type=block_type)

    hx_actor = None
    hx_critic = None


    # === Logging containers ===
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
            # === Get policy from actor ===
            actor_input = actor_buffer.get_tensor().float().to(config.device)
            logits, hx_actor, _ = actor(actor_input, hx_actor)
            probs = torch.softmax(logits / test_temp, dim=-1)

            dist = torch.distributions.Categorical(probs)
            current_action = dist.sample().item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            entropies.append(entropy.item())

            # === Correctly extract policy[p_risky] based on arm_map ===
            policy_p_risky = probs[0, risky_action].item()
            policy_p_risky_log.append(policy_p_risky)

            if episode % 50 == 0 and trial % 5 == 0:
                print(f"[Test | Ep {episode} | Trial {trial}] "
                      f"p_risky: {env.current_p_risky:.3f} | "
                      f"policy[p_risky]: {policy_p_risky:.3f} | "
                      f"action: {current_action}")

        # === Environment step ===
        reward, arm = env.get_reward(current_action)
        is_risky_arm = int(arm == "risky")
        is_safe_arm = int(arm == "safe")

        # === Critic value estimation and RPE ===
        critic_input = critic_buffer.get_tensor().float().to(config.device)
        value, hx_critic = critic(critic_input, hx_critic)
        value = value.view(-1).item()

        if trial < n_trials - 1:
          next_input = critic_buffer.peek_next(reward, current_action).float().to(config.device)
        # Save current hidden state before next forward pass
          hx_critic_copy = tuple(h.detach() for h in hx_critic)  # Detach to avoid backprop
          next_value, _ = critic(next_input, hx_critic_copy)
          next_value = next_value.view(-1).item()

        else:
          next_value = 0.0


        rpe = reward + config.gamma * next_value - value
        rpe = apply_dopamine_manipulation(rpe, block_type, is_risky_arm, is_safe_arm, reward)

        scaled_rpe = torch.clamp(rpe, min=-10.0, max=10.0)

        # === Update buffers ===
        actor_buffer.update(current_action, scaled_rpe.item())

        critic_buffer.update(reward, current_action)

        # === Append trial logs ===
        actions.append(current_action)
        arms.append(arm)
        rewards.append(reward)
        rpes.append(rpe if not is_forced else np.nan)

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

def meta_test_all_conditions(actor, critic, config):
    conditions = ["control", "block_risky_reward", "block_safe_reward", "block_risky_loss"]
    all_results = {}

    for condition in conditions:
        print(f"\nTesting condition: {condition}")
        results = initialize_test_results()

        for episode in range(config.num_episodes_test):
            if episode % 50 == 0:
                print(f"Episode {episode + 1}/{config.num_episodes_test}")

            trial_history = unrolled_session_without_learning(
                actor=actor,
                critic=critic,
                config=config,
                episode=episode,
                block_type=condition,
            )

            feature_dim_per_timestep = 3  # Or whatever your extract_features function uses
            default_size = config.seq_len * 3 + 5


            X, y = extract_features(trial_history, n_lags=config.seq_len)
            coef = compute_logistic_coefficients(X, y, default_size=default_size)


            results['rewards'].append(trial_history['rewards'])        # trial-level
            results['arms'].append(trial_history['arms'])              # trial-level
            results['actions'].append(trial_history['actions'])        # trial-level
            results['p_riskys'].append(trial_history['p_riskys'])      # trial-level
            results['entropies'].append(trial_history['entropies'])    # trial-level
            results['rpes'].append(trial_history.get('rpes', []))     # optional trial-level
            results['policy_p_risky'].append(trial_history.get('policy_p_risky', []))  # optional
            results['arm_map'].append(trial_history['arm_map'])        # episode-level
            results['safe_action'].append(trial_history['safe_action'])# episode-level
            results['risky_action'].append(trial_history['risky_action'])# episode-level
            results['switch_trials'].append(trial_history['switch_trials']) # episode-level

        all_results[condition] = results

    return all_results