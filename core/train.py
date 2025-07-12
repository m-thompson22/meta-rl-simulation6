import torch
import numpy as np
import os 
from .buffers import ActorInputBuffer, CriticInputBuffer
from .utils import get_beta_e, detach_hidden_state
from envs.risky_safe import RiskySafeTask
from envs.env_utils import get_arm_mapping
from envs.curriculum import get_block_schedule_by_episode
from utils.rl_math import compute_returns, compute_losses, compute_advantage, compute_entropy
from utils.optim_utils import run_optimizer_step
from utils.loss_regularization import compute_regularization_terms
from utils.logging_utils import BufferedLogger, print_episode_diagnostics
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.common import save_pickle

def train_unrolled_session(actor, critic, actor_optim, critic_optim, scheduler_a, scheduler_c, config, episode, block_type="control"):

    logger = BufferedLogger(
    episode=episode,
    condition=block_type,
    print_every=500,
    trial_every=5,
    to_file=False  # or True if you want to save logs to disk
    )

    gamma, beta_v, temp = config.gamma, config.beta_v, config.train_temp
    beta_e = get_beta_e(episode, config)
    

    actor_buffer = ActorInputBuffer(config.seq_len)
    critic_buffer = CriticInputBuffer(config.seq_len)

    block_schedule, switch_trials = get_block_schedule_by_episode(episode)
    switch_trials = [int(s) for s in switch_trials]

    arm_map, safe_action, risky_action = get_arm_mapping()
    forced_actions = np.random.permutation([safe_action] * 5 + [risky_action] * 5)
    env = RiskySafeTask(block_schedule, arm_map, block_type)

    hx_actor, hx_critic = None, None

    actions, log_probs, values, rewards, entropies, raw_rpes = [], [], [], [], [], []
    diagnostic_log = {"trial": [], "true_p_risky": [], "policy_p_risky": [], "chosen_arm": []}

    if episode % 500 == 0:
        print(f"[Ep {episode}] Block switches at trials: {switch_trials}")
        print(f"[Ep {episode}] Arm Mapping: {arm_map}")
        print(f"[Ep {episode}] hx_actor at start: {hx_actor}")

    for trial in range(config.n_trials):
        env.update_probability(trial)
        is_forced = trial < config.force_trials

        actor_input = actor_buffer.get_tensor().float().to(config.device)
        logits, hx_actor, _ = actor(actor_input, hx_actor)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)

        if is_forced:
            action = forced_actions[trial]
            log_prob = None
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor).unsqueeze(0)

        reward, arm = env.get_reward(action)

        # === Compute critic prediction *before* buffer update
        critic_input = critic_buffer.get_tensor().float().to(config.device)
        value, hx_critic = critic(critic_input, hx_critic)
        value = value.view(-1)

        # === Compute TD target using peek buffer (not yet updated!)
        if trial < config.n_trials - 1:
          next_critic_input = critic_buffer.peek_next(reward, action).float().to(config.device)
          hx_critic_detached = detach_hidden_state(hx_critic)
          next_value, _ = critic(next_critic_input, hx_critic_detached)
          next_value = next_value.view(-1)

        else:
          next_value = torch.zeros_like(value)

        # === Compute RPE
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=config.device)
        rpe = reward_tensor + gamma * next_value - value
        rpe = rpe.detach()
        scaled_rpe = torch.clamp(rpe, min=-10.0, max=10.0)
        

        # === Now update buffers
        actor_buffer.update(action, scaled_rpe.item())
        critic_buffer.update(reward, action)


        rewards.append(torch.tensor(reward, dtype=torch.float32, device=config.device))
        actions.append(action)

        if not is_forced:
            log_probs.append(log_prob)
            values.append(value)
            entropy = compute_entropy(probs)
            entropies.append(entropy)
            raw_rpes.append(rpe.detach().cpu().item())

            policy_p_risky = next(probs[0, idx].detach().cpu().item() for idx, label in arm_map.items() if label == "risky")
            chosen_arm = arm_map[action]

            diagnostic_log["trial"].append(trial)
            diagnostic_log["true_p_risky"].append(env.current_p_risky)
            diagnostic_log["policy_p_risky"].append(policy_p_risky)
            diagnostic_log["chosen_arm"].append(chosen_arm)

            logger.log(
                trial=trial,
                p_risky=env.current_p_risky,
                policy_p_risky=policy_p_risky,
                action=chosen_arm,
                entropy=entropy.item()
            )

    if log_probs:
        log_probs_tensor = torch.stack(log_probs)
        values_tensor = torch.stack(values)
        rewards_tensor = torch.stack(rewards[-len(log_probs):])
        entropy_tensor = torch.stack(entropies)

        returns = compute_returns(rewards_tensor, values_tensor, gamma, config.device)
        advantage = compute_advantage(returns, values_tensor, clamp=True)


        hx_actor = detach_hidden_state(hx_actor)
        hx_critic = detach_hidden_state(hx_critic)

        actor_loss, critic_loss = compute_losses(log_probs_tensor, advantage, returns,
                                                 values_tensor, entropy_tensor, beta_e, beta_v)
        
        with torch.no_grad():
            final_actor_input = actor_buffer.get_tensor().float().to(config.device)
            logits_final, _, _ = actor(final_actor_input, hx_actor)
           
        reg_loss = compute_regularization_terms(logits_final, temp, episode, config)
        actor_loss += reg_loss

        actor_grad_pre, actor_grad_post = run_optimizer_step(actor_loss, actor_optim, actor, max_norm=5.0, return_grad_norm=True)
        critic_grad_pre, critic_grad_post = run_optimizer_step(critic_loss, critic_optim, critic, max_norm=10.0, return_grad_norm=True)

        # Step learning rate schedulers
        scheduler_a.step()
        scheduler_c.step()

    else:
        actor_grad_pre, actor_grad_post = None, None
        critic_grad_pre, critic_grad_post = None, None

    if episode % 500 == 0 and hx_actor is not None:
        print_episode_diagnostics(
        episode=episode,
        trial=trial,
        scaled_logits=scaled_logits,
        hx_actor=hx_actor,
        entropies=entropies,
        actor_grad_post=actor_grad_post,
        critic_grad_post=critic_grad_post,
        diagnostic_log=diagnostic_log,
        raw_rpes=raw_rpes
    )
    logger.flush()
    
    return {
        'actions': actions,
        'rewards': [r.detach().cpu().item() for r in rewards],
        'p_riskys': diagnostic_log["true_p_risky"],
        'block_types': [block_type] * config.n_trials,
        'diagnostic_log': diagnostic_log,
        'switch_trials': switch_trials,
        'total_reward': sum(r.detach().cpu().item() for r in rewards),
        'avg_entropy': torch.stack(entropies).mean().detach().cpu().item() if entropies else None,
        'arm_map': arm_map,
        'safe_action': safe_action,
        'risky_action': risky_action,
        'actor_grad': actor_grad_pre,
        'critic_grad': critic_grad_pre,
        'actor_grad_clipped': actor_grad_post,
        'critic_grad_clipped': critic_grad_post,
        'rpes': raw_rpes
    }

def meta_train_control(actor, critic, actor_optim, critic_optim, scheduler_a, scheduler_c, config):

    checkpoint_path = os.path.join(config.ROOT_DIR, f"checkpoint.pt")
    start_episode = 0

    # Load checkpoint if it exists
    checkpoint_path = os.path.join(config.ROOT_DIR, "checkpoint.pt")
    
    start_episode = load_checkpoint(
            checkpoint_path, actor, critic,
            actor_optim, critic_optim,
            scheduler_a, scheduler_c,
            config.device
        )
    print(f"\nTRAINING: {config.num_episodes_train} EPISODES in CONTROL condition")

    # Initialize logs
    train_logs = {
        'episode': [],
        'total_reward': [],
        'avg_entropy': [],
    }

    for episode in range(start_episode, config.num_episodes_train):
        if episode % 1000 == 0:
            print(f"Episode {episode + 1}/{config.num_episodes_train}")

        # Run a training episode
        result = train_unrolled_session(
            actor=actor,
            critic=critic,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            scheduler_a=scheduler_a,
            scheduler_c=scheduler_c,
            config=config,
            episode=episode,
            block_type="control",
        )

        # Log metrics
        train_logs['episode'].append(episode)
        train_logs['total_reward'].append(result['total_reward'])
        train_logs['avg_entropy'].append(result['avg_entropy'])

        # Save checkpoints every 10,000 episodes (not at episode 0)
        if episode > 0 and episode % 1000 == 0:
            save_checkpoint(
                actor, 
                critic, 
                actor_optim, 
                critic_optim, 
                scheduler_a, scheduler_c, 
                episode, 
                checkpoint_path
            )
            print(f"[Checkpoint] Saved to {checkpoint_path}")

    # Save final model weights
    print("Training complete!\n")
    torch.save(actor.state_dict(), config.actor_path)
    torch.save(critic.state_dict(), config.critic_path)
    print(f"Saved final actor to: {config.actor_path}")
    print(f"Saved final critic to: {config.critic_path}")

    # Save training logs
    logs_path = os.path.join(config.ROOT_DIR, "train_logs2.pkl")
    save_pickle(train_logs, logs_path)
    print(f"Saved training logs to: {logs_path}")

    return actor, critic