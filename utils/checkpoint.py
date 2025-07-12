import os
import torch

def load_checkpoint(path, actor, critic, actor_optim, critic_optim, scheduler_a, scheduler_c, device):
    start_episode = 0
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"], strict=False)
        actor_optim.load_state_dict(checkpoint["actor_optimizer"])
        critic_optim.load_state_dict(checkpoint["critic_optimizer"])
        scheduler_a.load_state_dict(checkpoint["actor_scheduler"])
        scheduler_c.load_state_dict(checkpoint["critic_scheduler"])
        start_episode = checkpoint["episode"] + 1
        print(f"Resuming training from episode {start_episode}...")
    return start_episode

def save_checkpoint(actor, critic, actor_optim, critic_optim, scheduler_a, scheduler_c, episode, path):
    checkpoint_data = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer': actor_optim.state_dict(),
        'critic_optimizer': critic_optim.state_dict(),
        'actor_scheduler': scheduler_a.state_dict(),
        'critic_scheduler': scheduler_c.state_dict(),
        'episode': episode
    }
    torch.save(checkpoint_data, path)
    print(f"[Checkpoint] Saved to {path}")
