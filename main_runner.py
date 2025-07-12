import os
from config import Config
import torch
from utils.common import seed_everything, save_pickle
from core.init import initialize_actor_critic
from core.train import meta_train_control
from core.test import meta_test_all_conditions

def run_training_and_testing():
    # === Set up config and seed ===
    config = Config()
    seed_everything(config.seed)

    # === Initialize actor-critic models ===
    actor, critic, actor_optim, critic_optim, scheduler_a, scheduler_c = initialize_actor_critic(config)
    
    # === Train ===
    actor, critic = meta_train_control(
        actor=actor,
        critic=critic,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        scheduler_a=scheduler_a,
        scheduler_c=scheduler_c,
        config=config,
    )
    print("Training complete.")

    # === Save final models ===
    torch.save(actor.state_dict(), config.actor_path)
    torch.save(critic.state_dict(), config.critic_path)
    print(f"Saved actor to: {config.actor_path}")
    print(f"Saved critic to: {config.critic_path}")

    # === Test ===
    results = meta_test_all_conditions(actor, critic, config)
    save_pickle(results, config.results_path)
    print(f"Test results saved to: {config.results_path}")

if __name__ == "__main__":
    run_training_and_testing()
