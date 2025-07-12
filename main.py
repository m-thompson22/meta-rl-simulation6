from config import Config
from utils.common import seed_everything, save_pickle
from core.init import initialize_actor_critic
from core.train import meta_train_control
from core.test import meta_test_all_conditions
from lesion.memory_lesion import meta_test_lesion

def main(mode="train", lesion_actor=False, reset_actor_each_episode=False):
    config = Config()
    seed_everything(config.seed)

    actor, critic, actor_optim, critic_optim, scheduler_a, scheduler_c = initialize_actor_critic(
        device=config.device,
        lr=config.learning_rate,
        hidden_size=config.hidden_size
    )

    if mode == "train":
        actor, critic = meta_train_control(
            actor=actor,
            critic=critic,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            scheduler_a=scheduler_a,
            scheduler_c=scheduler_c,
            config=config,
        )
        print("Meta-RL training complete.")

    elif mode == "test":
        results = meta_test_all_conditions(actor, critic, config)
        save_pickle(results, config.results_path)
        print("Standard test run complete.")

    elif mode == "lesion":
        # Run both lesion types: seq_len_only (default memory), reset_hx (memory reset each episode)
        for lesion_type in ["seq_len_only", "reset_hx"]:
            lesion_flag = lesion_type == "reset_hx"
            print(f"\nRunning lesion test: {lesion_type}")

            results = meta_test_lesion(
                actor=actor,
                critic=critic,
                config=config,
                lesion_actor=lesion_flag,  # mid-episode lesion
                reset_actor_each_episode=lesion_flag  # per-episode lesion
            )
            lesion_path = config.results_path.replace("results.pkl", f"lesion_{lesion_type}.pkl")
            save_pickle(results, lesion_path)
            print(f"Saved lesion results to: {lesion_path}")

    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "lesion"])
    parser.add_argument("--lesion_actor", action="store_true")
    parser.add_argument("--reset_actor_each_episode", action="store_true")
    args = parser.parse_args()

    main(
        mode=args.mode,
        lesion_actor=args.lesion_actor,
        reset_actor_each_episode=args.reset_actor_each_episode
    )
