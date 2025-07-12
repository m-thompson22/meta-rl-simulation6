import os 
import torch

from core.config import Config
from core.init import initialize_actor_critic
from utils.common import save_pickle, seed_everything
from .memory_lesion import meta_test_lesion
from plotting.plot_utils import run_post_test_visualizations

def run_memory_lesioned_tests(base_config, seq_lens=[1, 5, 10], lesion_types=["seq_len_only", "reset_hx"], visualize=False):
    def get_default_feature_size(seq_len):
        return seq_len * 3 + 5  # reward, arm, interaction x seq + block types + bias

    results_by_condition = {}

    for seq_len in seq_lens:
        for lesion_type in lesion_types:
            print(f"\n=== Testing seq_len={seq_len}, lesion_type={lesion_type} ===")

            # Setup config
            config = Config()
            config.seq_len = seq_len
            config.device = base_config.device
            config.actor_path = base_config.actor_path
            config.critic_path = base_config.critic_path
            config.n_trials_test = base_config.n_trials_test
            config.num_episodes_test = 100

            # Output directory
            subdir = os.path.join(config.ROOT_DIR, "memory_lesions", f"{lesion_type}_seq{seq_len}")
            os.makedirs(subdir, exist_ok=True)
            config.results_path = os.path.join(subdir, "results.pkl")

            seed_everything(config.seed)

            # Load models
            actor, critic, *_ = initialize_actor_critic(
                device=config.device,
                lr=config.learning_rate,
                hidden_size=config.hidden_size
            )
            actor.load_state_dict(torch.load(config.actor_path, map_location=config.device))
            critic.load_state_dict(torch.load(config.critic_path, map_location=config.device))

            # Run lesion test with appropriate flags
            if lesion_type == "reset_hx":
                test_results = meta_test_lesion(
                    actor=actor,
                    critic=critic,
                    config=config,
                    lesion_actor=True
                )
            elif lesion_type == "seq_len_only":
                test_results = meta_test_lesion(
                    actor=actor,
                    critic=critic,
                    config=config,
                    lesion_actor=False
                )
            else:
                raise ValueError(f"Unknown lesion_type: {lesion_type}")

            results_by_condition[(lesion_type, seq_len)] = test_results
            save_pickle(test_results, config.results_path)
            print(f"Saved results to: {config.results_path}")

            if visualize:
                run_post_test_visualizations(
                    path=config.results_path,
                    n_trials=config.n_trials_test,
                    forced_trials=10
                )

    return results_by_condition
