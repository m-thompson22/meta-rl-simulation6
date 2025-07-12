import torch
import numpy as np
import pickle

from core.init import initialize_actor_critic
from .plot_diagnostics import plot_wang_figure_panel, plot_fixed_blocks_learning_rate_bar

def run_diagnostics():
    config = Config()

    # === Initialize and Load Trained Actor & Critic ===
    trained_actor, trained_critic, *_ = initialize_actor_critic(
        device=config.device,
        lr=config.learning_rate,
        hidden_size=config.hidden_size
    )

    trained_actor.load_state_dict(torch.load(config.actor_path, map_location=config.device))
    trained_critic.load_state_dict(torch.load(config.critic_path, map_location=config.device))
    trained_actor.eval()
    trained_critic.eval()
    print("[Info] Models loaded successfully.")

    # === Custom block schedule: volatile block mid-episode ===
    fixed_schedule = [
        (0, 0.5),      # Stable 1
        (50, 0.5),     # Continuation
        (75, 0.125),   # Volatile 1
        (120, 0.5),    # Volatile 2
        (140, 0.125),  # Stable 2
        (160, 0.125),  # Continuation
        (180, 0.125)   # Last segment (not evaluated if trials=200)
    ]

    save_path = "/content/gdrive_alt/MyDrive/train_meta_rl/lesion1_diagnostics.pkl"

    # === Run memory-lesioned meta-RL diagnostics ===
    diagnostic_results = run_meta_diagnostics(
        actor=trained_actor,
        critic=trained_critic,
        config=config,
        save_path=save_path,
        custom_block_schedule=fixed_schedule
    )

    print("[Success] Diagnostics complete.")
    print(f"[Saved] Results saved to: {save_path}")


if __name__ == "__main__":
    run_diagnostics()

import pickle
import matplotlib.pyplot as plt

def plot_impaired():
    diagnostic_path = "/content/gdrive_alt/MyDrive/train_meta_rl/lesion1_diagnostics.pkl"
    with open(diagnostic_path, "rb") as f:
        diagnostic_results = pickle.load(f)

    condition = "control"
    episode = 0

    for seq_len in [1, 5, 10, 20]:
        print(f"=== Plotting diagnostics with seq_len = {seq_len} ===")

        trial_data = {
            "rpes": diagnostic_results[f"seq_len_{seq_len}"][condition]["rpes"][episode],
            "rewards": diagnostic_results[f"seq_len_{seq_len}"][condition]["rewards"][episode],
            "p_riskys": diagnostic_results[f"seq_len_{seq_len}"][condition]["p_riskys"][episode],
            "arms": diagnostic_results[f"seq_len_{seq_len}"][condition]["arms"][episode],
            "switch_trials": diagnostic_results[f"seq_len_{seq_len}"][condition]["switch_trials"][episode],
        }

        # Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Wang-style diagnostic time series
        plot_wang_figure_panel(trial_data, title=f"seq_len={seq_len}", ax=axes[0])

        # Learning rate bar plot
        plot_fixed_blocks_learning_rate_bar(
            diagnostics_dict=diagnostic_results,
            seq_len_filter=f"seq_len_{seq_len}",
            condition_name_filter=condition,
            ax=axes[1]
        )

        plt.tight_layout()
        plt.show()