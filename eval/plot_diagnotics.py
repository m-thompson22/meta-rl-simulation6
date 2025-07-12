import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import defaultdict

def rolling_std(x, window=5):
    return np.array([
        np.std(x[max(0, i - window // 2): i + window // 2 + 1])
        for i in range(len(x))
    ])

def plot_wang_figure_panel(trial_data, title="", normalize=True, ax=None):
    rewards = np.array(trial_data["rewards"])
    rpes = np.clip(np.array(trial_data["rpes"]), -10, 10)
    p_riskys = np.array(trial_data["p_riskys"])
    arms = np.array(trial_data["arms"])
    switch_trials = trial_data.get("switch_trials", [50, 75, 120, 140, 180])  # fallback

    # Truncate all signals to equal length
    n_trials = min(len(rewards), len(rpes), len(p_riskys), len(arms)) - 5
    rewards, rpes, p_riskys, arms = (
        rewards[:n_trials], rpes[:n_trials], p_riskys[:n_trials], arms[:n_trials]
    )
    x_vals = np.arange(n_trials)

    # === Learning Rate (|RPE| smoothed) ===
    learning_rate = np.abs(rpes)
    learning_rate[np.isnan(learning_rate)] = 0
    learning_rate = savgol_filter(learning_rate, 11, 2)
    learning_rate = np.clip(learning_rate, 0, None)
    if normalize:
        learning_rate /= np.max(learning_rate) + 1e-6

    # === Inferred Volatility ===
    rewards_norm = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-6)
    volatility = rolling_std(rewards_norm, window=10)
    volatility = savgol_filter(volatility, 11, 2)
    volatility = np.clip(volatility, 0, None)
    if normalize:
        volatility /= np.max(volatility) + 1e-6

    # === Stepwise p(risky) ===
    p_riskys_step = np.repeat(p_riskys, 2)[1:]
    x_step = np.repeat(x_vals, 2)[1:]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    # === Plot signals ===
    ax.plot(x_step, p_riskys_step, color='blue', label='Reward probability', linewidth=2)
    ax.plot(x_vals, learning_rate, color='red', label='Learning rate (|RPE|)', linewidth=2)
    ax.plot(x_vals, volatility, color='green', linestyle='--', label='Inferred volatility', linewidth=1.5)

    for s in switch_trials:
        if s < n_trials:
            ax.axvline(x=s, color='gray', linestyle='--', alpha=0.4)

    for i in range(n_trials):
        marker = 'x' if arms[i] == 0 else 'o'
        ax.plot(i, 1.08, marker=marker, color='gray', markersize=4, linestyle='None', alpha=0.6)

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0, n_trials)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Normalized")
    ax.legend(loc='lower left', bbox_to_anchor=(0, -0.25), ncol=4)

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_fixed_blocks_learning_rate_bar(
    diagnostics_dict,
    seq_len_filter=None,
    condition_name_filter=None,
    ax=None
):
    def get_fixed_block_tags(n_trials):
        tags = [None] * n_trials
        ranges = {
            'stable 1': (50, 75),
            'volatile 1': (75, 120),
            'volatile 2': (120, 140),
            'stable 2': (140, 160)
        }
        for tag, (start, end) in ranges.items():
            for i in range(start, min(end, n_trials)):
                tags[i] = tag
        return tags

    blockwise_rpes = defaultdict(list)

    for seq_key, cond_dict in diagnostics_dict.items():
        if seq_len_filter and not seq_key.startswith(seq_len_filter):
            continue
        for condition_name, episode_dict in cond_dict.items():
            if condition_name_filter and condition_name != condition_name_filter:
                continue

            rpes_all_episodes = episode_dict.get("rpes", [])
            if not isinstance(rpes_all_episodes, list):
                continue

            for rpes in rpes_all_episodes:
                if len(rpes) < 160:
                    continue
                tags = get_fixed_block_tags(len(rpes))
                for i, tag in enumerate(tags):
                    if tag is None or i >= len(rpes):
                        continue
                    blockwise_rpes[tag].append(abs(rpes[i]))

    block_order = ['stable 1', 'volatile 1', 'volatile 2', 'stable 2']
    means = [np.mean(blockwise_rpes[tag]) if blockwise_rpes[tag] else 0 for tag in block_order]
    errors = [np.std(blockwise_rpes[tag]) / np.sqrt(len(blockwise_rpes[tag])) if blockwise_rpes[tag] else 0 for tag in block_order]
    colors = ['#7EB6FF' if 'stable' in tag else '#FFA155' for tag in block_order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(block_order, means, yerr=errors, color=colors, capsize=4)
    ax.set_ylabel("Learning rate (|RPE|)")
    title = f"Fixed Blockwise Learning Rate — {seq_len_filter or 'All'}"
    if condition_name_filter:
        title += f" | {condition_name_filter}"
    ax.set_title(title)
    ax.set_ylim(0, max(means + [0.1]))
    ax.set_xticks(np.arange(len(block_order)))
    ax.set_xticklabels(block_order, rotation=20)

    if ax is None:
        plt.tight_layout()
        plt.show()

