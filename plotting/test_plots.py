import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

from plot_utils import get_color_map

color_map = get_color_map()

# === Plot 1: Risky choice over time ===
def plot_trialwise_risky_choice(test_results, n_trials):
    plt.figure(figsize=(10, 6))
    for condition, data in test_results.items():
        usable = [ep for ep in data['arms'] if len(ep) == n_trials]
        if usable:
            # Convert 'safe'/'risky' strings to numerical (e.g., 0/1)
            numerical_arms = [[1 if arm == 'risky' else 0 for arm in episode_arms] for episode_arms in usable]
            arr = np.array(numerical_arms)
            avg_risky = arr.mean(axis=0)
            plt.plot(range(n_trials), avg_risky,
                     label=condition.replace("_", " "),
                     color=color_map[condition],
                     linewidth=2)

    plt.axvline(x=10, color='black', linestyle='--', alpha=0.7, label='Free choice starts')
    plt.xlabel("Trial")
    plt.ylabel("Proportion Risky")
    plt.title("Risky Choice Over Time")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Plot 2: Behavior within blocks (proportion risky choice) ===
def plot_blockwise_risky_choice_by_p(test_results, n_trials=120, forced_trials=10):
    """
    For each dopamine condition, plot risky choice for high vs. low p(risky) blocks.
    Uses fixed colors: skyblue for high p(risky), salmon for low p(risky).
    Compatible with variable switch_trials per episode and excludes forced trials.
    """
    means, errors, labels, colors = [], [], [], []

    for cond, data in test_results.items():
        risky_high, risky_low = [], []

        for risky_ep, p_ep, switch_trials in zip(data["arms"], data["p_riskys"], data["switch_trials"]):
            if len(risky_ep) != n_trials:
                continue

            # Convert 'safe'/'risky' strings to numerical (e.g., 0/1)
            numerical_risky_ep = np.array([1 if arm == 'risky' else 0 for arm in risky_ep])

            # Create block boundaries, starting after forced trials
            block_starts = [forced_trials] + [s for s in switch_trials if s >= forced_trials] + [n_trials]

            for start, end in zip(block_starts[:-1], block_starts[1:]):
                if end <= start:
                    continue  # skip invalid ranges
                block_risky = np.mean(numerical_risky_ep[start:end])
                block_p = np.mean(p_ep[start:end])

                if block_p > 0.3:
                    risky_high.append(block_risky)
                else:
                    risky_low.append(block_risky)

        # Append means, errors, and labels
        means.extend([np.mean(risky_high), np.mean(risky_low)])
        errors.extend([sem(risky_high), sem(risky_low)])
        labels.extend([f"{cond}\nHigh p(risky)", f"{cond}\nLow p(risky)"])
        colors.extend(["skyblue", "salmon"])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(means))
    ax.bar(x, means, yerr=errors, capsize=4, color=colors, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Proportion Risky Chosen")
    ax.set_title("Risky Choice by Blockwise p(risky) Across Dopamine Conditions")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# === Plot 3: Risky choice by p_risky level ===
def plot_risky_choice_by_p_risky(test_results, n_trials):
    grouped = {cond: {'high': [], 'low': []} for cond in test_results}

    for cond, data in test_results.items():
        for ep_risky, ep_prob in zip(data['arms'], data['p_riskys']):
            if len(ep_risky) != n_trials:
                continue

            # Convert 'safe'/'risky' strings to numerical (e.g., 0/1)
            numerical_ep_risky = np.array([1 if arm == 'risky' else 0 for arm in ep_risky])

            p_start = ep_prob[0]
            key = 'high' if p_start > 0.3 else 'low'
            grouped[cond][key].append(numerical_ep_risky)

    plt.figure(figsize=(12, 6))
    for cond in test_results:
        for level, label_suffix in [('high', ' (High p_risky)'), ('low', ' (Low p_risky)')]:
            data = np.array(grouped[cond][level])
            if len(data) == 0:
                continue
            mean = data.mean(axis=0)
            label = f"{cond.replace('_', ' ')}{label_suffix}"
            linestyle = "-" if level == "high" else "--"
            plt.plot(range(n_trials), mean, label=label,
                     color=color_map[cond], linestyle=linestyle, linewidth=2)

    plt.axvline(x=10, linestyle='--', color='k', label="Free choice starts")
    plt.ylim(0, 1)
    plt.xlabel("Trial")
    plt.ylabel("Proportion Risky Chosen")
    plt.title("Risky Choice by p(risky) Level")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def choice_plot(test_results, n_trials):
    grouped = {cond: {'high': [], 'low': []} for cond in test_results}

    for cond, data in test_results.items():
        for ep_risky, ep_prob in zip(data['arms'], data['p_riskys']):
            if len(ep_risky) != n_trials:
                continue

            numerical_ep_risky = np.array([1 if arm == 'risky' else 0 for arm in ep_risky])
            p_start = ep_prob[0]
            key = 'high' if p_start > 0.3 else 'low'
            grouped[cond][key].append(numerical_ep_risky)

    n_conditions = len(test_results)
    ncols = 2
    nrows = (n_conditions + 1) // ncols  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), sharey=True)
    axes = axes.flatten()

    for ax, cond in zip(axes, test_results):
        for level, label_suffix in [('high', ' (High p_risky)'), ('low', ' (Low p_risky)')]:
            data = np.array(grouped[cond][level])
            if len(data) == 0:
                continue
            mean = data.mean(axis=0)
            label = f"{cond.replace('_', ' ')}{label_suffix}"
            linestyle = "-" if level == "high" else "--"
            ax.plot(range(n_trials), mean, label=label,
                    color=color_map.get(cond, 'black'), linestyle=linestyle, linewidth=2)

        ax.axvline(x=10, linestyle='--', color='k', label="Free choice starts")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Proportion Risky Chosen")
        ax.set_title(f"{cond.replace('_', ' ').title()}")
        ax.grid(alpha=0.3)
        ax.legend()

    # Hide any unused axes
    for ax in axes[n_conditions:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()