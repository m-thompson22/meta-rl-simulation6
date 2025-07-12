from utils.common import load_pickle
from .test_plots import plot_trialwise_risky_choice, plot_risky_choice_by_p_risky, plot_blockwise_risky_choice_by_p, choice_plot

def get_color_map():
    return {
        "control": "blue",
        "block_risky_reward": "red",
        "block_safe_reward": "orange",
        "block_risky_loss": "green"
    }

def run_post_test_visualizations(path, n_trials, forced_trials=10):
    test_results = load_pickle(path)
    print("Loaded test_results from:", path)

    plot_trialwise_risky_choice(test_results, n_trials=n_trials)

    plot_risky_choice_by_p_risky(test_results, n_trials=n_trials)

    plot_blockwise_risky_choice_by_p(test_results, n_trials=n_trials, forced_trials=forced_trials)
    
    choice_plot(test_results, n_trials=n_trials)