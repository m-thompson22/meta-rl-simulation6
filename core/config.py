import os
import sys
import argparse
import torch
from datetime import datetime

class Config:
    def __init__(self):
        # === Defaults set in self ===
        self.root_dir = "./runs/meta_rl"
        self.use_timestamp_subdir = False # set true to enable timestamp

        self.seed = 0
        self.force_trials = 10
        self.n_trials = 100
        self.n_trials_test = 120
        self.seq_len = 10
        self.hidden_size = 48

        self.gamma = 0.9
        self.beta_v = 0.03
        self.beta_e = 0.1
        self.entropy_min = 0.01
        self.actor_lr = 1e-4
        self.critic_lr = 1e-5

        self.num_episodes_train = 100000
        self.num_episodes_test = 1000

        self.train_temp = 0.5
        self.beta_start = 0.1
        self.beta_min = 0.05
        self.beta_decay_start = 30000
        self.beta_decay_rate = 0.99995

        self.symmetry_lambda_start = 0.01
        self.kl_lambda_start = 0.01
        self.lambda_decay_rate = 0.0005
        self.lambda_stop_episode = 20000

        self.test_temp = 0.1

        # === Optional overrides from argparse ===
        self.override_from_args()

        # === Timestamped run directory ===
        if self.use_timestamp_subdir:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.root_dir = os.path.join(self.root_dir, timestamp)

        os.makedirs(self.root_dir, exist_ok=True)

        # === Paths and device ===
        self.actor_path = os.path.join(self.root_dir, "actor_final.pt")
        self.critic_path = os.path.join(self.root_dir, "critic_final.pt")
        self.results_path = os.path.join(self.root_dir, "test_results.pkl")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _prepare_run_directory(self):
        if self.use_timestamp_subdir:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            root = os.path.join(self.root_dir, timestamp)
        else:
            root = self.root_dir
        os.makedirs(root, exist_ok=True)
        return root
    
    def override_from_args(self):
        parser = argparse.ArgumentParser()

        # Define only the args you want to override
        parser.add_argument("--root-dir", type=str)
        parser.add_argument("--use-timestamp-subdir", action="store_true")
        parser.add_argument("--seed", type=int)
        parser.add_argument("--n-trials", type=int)
        parser.add_argument("--n-trials-test", type=int)
        parser.add_argument("--seq-len", type=int)
        parser.add_argument("--hidden-size", type=int)
        parser.add_argument("--learning-rate", type=float)
        parser.add_argument("--gamma", type=float)
        parser.add_argument("--beta-v", type=float)
        parser.add_argument("--beta-e", type=float)
        parser.add_argument("--temp", type=float)
        parser.add_argument("--entropy-min", type=float)
        parser.add_argument("--num-episodes-train", type=int)
        parser.add_argument("--num-episodes-test", type=int)

        args = parser.parse_args([] if "google.colab" in sys.modules else None)

        for key, value in vars(args).items():
            if value is not None:  # Only override if user passed it
                setattr(self, key, value)
