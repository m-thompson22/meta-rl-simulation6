import numpy as np
from sklearn.linear_model import LogisticRegression

def compute_logistic_coefficients(X, y, n_lags=None):
    default_size = X.shape[1] if len(X) > 0 else (n_lags * 3 + 4 + 1 if n_lags else 16)
    if len(X) == 0 or len(np.unique(y)) < 2:
        return np.zeros(X.shape[1]) if len(X) > 0 else np.zeros(default_size)
    try:
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X, y)
        return logreg.coef_[0]
    except Exception as e:
        print(f"Logistic regression error: {e}")
        return np.zeros(X.shape[1])
    
def extract_features(trial_history, n_lags=5):
    raw_arms = trial_history['arms']
    # Convert 'risky'/'safe' to 1/0 if needed
    if isinstance(raw_arms[0], str):
        arms = np.array([1 if a == 'risky' else 0 for a in raw_arms])
    else:
        arms = np.array(raw_arms)

    rewards = np.array(trial_history['rewards'])
    block_types = np.array(trial_history.get('block_types', ['control'] * len(arms)))

    block_names = ["control", "block_risky_reward", "block_safe_reward", "block_risky_loss"]
    X, y = [], []

    for t in range(n_lags, len(arms)):
        past_rewards = rewards[t - n_lags:t]
        past_arms = arms[t - n_lags:t]
        past_interactions = past_rewards * past_arms

        if len(past_rewards) < n_lags:
            past_rewards = np.pad(past_rewards, (n_lags - len(past_rewards), 0))
            past_arms = np.pad(past_arms, (n_lags - len(past_arms), 0))
            past_interactions = np.pad(past_interactions, (n_lags - len(past_interactions), 0))

        block_vector = [int(block_types[t] == b) for b in block_names]
        bias = [1.0]

        features = np.concatenate([
            past_rewards,
            past_arms,
            past_interactions,
            block_vector,
            bias
        ])

        X.append(features)
        y.append(arms[t])  # Predicting if risky arm is chosen

    return np.array(X), np.array(y)