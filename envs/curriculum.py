import numpy as np

def get_block_schedule_by_episode(episode_num, n_trials=100, transition_episode=5000):
    """
    Returns a block schedule and switch trials based on training phase.

    - Early phase (episode < transition_episode):
        2 blocks, switch at trial 30 ± 5
    - Later phase (episode ≥ transition_episode):
        3 blocks, first switch between 15–35, second switch ≥10 trials later, ending ≤ 55
    """
    if episode_num < transition_episode:
        return generate_early_curriculum_schedule(n_trials=n_trials)
    else:
        return generate_block_schedule_balanced(n_trials=n_trials)

def generate_early_curriculum_schedule(n_trials=100):
    """
    Generate a 2-block schedule with a fixed switch around trial trial 45-55, 
    ensuring scheule fits within n_trials
    """
    schedule = []
    switches = []

    # Initial block
    trial = 0
    current_p = np.random.choice([0.125, 0.5])
    schedule.append((trial, current_p))

    # Switch point
    switch = np.random.randint(45, 55)
    switches.append(switch)

    # Second block
    trial = switch
    current_p = 0.125 if current_p == 0.5 else 0.5
    schedule.append((trial, current_p))

    return schedule, switches

def generate_block_schedule_balanced(n_trials,
                                     first_switch_range=(30, 45),
                                     min_gap=20,
                                     min_last_block_len=10,
                                     gaussian_std=5,
                                     second_switch_prob=0.7):
    """
    Generates a 2–3 block schedule with:
      - First switch drawn from a soft Gaussian centered in first_switch_range.
      - If no second switch is added, center the first switch.
    """
    schedule = []
    switches = []

    # === Block 1 ===
    trial = 0
    current_p = np.random.choice([0.125, 0.5])
    schedule.append((trial, current_p))

    # === Decide early whether we will use a second switch ===
    use_second_switch = (
        np.random.rand() < second_switch_prob and
        (first_switch_range[1] + min_gap < n_trials - min_last_block_len)
    )

    # === First switch ===
    if use_second_switch:
        # Use broad Gaussian within first_switch_range
        options1 = np.arange(first_switch_range[0], first_switch_range[1] + 1)
        mean1 = np.mean(first_switch_range)
    else:
        # Force centered switch (e.g., trials 40–60)
        options1 = np.arange(40, 61)
        mean1 = np.mean(options1)  # center of 40–60

    weights1 = np.exp(-((options1 - mean1) ** 2) / (2 * gaussian_std ** 2))
    weights1 /= weights1.sum()

    switch1 = np.random.choice(options1, p=weights1)
    switches.append(switch1)
    trial = switch1
    current_p = 0.125 if current_p == 0.5 else 0.5
    schedule.append((trial, current_p))

    # === Optional second switch ===
    if use_second_switch:
        min_switch2 = switch1 + min_gap
        max_switch2 = n_trials - min_last_block_len

        if min_switch2 < max_switch2:
            options2 = np.arange(min_switch2, max_switch2 + 1)
            mean2 = (min_switch2 + max_switch2) / 2
            weights2 = np.exp(-((options2 - mean2) ** 2) / (2 * gaussian_std ** 2))
            weights2 /= weights2.sum()

            switch2 = np.random.choice(options2, p=weights2)
            switches.append(switch2)
            trial = switch2
            current_p = 0.125 if current_p == 0.5 else 0.5
            schedule.append((trial, current_p))

    return schedule, switches

def generate_test_blocks(
    n_trials=120,
    cluster_centers=[30, 60, 90],
    min_gap=15,
    min_last_block_len=15,
    gaussian_std=6
):
    """
    Generates a 4-block schedule (3 switches) clustered around specified centers.
    """
    assert len(cluster_centers) == 3, "Must provide 3 cluster centers for 3 switches."

    schedule = []
    switches = []

    current_p = np.random.choice([0.125, 0.5])
    schedule.append((0, current_p))

    last_switch = 0
    for i, center in enumerate(cluster_centers):
        min_switch = last_switch + min_gap
        max_switch = n_trials - (min_last_block_len * (3 - i))

        options = np.arange(min_switch, max_switch + 1)
        weights = np.exp(-((options - center) ** 2) / (2 * gaussian_std ** 2))
        weights /= weights.sum()

        switch = np.random.choice(options, p=weights)
        switches.append(switch)

        current_p = 0.125 if current_p == 0.5 else 0.5
        schedule.append((switch, current_p))

        last_switch = switch

    return schedule, switches