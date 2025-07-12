import numpy as np

def get_arm_mapping():
    if np.random.rand() < 0.5:
        arm_map = {0: "safe", 1: "risky"}
    else:
        arm_map = {0: "risky", 1: "safe"}

    safe_action = [k for k, v in arm_map.items() if v == "safe"][0]
    risky_action = [k for k, v in arm_map.items() if v == "risky"][0]
    return arm_map, safe_action, risky_action