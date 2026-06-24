"""
QUICK baseline run — 论文原版 TD3-BC (TD3_BC.py, NOT TD3_BC_ct.py).

50k steps, 1 seed, same patient (adult#1), same replay buffer.
Identical protocol to run_path_b_quick.py so the two are directly
comparable.

Weights saved with tag 'TD3_offline_BC_weights' (the paper baseline tag).
"""
import math
import numpy as np
from utils import get_params, create_env
from TD3_BC import td3_bc   # 论文原版 baseline


# ----- Register simglucose envs -----
prob       = [0.95, 0.1, 0.95, 0.1, 0.95, 0.1]
time_lb    = np.array([5, 9, 10, 14, 16, 20])
time_ub    = np.array([9, 10, 14, 16, 20, 23])
time_mu    = np.array([7, 9.5, 12, 15, 18, 21.5])
time_sigma = np.array([30, 15, 30, 15, 30, 15])
amount_mu    = [50, 15, 70, 15, 90, 30]
amount_sigma = [10, 5, 10, 5, 10, 5]
schedule = [prob, time_lb, time_ub, time_mu, time_sigma, amount_mu, amount_sigma]

try:
    create_env(schedule=schedule)
    print("Environments registered.")
except Exception as e:
    if "re-register" in str(e):
        print("Environments already registered (skipping).")
    else:
        raise


# ----- Patient + params (identical structure to run_path_b_quick.py) -----
patient_params = get_params()["adult#1"]
bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3

params = {
    "state_size": 3,
    "basal_default": bas,
    "target_blood_glucose": 144.0,
    "days": 10,

    "carbohydrate_ratio": patient_params["carbohydrate_ratio"],
    "correction_factor":  patient_params["correction_factor"],
    "kp": patient_params["kp"],
    "ki": patient_params["ki"],
    "kd": patient_params["kd"],

    "training_timesteps": int(5e4),   # QUICK: 50k steps
    "device": "cpu",                  # HANDOFF §8: RTX 5070 Ti needs CPU
    "rnn": None,
    # Note: dt, beta, use_path_b, use_hjb are NOT passed -- TD3_BC.py
    # is the paper baseline and ignores them.
}

print(f"Patient: adult#1   bas={bas:.4f}")
print(f"Mode:    QUICK BASELINE  (steps={params['training_timesteps']:,}, seed=0)")
print(f"Algorithm: paper TD3-BC (discrete Bellman, gamma=0.99)")

agent = td3_bc(init_seed=0, patient_params=patient_params, params=params)
agent.train_model()

print("\n===== BASELINE TRAINING DONE =====")
print("Saved weights tag: TD3_offline_BC_weights")
print("Run compare_quick.py to evaluate baseline vs Path B")
