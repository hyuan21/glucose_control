"""
Baseline TD3-BC + reward normalization (思路 3) — QUICK 50k × 1 seed.

Exactly the paper baseline (TD3_BC.py) but with normalize_reward=True so
the buffer reward is z-scored before being passed to the critic. This
restores the BC vs Q balance that paper alpha=2.5 was tuned for, which
the raw -Magni_risk reward in glucose control breaks (|Q| ~ 1000,
lambda = alpha/|Q| ~ 0.0025, BC overwhelms Q by ~400x).

Important: reward SHAPE is unchanged — low-BG penalty is still 7x larger
than high-BG penalty, terminal -1e5 cliff is still there. Only the
linear scale changes (raw - mean) / std.

Weights saved to baseline tag — same path as the previous 100k baseline
weights, so this WILL overwrite the seed=0 baseline. Back up first if
you want to keep the old weights:

    Copy-Item Models\\simglucose-adult1-v00TD3_offline_BC_weights_actor1e5 \\
              Models\\simglucose-adult1-v00TD3_offline_BC_weights_actor1e5.100k_raw
    Copy-Item Models\\simglucose-adult1-v00TD3_offline_BC_weights_critic1e5 \\
              Models\\simglucose-adult1-v00TD3_offline_BC_weights_critic1e5.100k_raw
"""
import math
import numpy as np
from utils import get_params, create_env
from TD3_BC import td3_bc


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

    "training_timesteps": int(5e4),
    "device": "cpu",
    "rnn": None,

    # ===== 思路 3 =====
    "normalize_reward": True,
}

print(f"Patient: adult#1   bas={bas:.4f}")
print(f"Mode:    QUICK BASELINE + normalized reward  (steps=50,000, seed=0)")
print()

agent = td3_bc(init_seed=0, patient_params=patient_params, params=params)
agent.train_model()

print("\n===== TRAINING DONE =====")
print("Saved weights tag: TD3_offline_BC_weights (overwrites 100k raw-reward baseline)")
