"""
FULL Path B run — 100k steps, 1 seed.

Identical to run_path_b_quick.py except:
- training_timesteps = 100_000 (matches HANDOFF §6.2 baseline FULL protocol)
- Relies on the gradient-clipping (max_norm=10) added inside TD3_BC_ct.py
  to prevent critic-loss runaway observed in the 50k QUICK run.

Output weights saved with tag 'ct_path_b' -- this WILL OVERWRITE the
50k QUICK weights for seed=0. If you want to keep the QUICK weights for
comparison, back them up first:
    cp Models/...ct_path_b_weights_actor1e5 Models/...ct_path_b_50k_actor1e5
    cp Models/...ct_path_b_weights_critic1e5 Models/...ct_path_b_50k_critic1e5

Expected runtime: 3-5 hours on CPU (50k QUICK was ~1.5-2.5h; 100k ~2x).
"""
import math
import numpy as np
from utils import get_params, create_env
from TD3_BC_ct import td3_bc_ct


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


# ----- Patient + params -----
patient_params = get_params()["adult#1"]
bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3

DT = 3.0
BETA = -math.log(0.99) / 3.0

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

    "training_timesteps": int(1e5),   # FULL: 100k steps
    "device": "cpu",
    "rnn": None,

    "dt": DT,
    "beta": BETA,

    "use_path_b": True,
    "use_hjb": False,
}

print(f"Patient: adult#1   bas={bas:.4f}")
print(f"Mode:    FULL  (steps={params['training_timesteps']:,}, seed=0)")
print(f"CT params: dt={DT} min, beta={BETA:.6e} /min, "
      f"exp(-beta*dt)={math.exp(-BETA*DT):.6f}")
print(f"Path B enabled: use_path_b={params['use_path_b']}")
print(f"Gradient clipping active (max_norm=10) -- see TD3_BC_ct.py")
print()

agent = td3_bc_ct(init_seed=0, patient_params=patient_params, params=params)
agent.train_model()

print("\n===== FULL TRAINING DONE =====")
print("Saved weights tag: ct_path_b")
print("Now run compare_quick.py to evaluate baseline vs Path B-FULL")
