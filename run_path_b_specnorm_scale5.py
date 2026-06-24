"""
Path B + normalize + spectral_norm with sn_scale=5.

Previous spec_norm run (sn_scale=1, default) gave TIR 56.96% — training
fully stable but Critic's Lipschitz constraint too tight, capping
expressive power below baseline (74.29%).

This run relaxes the constraint: each spec-norm-wrapped Linear's
effective Lipschitz becomes 5 instead of 1. Critic can express
sharper Q changes (e.g. between hypo and in-range BG regions) while
still staying Lipschitz-bounded.

Standard hyperparameters elsewhere in spec-norm-regularised SAC use
sn_scale in [3, 10]; 5 is a middle starting point.

QUICK 50k × 1 seed. Overwrites previous Path B weights.
"""
import math
import numpy as np
from utils import get_params, create_env
from TD3_BC_ct import td3_bc_ct


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

    "training_timesteps": int(5e4),
    "device": "cpu",
    "rnn": None,

    "dt": DT,
    "beta": BETA,

    "use_path_b": True,
    "use_hjb": False,

    "normalize_reward": True,

    "use_spectral_norm": True,
    "sn_scale": 5.0,             # <-- the key new knob

    "path_b_grad_clip": 100.0,   # very loose, spectral_norm should dominate
}

print(f"Patient: adult#1   bas={bas:.4f}")
print(f"Mode:    QUICK PATH B + norm + spec_norm(sn_scale=5)  (steps=50,000, seed=0)")
print(f"CT params: dt={DT} min, beta={BETA:.6e} /min, "
      f"exp(-beta*dt)={math.exp(-BETA*DT):.6f}")
print()

agent = td3_bc_ct(init_seed=0, patient_params=patient_params, params=params)
agent.train_model()

print("\n===== TRAINING DONE =====")
print("Saved weights tag: ct_path_b (overwrites previous Path B weights)")
print("Run compare_quick.py to evaluate.")
