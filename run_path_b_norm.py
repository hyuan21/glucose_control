"""
Path B + reward normalization (思路 3) — QUICK 50k × 1 seed.

Path B with normalize_reward=True. This combines our existing
Taylor-expanded critic target with z-scored rewards. Expected effects:

- |Q| drops from ~1000 to ~1-10, restoring lambda = alpha/|Q| ~ 0.1-1
  (vs the current ~0.0025).
- ∇_s Q · Δs term should grow much more slowly during training because
  Q itself is bounded — runaway less likely.
- BC vs Q balance returns to D4RL-style regime that alpha=2.5 expects.

Still uses path_b_grad_clip=10 as an extra safety belt, but if the
normalisation theory is correct the clip should rarely actually trigger.

Weights saved to ct_path_b tag (overwrites previous Path B weights).
Back up first if you want to keep the previous Path B weights:

    Copy-Item Models\\simglucose-adult1-v00TD3_offline_BC_ct_path_b_weights_actor1e5 \\
              Models\\simglucose-adult1-v00TD3_offline_BC_ct_path_b_weights_actor1e5.clip10_rawreward
    Copy-Item Models\\simglucose-adult1-v00TD3_offline_BC_ct_path_b_weights_critic1e5 \\
              Models\\simglucose-adult1-v00TD3_offline_BC_ct_path_b_weights_critic1e5.clip10_rawreward
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
    "path_b_grad_clip": 10.0,    # safety belt — may rarely trigger now

    # ===== 思路 3 =====
    "normalize_reward": True,
}

print(f"Patient: adult#1   bas={bas:.4f}")
print(f"Mode:    QUICK PATH B + normalized reward  (steps=50,000, seed=0)")
print(f"CT params: dt={DT} min, beta={BETA:.6e} /min, "
      f"exp(-beta*dt)={math.exp(-BETA*DT):.6f}")
print()

agent = td3_bc_ct(init_seed=0, patient_params=patient_params, params=params)
agent.train_model()

print("\n===== TRAINING DONE =====")
print("Saved weights tag: ct_path_b (overwrites previous Path B weights)")
