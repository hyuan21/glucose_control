"""
QUICK sanity-check run for Path B.

50k steps, 1 seed, default beta = -ln(0.99)/3, default dt = 3.
Matches the QUICK protocol used for Approach C v1/v2 in HANDOFF §6.3-§6.4.

Path B replaces the critic target with the second-order Taylor expansion
of the professor's continuous-time Q function around (s_t, a_t):

    target = r + exp(-beta*dt) * [
        Q_target(s, a)
        + Delta_s^T grad_s Q_target
        + Delta_a^T grad_a Q_target
        + 1/2 Delta_s^T grad_s^2 Q_target Delta_s
    ]

Outputs are saved to ./Models/ with tag 'ct_path_b'.

To evaluate after training: run eval_path_b.py.
"""
import math
import numpy as np
from utils import get_params, create_env
from TD3_BC_ct import td3_bc_ct


# ----- Register the simglucose gym environments (idempotent) -----
# Same schedule as Comparison_CT.ipynb cell 1.
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


# ----- Patient + env params (adult#1, same as the paper baseline) -----
patient_params = get_params()["adult#1"]
bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3

# ----- Continuous-time hyperparameters -----
DT = 3.0
BETA = -math.log(0.99) / 3.0

# ----- Build the params dict the same way Comparison_CT.ipynb does -----
params = {
    # Environmental
    "state_size": 3,
    "basal_default": bas,
    "target_blood_glucose": 144.0,
    "days": 10,

    # PID and Bolus
    "carbohydrate_ratio": patient_params["carbohydrate_ratio"],
    "correction_factor":  patient_params["correction_factor"],
    "kp": patient_params["kp"],
    "ki": patient_params["ki"],
    "kd": patient_params["kd"],

    # RL
    "training_timesteps": int(5e4),   # QUICK: 50k steps
    "device": "cpu",                  # HANDOFF §8: RTX 5070 Ti needs CPU
    "rnn": None,

    # Continuous-time
    "dt": DT,
    "beta": BETA,

    # ===== ENABLE PATH B =====
    "use_path_b": True,
    "use_hjb": False,   # must be False; Path B and HJB are mutually exclusive
}

print(f"Patient: adult#1   bas={bas:.4f}")
print(f"Mode:    QUICK  (steps={params['training_timesteps']:,}, seed=0)")
print(f"CT params: dt={DT} min, beta={BETA:.6e} /min, "
      f"exp(-beta*dt)={math.exp(-BETA*DT):.6f}")
print(f"Path B enabled: use_path_b={params['use_path_b']}")

agent = td3_bc_ct(init_seed=0, patient_params=patient_params, params=params)
agent.train_model()

print("\n===== TRAINING DONE =====")
print("Saved weights tag: ct_path_b")
print("Now run eval_path_b.py to compute TIR / TBR / Magni risk")
