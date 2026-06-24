"""
Path B + reward normalization + spectral-norm critic.

This is the principled attack on the Zhu 2025 open problem:
PhiBE-Q's Lemma 3.1 needs the iteration operator H to be Lipschitz,
which holds under linear basis (bounded c_1, c_2, c_3) but is NOT
automatic for an unconstrained DNN critic. We force it as an
architecture constraint via spectral normalization on every Linear
layer of the critic — bounding the network's Lipschitz constant <= 1.

See RESEARCH_POSITION.md §5.1 for the full motivation. The previous
input-gradient-clipping approach (path_b_grad_clip) was treating the
symptom; spectral_norm treats the cause.

Knobs in this run:
- use_spectral_norm = True  -- the key new knob
- normalize_reward = True   -- keep the |Q| ~ 1-10 regime
- path_b_grad_clip = 100    -- effectively off; only catches outliers
                              (spectral_norm should dominate)

QUICK 50k × 1 seed. Overwrites the previous Path B weights.
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

    # ===== Reward normalization (theory 3) =====
    "normalize_reward": True,

    # ===== Spectral-norm critic (RESEARCH_POSITION.md §5.1) =====
    # This is THE key new knob. Bounds Critic's Lipschitz constant
    # so |∇_s Q| stays naturally bounded, removing the need for
    # ad-hoc input-gradient clipping.
    "use_spectral_norm": True,

    # Set the input-grad clip very loose -- spectral_norm should
    # dominate. If clip still fires often, spectral_norm isn't tight
    # enough and we'd consider a tighter architecture constraint.
    "path_b_grad_clip": 100.0,
}

print(f"Patient: adult#1   bas={bas:.4f}")
print(f"Mode:    QUICK PATH B + norm + spectral_norm  (steps=50,000, seed=0)")
print(f"CT params: dt={DT} min, beta={BETA:.6e} /min, "
      f"exp(-beta*dt)={math.exp(-BETA*DT):.6f}")
print(f"Key knobs: use_spectral_norm=True, normalize_reward=True, "
      f"clip=100 (loose)")
print()

agent = td3_bc_ct(init_seed=0, patient_params=patient_params, params=params)
agent.train_model()

print("\n===== TRAINING DONE =====")
print("Saved weights tag: ct_path_b (overwrites previous Path B weights)")
print("Run compare_quick.py to evaluate.")
