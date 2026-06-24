"""
Path B + reward normalization + LINEAR BASIS critic.

This is the most aligned-with-paper variant. Critic becomes
Q(s, a) = Phi(s, a)^T theta with fixed handcrafted basis Phi (39 features
covering constant, linear, quadratic, and cross terms over the 11-dim
state and 1-dim action). Theta is the only trainable parameter.

WHY this matters research-wise:
- Yuhua's PhiBE-Q paper §3.2 (Theorem 3.2, 3.3) proves convergence under
  this exact function class.
- Lemma 3.1's Lipschitz bound becomes automatic — Phi is fixed, |Phi|,
  |∇_s Phi|, |∇_s^2 Phi| are all bounded, so |∇_s Q| = |(∇_s Phi)^T theta|
  is bounded whenever |theta| is bounded.
- This is the cleanest extension of paper's §4.1 LQR experiment to
  glucose control: same theory, harder dynamics + non-quadratic reward.

We KEEP reward normalization (it helped baseline jump 63 -> 74%, so it's
a separate improvement worth preserving) and we DROP spectral_norm and
input-grad clip (they were treating the symptom; now we treat the cause
by the function class).

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

    # ===== KEY KNOBS =====
    "critic_type": "linear_basis",
    # lambda_mode='fixed_by_reward_std' uses buffer reward_std (stable)
    # instead of Q.abs().mean (which swings under linear basis). With
    # normalize_reward=True effective_std=1, so lmbda=alpha=2.5 — the
    # D4RL-regime balance the paper's alpha=2.5 was tuned for.
    "lambda_mode": "fixed_by_reward_std",

    # The flags below are irrelevant for linear_basis critic (no NN
    # weights to spectral-norm, ∇_s Q is naturally bounded). Setting them
    # off keeps the code path clean.
    "use_spectral_norm": False,
    "path_b_grad_clip": 1e6,    # effectively off
}

print(f"Patient: adult#1   bas={bas:.4f}")
print(f"Mode:    QUICK PATH B + norm + LINEAR BASIS critic  (steps=50,000, seed=0)")
print(f"CT params: dt={DT} min, beta={BETA:.6e} /min")
print(f"Critic type: {params['critic_type']}")
print()

agent = td3_bc_ct(init_seed=0, patient_params=patient_params, params=params)
agent.train_model()

print("\n===== TRAINING DONE =====")
print("Saved weights tag: ct_path_b (overwrites previous Path B weights)")
print("Run compare_quick.py to evaluate.")
