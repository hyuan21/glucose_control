"""
Evaluate the QUICK Path B model on the simglucose adult#1 test.

Mirrors the `evaluate_agent` function inside Comparison_CT.ipynb cell 12:
loads the flat-path weights produced by save_model() (NOT the folder/Seed
structure that the legacy test_model() expects), runs a 4800-step rollout,
and prints TIR / TBR / Magni risk plus the BG-trace summary.

Run after run_path_b_quick.py finishes.
"""
import math
import os
import copy
import pickle
from collections import deque

import gym
import numpy as np
import torch

from utils import get_params, create_env, unpackage_replay
from utils.evaluation import test_algorithm
from TD3_BC_ct import td3_bc_ct


# ----- Register the simglucose gym environments -----
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
except Exception as e:
    if "re-register" not in str(e):
        raise


# ----- Patient + params (match run_path_b_quick.py exactly) -----
patient_params = get_params()["adult#1"]
bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3

DT = 3.0
BETA = -math.log(0.99) / 3.0
TRAIN_SEED = 0
TEST_SEED = 0          # paper convention: differs from training seed
TEST_TIMESTEPS = 4800  # = 10 days

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
}


# ----- Build agent, populate normalisation stats from the replay -----
agent = td3_bc_ct(init_seed=TRAIN_SEED, patient_params=patient_params, params=params)
agent.memory = deque(maxlen=agent.memory_size)

replay_path = "./Replays/" + patient_params["replay_name"] + ".txt"
with open(replay_path, "rb") as f:
    trajectories = pickle.load(f)

(agent.memory, agent.state_mean, agent.state_std,
 agent.action_mean, agent.action_std, _, _) = unpackage_replay(
    trajectories=trajectories, empty_replay=agent.memory,
    data_processing=agent.data_processing,
    sequence_length=agent.sequence_length,
)
agent.action_std = 1.75 * agent.bas * 0.25 / (agent.action_std / agent.bas)
agent.params["state_mean"], agent.params["state_std"] = agent.state_mean, agent.state_std
agent.params["action_mean"], agent.params["action_std"] = agent.action_mean, agent.action_std
agent.max_action = float(((agent.bas * 3) - agent.action_mean) / agent.action_std)
agent.init_model()


# ----- Load Path B weights from the flat save_model() path -----
suffix = patient_params["replay_name"].split("-")[-1]
prefix = f"./Models/{patient_params['env_name']}{TRAIN_SEED}TD3_offline_BC_ct_path_b_weights"
print(f"Loading: {prefix}_actor{suffix}")
agent.actor.load_state_dict(torch.load(prefix + "_actor" + suffix))
agent.critic.load_state_dict(torch.load(prefix + "_critic" + suffix))
agent.actor_target = copy.deepcopy(agent.actor); agent.actor.eval()
agent.critic_target = copy.deepcopy(agent.critic); agent.critic.eval()


# ----- Roll out RL agent vs PID -----
test_env = gym.make(patient_params["env_name"])
print(f"\nRolling out {TEST_TIMESTEPS} steps (seed={TEST_SEED})...")

rl_r, rl_bg, rl_a, rl_ins, rl_m, pid_r, pid_bg, pid_a = test_algorithm(
    env=test_env, agent_action=agent.select_action,
    seed=TEST_SEED, max_timesteps=TEST_TIMESTEPS,
    sequence_length=agent.sequence_length,
    data_processing=agent.data_processing,
    pid_run=True, params=agent.params,
)


# ----- Compute paper-style metrics from BG traces -----
def compute_metrics(bg_trace, label):
    bg = np.array(bg_trace).flatten()
    in_range  = ((bg >= 70) & (bg <= 180)).mean() * 100
    below_70  = (bg < 70).mean() * 100
    above_180 = (bg > 180).mean() * 100
    severe_low  = (bg < 54).mean() * 100
    severe_high = (bg > 250).mean() * 100
    # Magni risk index (paper convention)
    f_bg = 3.5506 * (np.log(np.maximum(bg, 1.0)) ** 0.8353 - 3.7932)
    magni = (10 * f_bg ** 2).mean()
    print(f"--- {label} ---")
    print(f"  TIR  (70-180 mg/dL): {in_range:6.2f}%")
    print(f"  TBR  (<70   mg/dL): {below_70:6.2f}%")
    print(f"  TAR  (>180  mg/dL): {above_180:6.2f}%")
    print(f"  Severe hypo (<54): {severe_low:6.2f}%")
    print(f"  Severe hyper(>250):{severe_high:6.2f}%")
    print(f"  Magni risk index : {magni:6.2f}")
    print(f"  BG mean / std    : {bg.mean():.1f} / {bg.std():.1f}")
    print()
    return in_range, below_70, magni


print("\n========== RESULTS ==========")
compute_metrics(pid_bg, "PID")
compute_metrics(rl_bg, "Path B (ct_path_b)")

print("Reference (HANDOFF §6):")
print("  PID                : TIR 59.50%")
print("  Baseline / A v2    : TIR 67.48% (FULL 100k x 3 seed)")
print("  Approach C v1 50k  : TIR 57.92%")
print("  Approach C v2 50k  : TIR 64.44%")
print("  Approach C v3 100k : TIR 63.67%")
