"""
Side-by-side QUICK comparison: paper TD3-BC baseline vs Path B vs PID.

Loads each model from the flat save_model() path, runs a 4800-step rollout
(= 10 days) under TEST_SEED=0 (paper convention: differs from training
seed), and prints TIR / TBR / TAR / Magni side-by-side.

The PID rollout is produced by the same test_algorithm call (set
pid_run=True), so it uses the same meal schedule and the same patient.
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
from TD3_BC import td3_bc
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
except Exception as e:
    if "re-register" not in str(e):
        raise


# ----- Patient + params -----
patient_params = get_params()["adult#1"]
bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3

DT = 3.0
BETA = -math.log(0.99) / 3.0
TRAIN_SEED = 0
TEST_SEED = 0
TEST_TIMESTEPS = 4800

base_params = {
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
}


def build_and_load(label):
    """Build the agent matching `label` and load its trained weights."""
    if label == "baseline":
        agent_cls = td3_bc
        params = dict(base_params)
        tag = "TD3_offline_BC_weights"
    elif label == "path_b":
        agent_cls = td3_bc_ct
        params = dict(base_params)
        params["use_path_b"] = True
        params["use_hjb"] = False
        # Critic architecture MUST match the one that saved the weights,
        # or state_dict keys (theta1/theta2 vs l1.weight_orig/_u/_v vs
        # l1.weight) will mismatch on load. Update these when switching
        # the training-side critic_type / spec_norm.
        params["critic_type"] = "linear_basis"
        params["use_spectral_norm"] = False
        params["normalize_reward"] = True
        tag = "TD3_offline_BC_ct_path_b_weights"
    else:
        raise ValueError(label)

    agent = agent_cls(init_seed=TRAIN_SEED, patient_params=patient_params, params=params)
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

    suffix = patient_params["replay_name"].split("-")[-1]
    prefix = f"./Models/{patient_params['env_name']}{TRAIN_SEED}{tag}"
    actor_path = prefix + "_actor" + suffix
    critic_path = prefix + "_critic" + suffix
    if not os.path.exists(actor_path):
        raise FileNotFoundError(
            f"Weights not found for label='{label}': {actor_path}\n"
            f"Run the matching training script first."
        )
    print(f"  Loading {label}: {actor_path}")
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))
    agent.actor_target = copy.deepcopy(agent.actor); agent.actor.eval()
    agent.critic_target = copy.deepcopy(agent.critic); agent.critic.eval()
    return agent


def rollout(agent, with_pid):
    """Run a 4800-step rollout under TEST_SEED. Returns (rl_bg, pid_bg)."""
    test_env = gym.make(patient_params["env_name"])
    rl_r, rl_bg, rl_a, rl_ins, rl_m, pid_r, pid_bg, pid_a = test_algorithm(
        env=test_env, agent_action=agent.select_action,
        seed=TEST_SEED, max_timesteps=TEST_TIMESTEPS,
        sequence_length=agent.sequence_length,
        data_processing=agent.data_processing,
        pid_run=with_pid, params=agent.params,
    )
    return rl_bg, pid_bg


def compute_metrics(bg_trace):
    bg = np.array(bg_trace).flatten()
    bg = bg[~np.isnan(bg)]
    if len(bg) == 0:
        return dict(TIR=float("nan"), TBR=float("nan"), TAR=float("nan"),
                    magni=float("nan"), mean=float("nan"), std=float("nan"))
    in_range  = ((bg >= 70) & (bg <= 180)).mean() * 100
    below_70  = (bg < 70).mean() * 100
    above_180 = (bg > 180).mean() * 100
    f_bg = 3.5506 * (np.log(np.maximum(bg, 1.0)) ** 0.8353 - 3.7932)
    magni = (10 * f_bg ** 2).mean()
    return dict(TIR=in_range, TBR=below_70, TAR=above_180,
                magni=magni, mean=bg.mean(), std=bg.std())


# ----- Run baseline -----
print("===== Loading + rolling out BASELINE =====")
agent_b = build_and_load("baseline")
print(f"  Rolling out {TEST_TIMESTEPS} steps (TEST_SEED={TEST_SEED})...")
bg_baseline, bg_pid = rollout(agent_b, with_pid=True)

# ----- Run Path B -----
print("\n===== Loading + rolling out PATH B =====")
agent_p = build_and_load("path_b")
print(f"  Rolling out {TEST_TIMESTEPS} steps (TEST_SEED={TEST_SEED})...")
bg_pathb, _ = rollout(agent_p, with_pid=False)

# ----- Metrics -----
m_pid      = compute_metrics(bg_pid)
m_baseline = compute_metrics(bg_baseline)
m_pathb    = compute_metrics(bg_pathb)

print("\n" + "=" * 64)
print(f"{'Algorithm':<12}{'TIR':>8}{'TBR':>8}{'TAR':>8}{'Magni':>10}{'BG mean':>10}{'BG std':>8}")
print("-" * 64)
for label, m in [("PID", m_pid), ("Baseline", m_baseline), ("Path B", m_pathb)]:
    print(f"{label:<12}{m['TIR']:>7.2f}%{m['TBR']:>7.2f}%{m['TAR']:>7.2f}%"
          f"{m['magni']:>10.2f}{m['mean']:>10.1f}{m['std']:>8.1f}")
print("=" * 64)

print("\nReference (HANDOFF §6, FULL 100k x 3 seed):")
print("  PID            : TIR 59.50%")
print("  Baseline / A v2: TIR 67.48% +/- 3.16%")
