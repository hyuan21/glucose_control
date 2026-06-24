"""
Multi-Δt policy evaluation (professor's directive, 2026-06).

WHAT THIS DOES
--------------
Tests an ALREADY-TRAINED policy under several CGM sampling intervals (Δt)
instead of the fixed 3-min grid it was trained on. simglucose's Δt is set
entirely by the CGM sensor, so we swap the sensor on the test env via
`set_env_sample_time` (utils/parameters.py) and re-roll out. No retraining,
no third-party source edits.

Recommended grid (real devices, see RECOMMENDED_SAMPLE_TIMES):
    Δt = 1 min  (Navigator)
    Δt = 3 min  (Dexcom — training anchor)
    Δt = 5 min  (GuardianRT)

KEY DESIGN DECISIONS
--------------------
1. Same REAL duration at every Δt. A run is fixed at TEST_DAYS days, so the
   step budget scales as max_timesteps = TEST_DAYS * 1440 / Δt. Otherwise a
   1-min run would only cover 1/3 the real time of a 3-min run and TIR would
   not be comparable.
2. Policy input interface is untouched. test_algorithm rebuilds the same
   11-D condensed state on the Δt grid (see its docstring), so any TIR change
   is attributable to sampling-rate mismatch — which is exactly what the
   professor wants to expose — not to a wrong-shaped network input.
3. Δt=3 must reproduce the legacy single-rate numbers (sanity / regression).

USAGE
-----
    python eval_multi_dt.py                 # baseline + path_b, Δt in {1,3,5}
    python eval_multi_dt.py --models baseline
    python eval_multi_dt.py --dts 1 2 3 4 5 # (2,4 need custom sensor rows)
"""
import argparse
import math
import os
import copy
import pickle
from collections import deque

import gym
import numpy as np
import torch

from utils import (get_params, create_env, unpackage_replay,
                   set_env_sample_time, RECOMMENDED_SAMPLE_TIMES,
                   SAMPLE_TIME_TO_SENSOR)
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

DT_TRAIN = 3.0
BETA = -math.log(0.99) / 3.0
TRAIN_SEED = 0
TEST_SEED = 0
TEST_DAYS = 10                      # fixed REAL duration at every Δt

base_params = {
    "state_size": 3,
    "basal_default": bas,
    "target_blood_glucose": 144.0,
    "days": TEST_DAYS,
    "carbohydrate_ratio": patient_params["carbohydrate_ratio"],
    "correction_factor":  patient_params["correction_factor"],
    "kp": patient_params["kp"],
    "ki": patient_params["ki"],
    "kd": patient_params["kd"],
    "training_timesteps": int(5e4),
    "device": "cpu",
    "rnn": None,
    "dt": DT_TRAIN,
    "beta": BETA,
}


# Per-model build recipe. The critic architecture MUST match the weights on
# disk or state_dict keys mismatch on load. Keep these in sync with the
# training script that produced each checkpoint.
MODEL_SPECS = {
    "baseline": dict(
        cls=td3_bc, tag="TD3_offline_BC_weights",
        extra={},
    ),
    "path_b": dict(
        cls=td3_bc_ct, tag="TD3_offline_BC_ct_path_b_weights",
        extra={"use_path_b": True, "use_hjb": False,
               "critic_type": "linear_basis", "use_spectral_norm": False,
               "normalize_reward": True},
    ),
}


def build_and_load(label):
    spec = MODEL_SPECS[label]
    params = dict(base_params)
    params.update(spec["extra"])

    agent = spec["cls"](init_seed=TRAIN_SEED, patient_params=patient_params, params=params)
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
    prefix = f"./Models/{patient_params['env_name']}{TRAIN_SEED}{spec['tag']}"
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


def rollout_at_dt(agent, dt, with_pid):
    """Roll out for TEST_DAYS real days at sampling interval `dt`.

    `with_pid=True` also returns the PID reference trace for this dt.
    Returns (rl_bg, pid_bg, actual_dt, max_timesteps).
    """
    test_env = gym.make(patient_params["env_name"])

    # Swap the CGM sensor so the env actually samples every `dt` minutes.
    actual_dt = set_env_sample_time(test_env, dt)

    # Same real duration at every dt -> scale the step budget.
    max_timesteps = int(round(TEST_DAYS * 24 * 60 / actual_dt))

    # test_algorithm internally: `if not pid_run: runs = 2 (RL + PID) else 1`.
    # To ALSO get the PID trace we must pass pid_run=False. So with_pid maps
    # to pid_run = not with_pid.
    rl_r, rl_bg, rl_a, rl_ins, rl_m, pid_r, pid_bg, pid_a = test_algorithm(
        env=test_env, agent_action=agent.select_action,
        seed=TEST_SEED, max_timesteps=max_timesteps,
        sequence_length=agent.sequence_length,
        data_processing=agent.data_processing,
        pid_run=(not with_pid), params=agent.params,
        sample_time=actual_dt,                       # <- the key new arg
    )
    return rl_bg, pid_bg, actual_dt, max_timesteps


def compute_metrics(bg_trace, max_steps=None):
    bg = np.array(bg_trace).flatten()
    bg = bg[~np.isnan(bg)]
    n = len(bg)
    # Did the episode terminate early (patient driven to BG<10 or >1000)?
    # We compare the trace length to the requested step budget.
    early = (max_steps is not None) and (n < 0.99 * max_steps)
    if n == 0:
        return dict(TIR=float("nan"), TBR=float("nan"), TAR=float("nan"),
                    magni=float("nan"), mean=float("nan"), std=float("nan"),
                    n=0, early=True, coverage=0.0)
    in_range  = ((bg >= 70) & (bg <= 180)).mean() * 100
    below_70  = (bg < 70).mean() * 100
    above_180 = (bg > 180).mean() * 100
    f_bg = 3.5506 * (np.log(np.maximum(bg, 1.0)) ** 0.8353 - 3.7932)
    magni = (10 * f_bg ** 2).mean()
    coverage = 100.0 * n / max_steps if max_steps else 100.0
    return dict(TIR=in_range, TBR=below_70, TAR=above_180,
                magni=magni, mean=bg.mean(), std=bg.std(),
                n=n, early=early, coverage=coverage)


def _fmt_row(label, dt, m):
    flag = " *EARLY*" if m["early"] else ""
    return (f"{label:<10}{dt:>7.0f}{m['TIR']:>8.2f}%{m['TBR']:>7.2f}%"
            f"{m['TAR']:>7.2f}%{m['magni']:>9.2f}{m['mean']:>9.1f}"
            f"{m['n']:>7}{m['coverage']:>8.1f}%{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["baseline", "path_b"],
                    choices=list(MODEL_SPECS))
    ap.add_argument("--dts", nargs="+", type=float, default=RECOMMENDED_SAMPLE_TIMES,
                    help="sampling intervals in minutes")
    args = ap.parse_args()

    print("Multi-dt evaluation")
    print(f"  patient = adult#1 | TEST_DAYS = {TEST_DAYS} | TEST_SEED = {TEST_SEED}")
    print(f"  dt grid = {args.dts}  (sensors: "
          f"{[SAMPLE_TIME_TO_SENSOR.get(d, '??') for d in args.dts]})")
    print(f"  models  = {args.models}\n")

    rows = {}
    pid_rows = {}
    budget = {}                      # dt -> max_timesteps

    for label in args.models:
        print(f"===== {label.upper()} =====")
        agent = build_and_load(label)
        for dt in args.dts:
            need_pid = dt not in pid_rows
            bg_rl, bg_pid, actual_dt, steps = rollout_at_dt(agent, dt, with_pid=need_pid)
            budget[dt] = steps
            rows[(label, dt)] = compute_metrics(bg_rl, max_steps=steps)
            if need_pid:
                pid_rows[dt] = compute_metrics(bg_pid, max_steps=steps)
            m = rows[(label, dt)]
            tail = "  (EARLY TERMINATION - trace incomplete)" if m["early"] else ""
            print(f"  dt={dt:>3} min ({steps} steps): TIR {m['TIR']:.2f}%  "
                  f"[{m['n']}/{steps} steps]{tail}")
        print()

    # ---- Results table ----
    print("=" * 82)
    print(f"{'Model':<10}{'dt(min)':>7}{'TIR':>9}{'TBR':>8}{'TAR':>8}"
          f"{'Magni':>9}{'BGmean':>9}{'n':>7}{'cover':>9}")
    print("-" * 82)
    for dt in args.dts:
        print(_fmt_row("PID", dt, pid_rows[dt]))
    for label in args.models:
        for dt in args.dts:
            print(_fmt_row(label, dt, rows[(label, dt)]))
    print("=" * 82)
    print("cover = % of the requested step budget actually simulated; "
          "*EARLY* = patient driven to BG<10 or >1000 (sim terminated).")
    print("WARNING: TIR from an *EARLY* row is computed on a truncated trace "
          "and is NOT comparable to full-length rows.")

    # ---- dt-robustness summary (vs the training anchor dt=3) ----
    if 3.0 in args.dts:
        print("\ndt-robustness (delta-TIR vs dt=3 anchor; full-trace rows only):")
        for label in args.models:
            anchor = rows[(label, 3.0)]
            base_tir = anchor["TIR"]
            anchor_note = " [anchor itself EARLY!]" if anchor["early"] else ""
            parts = []
            for dt in args.dts:
                if dt == 3.0:
                    continue
                m = rows[(label, dt)]
                tag = "*" if m["early"] else ""
                parts.append(f"dt={dt:.0f}: {m['TIR'] - base_tir:+.2f}%{tag}")
            print(f"  {label:<10} anchor TIR {base_tir:.2f}%{anchor_note}  | "
                  + ", ".join(parts))
        print("  (* = row was an early-termination trace; treat with caution)")

    print("\nReference (single-rate dt=3, prior sessions):")
    print("  PID 59.50% | Baseline ~63-67% (raw) / 74.29% (reward-norm) | "
          "Path B ceiling ~57%")


if __name__ == "__main__":
    main()
