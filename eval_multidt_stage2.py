"""
Multi-delta-t evaluation for the current Adult#1 TD3-BC vs PhiBE result.

This follows the professor's continuous-time testing request:
evaluate the same trained policy at CGM sampling intervals Δt = 1, 3, 5 min.
No retraining is performed. Each rollout covers the same real duration.
"""

import argparse
import copy
import csv
import os
import pickle
from collections import deque

import gym
import numpy as np
import torch

from TD3_BC import td3_bc
from TD3_BC_phibe import td3_bc_phibe
from utils import (
    create_env,
    get_params,
    set_env_sample_time,
    test_algorithm,
    unpackage_replay,
)


SCHEDULE = [
    [0.95, 0.1, 0.95, 0.1, 0.95, 0.1],
    np.array([5, 9, 10, 14, 16, 20]),
    np.array([9, 10, 14, 16, 20, 23]),
    np.array([7, 9.5, 12, 15, 18, 21.5]),
    np.array([30, 15, 30, 15, 30, 15]),
    [50, 15, 70, 15, 90, 30],
    [10, 5, 10, 5, 10, 5],
]

SENSOR_BY_DT = {
    1.0: "Navigator",
    3.0: "Dexcom",
    5.0: "GuardianRT",
}


def register_envs():
    try:
        create_env(schedule=SCHEDULE)
    except Exception as exc:
        if "re-register" not in str(exc):
            raise


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def base_params(patient_params, training_timesteps, normalize_reward=False):
    bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3
    return {
        "state_size": 3,
        "basal_default": bas,
        "target_blood_glucose": 144.0,
        "days": 10,
        "carbohydrate_ratio": patient_params["carbohydrate_ratio"],
        "correction_factor": patient_params["correction_factor"],
        "kp": patient_params["kp"],
        "ki": patient_params["ki"],
        "kd": patient_params["kd"],
        "training_timesteps": int(training_timesteps),
        "device": "cpu",
        "rnn": None,
        "normalize_reward": normalize_reward,
    }


def prepare_agent(agent, patient_params, training_timesteps):
    agent.memory = deque(maxlen=training_timesteps)
    with open(f"./Replays/{patient_params['replay_name']}.txt", "rb") as file:
        trajectories = pickle.load(file)

    (
        agent.memory,
        agent.state_mean,
        agent.state_std,
        agent.action_mean,
        agent.action_std,
        _,
        _,
    ) = unpackage_replay(
        trajectories=trajectories,
        empty_replay=agent.memory,
        data_processing=agent.data_processing,
        sequence_length=agent.sequence_length,
    )
    agent.action_std = 1.75 * agent.bas * 0.25 / (agent.action_std / agent.bas)
    agent.params["state_mean"], agent.params["state_std"] = agent.state_mean, agent.state_std
    agent.params["action_mean"], agent.params["action_std"] = agent.action_mean, agent.action_std
    agent.max_action = float(((agent.bas * 3.0) - agent.action_mean) / agent.action_std)
    agent.init_model()


def load_weights(agent, actor_path, critic_path):
    print(f"Loading actor: {actor_path}")
    agent.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    agent.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
    agent.actor_target = copy.deepcopy(agent.actor)
    agent.critic_target = copy.deepcopy(agent.critic)
    agent.actor.eval()
    agent.critic.eval()


def build_baseline(patient_params, train_seed, training_timesteps):
    params = base_params(patient_params, training_timesteps, normalize_reward=False)
    agent = td3_bc(init_seed=train_seed, patient_params=patient_params, params=params)
    prepare_agent(agent, patient_params, training_timesteps)

    folder_actor = f"./Models/{patient_params['folder_name']}/Seed{train_seed}/TD3_offline_BC_weights_actor"
    folder_critic = f"./Models/{patient_params['folder_name']}/Seed{train_seed}/TD3_offline_BC_weights_critic"
    if os.path.exists(folder_actor) and os.path.exists(folder_critic):
        load_weights(agent, folder_actor, folder_critic)
        return agent

    suffix = patient_params["replay_name"].split("-")[-1]
    prefix = f"./Models/{patient_params['env_name']}{train_seed}TD3_offline_BC_weights"
    load_weights(agent, prefix + "_actor" + suffix, prefix + "_critic" + suffix)
    return agent


def build_phibe(patient_params, train_seed, training_timesteps, tag, phibe_mode, lambda_phibe, alpha):
    params = base_params(patient_params, training_timesteps, normalize_reward=True)
    params.update(
        {
            "dt": 3.0,
            "beta": -np.log(0.99) / 3.0,
            "critic_activation": "softplus",
            "phibe_mode": phibe_mode,
            "lambda_phibe": lambda_phibe,
            "alpha": alpha,
            "save_tag": tag,
            "current_glucose_index": 8,
        }
    )
    agent = td3_bc_phibe(init_seed=train_seed, patient_params=patient_params, params=params)
    prepare_agent(agent, patient_params, training_timesteps)
    suffix = patient_params["replay_name"].split("-")[-1]
    prefix = f"./Models/{patient_params['env_name']}{train_seed}TD3_offline_BC_{tag}_weights"
    load_weights(agent, prefix + "_actor" + suffix, prefix + "_critic" + suffix)
    return agent


def compute_metrics(bg_values, reward, actions=None, insulin=None, max_steps=None):
    bg = np.asarray(bg_values, dtype=np.float64)
    action_arr = np.asarray(actions if actions is not None else [], dtype=np.float64)
    insulin_arr = np.asarray(insulin if insulin is not None else [], dtype=np.float64)
    n = len(bg)
    early = bool(max_steps is not None and n < 0.99 * max_steps)
    coverage = float(100.0 * n / max_steps) if max_steps else 100.0
    f_bg = 3.5506 * (np.log(np.maximum(bg, 1.0)) ** 0.8353 - 3.7932)

    row = {
        "reward_sum": float(reward),
        "tir": float(np.mean((bg >= 70) & (bg <= 180)) * 100.0),
        "tbr": float(np.mean(bg < 70) * 100.0),
        "tar": float(np.mean(bg > 180) * 100.0),
        "cv": float(np.std(bg) / np.mean(bg) * 100.0),
        "bg_mean": float(np.mean(bg)),
        "bg_std": float(np.std(bg)),
        "magni": float(np.mean(10 * f_bg ** 2)),
        "n_steps": int(n),
        "expected_steps": int(max_steps) if max_steps else int(n),
        "coverage": coverage,
        "early_termination": early,
    }
    if len(action_arr) > 0:
        row.update(
            {
                "action_mean": float(np.mean(action_arr)),
                "action_std": float(np.std(action_arr)),
                "action_min": float(np.min(action_arr)),
                "action_max": float(np.max(action_arr)),
            }
        )
    if len(insulin_arr) > 0:
        row.update(
            {
                "insulin_mean": float(np.mean(insulin_arr)),
                "insulin_std": float(np.std(insulin_arr)),
                "insulin_min": float(np.min(insulin_arr)),
                "insulin_max": float(np.max(insulin_arr)),
            }
        )
    return row


def rollout(agent, patient_params, test_seed, dt, test_days, include_pid):
    env = gym.make(patient_params["env_name"])
    actual_dt = set_env_sample_time(env, dt)
    max_timesteps = int(round(test_days * 24 * 60 / actual_dt))

    (
        rl_reward,
        rl_bg,
        rl_action,
        rl_insulin,
        _,
        pid_reward,
        pid_bg,
        pid_action,
    ) = test_algorithm(
        env=env,
        agent_action=agent.select_action,
        seed=test_seed,
        max_timesteps=max_timesteps,
        sequence_length=agent.sequence_length,
        data_processing=agent.data_processing,
        pid_run=(not include_pid),
        params=agent.params,
        sample_time=actual_dt,
    )

    rl_row = compute_metrics(rl_bg, rl_reward, rl_action, rl_insulin, max_timesteps)
    pid_row = None
    if include_pid:
        pid_row = compute_metrics(pid_bg, pid_reward, pid_action, pid_action, max_timesteps)
    return rl_row, pid_row, actual_dt, max_timesteps


def write_rows(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "patient",
        "algorithm",
        "train_seed",
        "test_seed",
        "dt_requested",
        "dt_actual",
        "sensor",
        "test_days",
        "tir",
        "tbr",
        "tar",
        "cv",
        "bg_mean",
        "bg_std",
        "magni",
        "action_mean",
        "coverage",
        "early_termination",
    ]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved raw rows: {output_path}")


def summarize(rows):
    groups = []
    for row in rows:
        key = (row["algorithm"], row["dt_actual"])
        if key not in groups:
            groups.append(key)

    summary = []
    for algorithm, dt_actual in groups:
        subset = [row for row in rows if row["algorithm"] == algorithm and row["dt_actual"] == dt_actual]
        out = {"algorithm": algorithm, "dt_actual": dt_actual, "n": len(subset)}
        for key in ["tir", "tbr", "tar", "cv", "bg_mean", "magni", "action_mean", "coverage"]:
            vals = [row[key] for row in subset if key in row and row[key] != ""]
            if vals:
                out[key] = float(np.mean(vals))
                out[key + "_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out["early_count"] = sum(1 for row in subset if row["early_termination"])
        summary.append(out)
    return summary


def write_summary(summary, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = sorted({key for row in summary for key in row.keys()})
    preferred = ["algorithm", "dt_actual", "n", "tir", "tbr", "tar", "cv", "bg_mean", "magni", "action_mean", "early_count"]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary: {output_path}")


def print_summary(summary):
    print("\n" + "=" * 104)
    print(
        f"{'Algorithm':<18}{'dt':>6}{'N':>4}{'TIR':>10}{'TBR':>10}"
        f"{'TAR':>10}{'CV':>9}{'BG mean':>10}{'Action':>10}{'Early':>8}"
    )
    print("-" * 104)
    for row in summary:
        print(
            f"{row['algorithm']:<18}{row['dt_actual']:>6.1f}{row['n']:>4}"
            f"{row.get('tir', float('nan')):>9.2f}%"
            f"{row.get('tbr', float('nan')):>9.2f}%"
            f"{row.get('tar', float('nan')):>9.2f}%"
            f"{row.get('cv', float('nan')):>8.2f}%"
            f"{row.get('bg_mean', float('nan')):>10.1f}"
            f"{row.get('action_mean', float('nan')):>10.4f}"
            f"{row.get('early_count', 0):>8}"
        )
    print("=" * 104)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="adult#1")
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--test-seeds", default="0,1,2")
    parser.add_argument("--dts", default="1,3,5")
    parser.add_argument("--test-days", type=float, default=10.0)
    parser.add_argument("--training-timesteps", type=int, default=int(1e5))
    parser.add_argument("--phibe-tag", default="phibe_stage2_first_order_lam1e-4_norm_alpha1p75")
    parser.add_argument(
        "--phibe-mode",
        default="first_order",
        choices=["first_order", "second_order", "full_second_order", "none"],
    )
    parser.add_argument("--lambda-phibe", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=1.75)
    parser.add_argument("--output", default="./Results/stage2_multidt_adult1_raw.csv")
    parser.add_argument("--summary-output", default="./Results/stage2_multidt_adult1_summary.csv")
    args = parser.parse_args()

    register_envs()
    patient_params = get_params()[args.patient]
    train_seeds = parse_int_list(args.train_seeds)
    test_seeds = parse_int_list(args.test_seeds)
    dts = parse_float_list(args.dts)

    print("Multi-delta-t evaluation")
    print(f"Patient: {args.patient}")
    print(f"Train seeds: {train_seeds}")
    print(f"Test seeds: {test_seeds}")
    print(f"Delta-t grid: {dts}")
    print(f"Real duration per rollout: {args.test_days} days")
    print("No retraining is performed.\n")

    phibe_label = f"PhiBE-alpha{args.alpha:g}"
    rows = []
    pid_cache = set()

    for train_seed in train_seeds:
        print(f"Preparing train seed {train_seed}")
        baseline = build_baseline(patient_params, train_seed, args.training_timesteps)
        phibe = build_phibe(
            patient_params,
            train_seed,
            args.training_timesteps,
            args.phibe_tag,
            args.phibe_mode,
            args.lambda_phibe,
            args.alpha,
        )

        for test_seed in test_seeds:
            for dt in dts:
                sensor = SENSOR_BY_DT.get(float(dt), "custom")
                pid_key = (test_seed, float(dt))
                include_pid = pid_key not in pid_cache
                print(
                    f"train_seed={train_seed} test_seed={test_seed} "
                    f"dt={dt:g} min | baseline"
                )
                baseline_row, pid_row, actual_dt, _ = rollout(
                    baseline,
                    patient_params,
                    test_seed,
                    dt,
                    args.test_days,
                    include_pid=include_pid,
                )
                baseline_row.update(
                    {
                        "patient": args.patient,
                        "algorithm": "TD3-BC",
                        "train_seed": train_seed,
                        "test_seed": test_seed,
                        "dt_requested": float(dt),
                        "dt_actual": actual_dt,
                        "sensor": sensor,
                        "test_days": args.test_days,
                    }
                )
                rows.append(baseline_row)

                if include_pid and pid_row is not None:
                    pid_row.update(
                        {
                            "patient": args.patient,
                            "algorithm": "PID",
                            "train_seed": "",
                            "test_seed": test_seed,
                            "dt_requested": float(dt),
                            "dt_actual": actual_dt,
                            "sensor": sensor,
                            "test_days": args.test_days,
                        }
                    )
                    rows.append(pid_row)
                    pid_cache.add(pid_key)

                print(
                    f"train_seed={train_seed} test_seed={test_seed} "
                    f"dt={dt:g} min | PhiBE"
                )
                phibe_row, _, actual_dt, _ = rollout(
                    phibe,
                    patient_params,
                    test_seed,
                    dt,
                    args.test_days,
                    include_pid=False,
                )
                phibe_row.update(
                    {
                        "patient": args.patient,
                        "algorithm": phibe_label,
                        "train_seed": train_seed,
                        "test_seed": test_seed,
                        "dt_requested": float(dt),
                        "dt_actual": actual_dt,
                        "sensor": sensor,
                        "test_days": args.test_days,
                    }
                )
                rows.append(phibe_row)

    write_rows(rows, args.output)
    summary = summarize(rows)
    write_summary(summary, args.summary_output)
    print_summary(summary)


if __name__ == "__main__":
    main()
