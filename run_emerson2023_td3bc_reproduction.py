"""
Reproduce the Emerson et al. 2023 TD3-BC baseline protocol.

Default mode is a small pilot on adult#1 so the pipeline can be verified on a
laptop. Use --patients all --train-seeds 0,1,2 for the paper-scale run.
"""

import argparse
import csv
import os
import pickle
import shutil
from collections import deque

import gym
import numpy as np

from TD3_BC import td3_bc
from utils import (
    create_env,
    fill_replay_split,
    get_params,
    test_algorithm,
    unpackage_replay,
)
from utils.general import calculate_risk


SCHEDULE = [
    [0.95, 0.1, 0.95, 0.1, 0.95, 0.1],
    np.array([5, 9, 10, 14, 16, 20]),
    np.array([9, 10, 14, 16, 20, 23]),
    np.array([7, 9.5, 12, 15, 18, 21.5]),
    np.array([30, 15, 30, 15, 30, 15]),
    [50, 15, 70, 15, 90, 30],
    [10, 5, 10, 5, 10, 5],
]


def register_envs():
    try:
        create_env(schedule=SCHEDULE)
        print("Environments registered.")
    except Exception as exc:
        if "re-register" in str(exc):
            print("Environments already registered.")
        else:
            raise


def patient_keys(selection):
    if selection == "all":
        return (
            [f"child#{i}" for i in range(1, 11)]
            + [f"adolescent#{i}" for i in range(1, 11)]
            + [f"adult#{i}" for i in range(1, 11)]
        )
    return [item.strip() for item in selection.split(",") if item.strip()]


def make_params(patient_params, training_timesteps, device, normalize_reward):
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
        "device": device,
        "rnn": None,
        "normalize_reward": normalize_reward,
    }


def ensure_replay(patient_params, params, replay_length, replay_seed, overwrite):
    os.makedirs("./Replays", exist_ok=True)
    replay_path = f"./Replays/{patient_params['replay_name']}.txt"
    if os.path.exists(replay_path) and not overwrite:
        print(f"Replay exists: {replay_path}")
        return

    print(
        "Generating replay "
        f"{replay_path} | length={replay_length:,} | PID-only | "
        f"OU noise=True | bolus_noise=0.1 | seed={replay_seed}"
    )
    env = gym.make(patient_params["env_name"])
    fill_replay_split(
        env=env,
        replay_name=patient_params["replay_name"],
        data_split=0.0,
        replay_length=int(replay_length),
        noise=True,
        bolus_noise=0.1,
        seed=int(replay_seed),
        params=params,
    )


def copy_weights_to_test_location(patient_params, train_seed):
    env_name = patient_params["env_name"]
    suffix = patient_params["replay_name"].split("-")[-1]
    src_actor = f"./Models/{env_name}{train_seed}TD3_offline_BC_weights_actor{suffix}"
    src_critic = f"./Models/{env_name}{train_seed}TD3_offline_BC_weights_critic{suffix}"

    target_dir = f"./Models/{patient_params['folder_name']}/Seed{train_seed}"
    os.makedirs(target_dir, exist_ok=True)
    dst_actor = f"{target_dir}/TD3_offline_BC_weights_actor"
    dst_critic = f"{target_dir}/TD3_offline_BC_weights_critic"

    shutil.copyfile(src_actor, dst_actor)
    shutil.copyfile(src_critic, dst_critic)


def prepare_agent_for_eval(patient_params, params, train_seed):
    agent = td3_bc(init_seed=train_seed, patient_params=patient_params, params=dict(params))
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
        empty_replay=deque(maxlen=params["training_timesteps"]),
        data_processing=agent.data_processing,
        sequence_length=agent.sequence_length,
    )
    agent.action_std = 1.75 * agent.bas * 0.25 / (agent.action_std / agent.bas)
    agent.params["state_mean"], agent.params["state_std"] = agent.state_mean, agent.state_std
    agent.params["action_mean"], agent.params["action_std"] = agent.action_mean, agent.action_std
    agent.max_action = float(((agent.bas * 3.0) - agent.action_mean) / agent.action_std)
    agent.init_model()
    agent.load_model(
        f"./Models/{patient_params['folder_name']}/Seed{train_seed}/TD3_offline_BC_weights"
    )
    return agent


def metrics(bg_values, total_reward):
    bg = np.asarray(bg_values, dtype=np.float64)
    if len(bg) == 0:
        return {
            "reward": float(total_reward),
            "tir": 0.0,
            "tbr": 0.0,
            "tar": 0.0,
            "cv": 0.0,
            "bg_mean": 0.0,
            "bg_std": 0.0,
            "magni": 0.0,
            "failure": 1,
        }
    return {
        "reward": float(total_reward),
        "tir": float(np.mean((bg >= 70) & (bg <= 180)) * 100.0),
        "tbr": float(np.mean(bg < 70) * 100.0),
        "tar": float(np.mean(bg > 180) * 100.0),
        "cv": float(np.std(bg) / np.mean(bg) * 100.0),
        "bg_mean": float(np.mean(bg)),
        "bg_std": float(np.std(bg)),
        "magni": float(np.mean(calculate_risk(bg))),
        "failure": int(len(bg) < 4800),
    }


def evaluate_patient(patient_key, patient_params, params, train_seed, test_seeds):
    agent = prepare_agent_for_eval(patient_params, params, train_seed)
    env = gym.make(patient_params["env_name"])
    rows = []

    for test_seed in test_seeds:
        (
            rl_reward,
            rl_bg,
            _,
            _,
            _,
            pid_reward,
            pid_bg,
            _,
        ) = test_algorithm(
            env=env,
            agent_action=agent.select_action,
            seed=int(test_seed),
            max_timesteps=4800,
            sequence_length=agent.sequence_length,
            data_processing=agent.data_processing,
            pid_run=False,
            params=agent.params,
        )

        rl_metrics = metrics(rl_bg, rl_reward)
        pid_metrics = metrics(pid_bg, pid_reward)
        print(
            f"{patient_key} train_seed={train_seed} test_seed={test_seed} "
            f"TD3-BC TIR={rl_metrics['tir']:.2f}% "
            f"PID TIR={pid_metrics['tir']:.2f}%"
        )
        rows.append(
            {
                "patient": patient_key,
                "age_group": patient_key.split("#")[0],
                "train_seed": train_seed,
                "test_seed": int(test_seed),
                **{f"td3bc_{key}": value for key, value in rl_metrics.items()},
                **{f"pid_{key}": value for key, value in pid_metrics.items()},
            }
        )
    return rows


def write_results(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results: {output_path}")


def print_summary(rows):
    for prefix in ("td3bc", "pid"):
        tir = np.array([row[f"{prefix}_tir"] for row in rows], dtype=np.float64)
        tbr = np.array([row[f"{prefix}_tbr"] for row in rows], dtype=np.float64)
        reward = np.array([row[f"{prefix}_reward"] for row in rows], dtype=np.float64)
        cv = np.array([row[f"{prefix}_cv"] for row in rows], dtype=np.float64)
        print(
            f"{prefix.upper()} mean over {len(rows)} evals | "
            f"Reward={reward.mean():.0f} | TIR={tir.mean():.2f}% | "
            f"TBR={tbr.mean():.2f}% | CV={cv.mean():.2f}%"
        )


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", default="adult#1")
    parser.add_argument("--train-seeds", default="0")
    parser.add_argument("--test-seeds", default="0,1,2")
    parser.add_argument("--replay-length", type=int, default=int(1e5))
    parser.add_argument("--training-timesteps", type=int, default=int(1e5))
    parser.add_argument("--replay-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--normalize-reward", action="store_true")
    parser.add_argument("--overwrite-replay", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--output", default="./Results/emerson2023_td3bc_reproduction.csv")
    args = parser.parse_args()

    register_envs()
    params_by_patient = get_params()
    all_rows = []

    for key in patient_keys(args.patients):
        if key not in params_by_patient:
            raise KeyError(f"Unknown patient key: {key}")

        patient_params = params_by_patient[key]
        params = make_params(
            patient_params=patient_params,
            training_timesteps=args.training_timesteps,
            device=args.device,
            normalize_reward=args.normalize_reward,
        )

        ensure_replay(
            patient_params=patient_params,
            params=params,
            replay_length=args.replay_length,
            replay_seed=args.replay_seed,
            overwrite=args.overwrite_replay,
        )

        for train_seed in parse_int_list(args.train_seeds):
            if not args.skip_train:
                print(
                    f"Training TD3-BC | patient={key} | seed={train_seed} | "
                    f"steps={args.training_timesteps:,}"
                )
                agent = td3_bc(
                    init_seed=int(train_seed),
                    patient_params=patient_params,
                    params=dict(params),
                )
                agent.train_model()
                copy_weights_to_test_location(patient_params, int(train_seed))

            all_rows.extend(
                evaluate_patient(
                    patient_key=key,
                    patient_params=patient_params,
                    params=params,
                    train_seed=int(train_seed),
                    test_seeds=parse_int_list(args.test_seeds),
                )
            )

    write_results(all_rows, args.output)
    print_summary(all_rows)


if __name__ == "__main__":
    main()
