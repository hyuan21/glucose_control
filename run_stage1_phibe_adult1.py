"""
Stage 1 PhiBE training on the corrected Emerson-style adult#1 replay.

This script intentionally does not regenerate data. It assumes
Replays/Adult#1-1e5.txt was generated with the paper protocol:
PID-only data, OU basal noise, and 10% bolus noise.
"""

import argparse
import math
import os

import numpy as np

from TD3_BC_phibe import td3_bc_phibe
from utils import create_env, get_params


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


def make_params(patient_params, args):
    bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3
    dt = 3.0
    beta = -math.log(0.99) / dt
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
        "training_timesteps": int(args.training_timesteps),
        "num_train_steps": int(args.num_train_steps),
        "device": args.device,
        "rnn": None,
        "dt": dt,
        "beta": beta,
        "normalize_reward": args.normalize_reward,
        "critic_activation": args.critic_activation,
        "phibe_mode": args.phibe_mode,
        "lambda_phibe": float(args.lambda_phibe),
        "alpha": float(args.alpha),
        "use_safe_phibe": args.use_safe_phibe,
        "save_tag": args.save_tag,
        "current_glucose_index": 8,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="adult#1")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--training-timesteps", type=int, default=int(1e5))
    parser.add_argument("--num-train-steps", type=int, default=int(1e5))
    parser.add_argument(
        "--phibe-mode",
        default="first_order",
        choices=["first_order", "second_order", "full_second_order", "none"],
    )
    parser.add_argument("--lambda-phibe", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--critic-activation", default="softplus")
    parser.add_argument("--normalize-reward", action="store_true", default=True)
    parser.add_argument("--no-normalize-reward", dest="normalize_reward", action="store_false")
    parser.add_argument("--use-safe-phibe", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-tag", default="phibe_stage1_first_order_lam1e-3_norm")
    args = parser.parse_args()

    register_envs()
    patient_params = get_params()[args.patient]
    replay_path = f"./Replays/{patient_params['replay_name']}.txt"
    if not os.path.exists(replay_path):
        raise FileNotFoundError(
            f"Missing {replay_path}. Run run_emerson2023_td3bc_reproduction.py first."
        )

    params = make_params(patient_params, args)
    print(f"Patient: {args.patient}")
    print(f"Train seed: {args.train_seed}")
    print(f"Replay: {replay_path}")
    print(f"Replay capacity: {args.training_timesteps:,}")
    print(f"Train updates: {args.num_train_steps:,}")
    print(
        "PhiBE: "
        f"mode={args.phibe_mode} lambda={args.lambda_phibe} "
        f"activation={args.critic_activation} normalize_reward={args.normalize_reward} "
        f"alpha={args.alpha}"
    )
    print(f"Save tag: TD3_offline_BC_{args.save_tag}_weights")

    agent = td3_bc_phibe(
        init_seed=args.train_seed,
        patient_params=patient_params,
        params=params,
    )
    agent.train_model()

    print("\n===== STAGE 1 PHIBE TRAINING DONE =====")
    print(f"Saved tag: TD3_offline_BC_{args.save_tag}_weights")


if __name__ == "__main__":
    main()
