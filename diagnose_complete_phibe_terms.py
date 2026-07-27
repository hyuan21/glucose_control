#!/usr/bin/env python3
"""Diagnose loss scales for the current Complete PhiBE-TD3-BC critic."""

import argparse
import copy
import csv
import math
import os
import pickle
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from TD3_BC_phibe import td3_bc_phibe
from utils import create_env, get_batch, get_params, unpackage_replay


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
    except Exception as exc:
        if "re-register" not in str(exc):
            raise


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def make_params(patient_params, args):
    bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3
    dt = 3.0
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
        "device": args.device,
        "rnn": None,
        "dt": dt,
        "beta": -math.log(0.99) / dt,
        "normalize_reward": True,
        "critic_activation": args.critic_activation,
        "phibe_mode": "full_second_order",
        "lambda_phibe": float(args.lambda_phibe),
        "alpha": float(args.alpha),
        "use_safe_phibe": False,
        "save_tag": args.tag,
        "current_glucose_index": 8,
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
        reward_mean,
        reward_std,
    ) = unpackage_replay(
        trajectories=trajectories,
        empty_replay=agent.memory,
        data_processing=agent.data_processing,
        sequence_length=agent.sequence_length,
    )

    agent.action_std = 1.75 * agent.bas * 0.25 / (agent.action_std / agent.bas)
    agent.params["state_mean"], agent.params["state_std"] = agent.state_mean, agent.state_std
    agent.params["action_mean"], agent.params["action_std"] = agent.action_mean, agent.action_std
    agent.params["reward_mean"] = float(reward_mean)
    agent.params["reward_std"] = float(reward_std)
    agent.max_action = float(((agent.bas * 3.0) - agent.action_mean) / agent.action_std)
    agent.init_model()


def load_weights(agent, patient_params, train_seed, tag):
    suffix = patient_params["replay_name"].split("-")[-1]
    prefix = f"./Models/{patient_params['env_name']}{train_seed}TD3_offline_BC_{tag}_weights"
    actor_path = prefix + "_actor" + suffix
    critic_path = prefix + "_critic" + suffix
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        raise FileNotFoundError(f"Missing weights for tag={tag}, seed={train_seed}")
    agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
    agent.critic.load_state_dict(torch.load(critic_path, map_location=agent.device))
    agent.actor_target = copy.deepcopy(agent.actor)
    agent.critic_target = copy.deepcopy(agent.critic)
    agent.actor.eval()
    agent.critic.eval()
    agent.actor_target.eval()
    agent.critic_target.eval()


def phibe_loss_for_mode(agent, mode, state, action, reward, next_state):
    old_mode = agent.phibe_mode
    agent.phibe_mode = mode
    loss_q1, diag_q1 = agent._phibe_residual_loss(state, action, reward, next_state, agent.critic.Q1)
    loss_q2, diag_q2 = agent._phibe_residual_loss(state, action, reward, next_state, agent.critic.Q2)
    agent.phibe_mode = old_mode
    return loss_q1 + loss_q2, diag_q1, diag_q2


def collect_one_batch(agent, batch_index):
    state, action, reward, next_state, done, _, _, _, _, _ = get_batch(
        replay=agent.memory,
        batch_size=agent.batch_size,
        data_processing=agent.data_processing,
        sequence_length=agent.sequence_length,
        device=agent.device,
        params=agent.params,
    )

    with torch.no_grad():
        noise = (torch.randn_like(action) * agent.policy_noise).clamp(
            -agent.noise_clip, agent.noise_clip
        )
        next_action = (agent.actor_target(next_state) + noise).clamp(
            -agent.max_action, agent.max_action
        )
        target_q1, target_q2 = agent.critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + done * agent.gamma * target_q

        current_q1, current_q2 = agent.critic(state, action)
        td_q1 = F.mse_loss(current_q1, target_q)
        td_q2 = F.mse_loss(current_q2, target_q)
        td_loss = td_q1 + td_q2

        pi = agent.actor(state)
        actor_q = agent.critic.Q1(state, pi)
        actor_lambda = agent.alpha / actor_q.abs().mean()
        actor_q_term = -actor_lambda * actor_q.mean()
        actor_bc_loss = F.mse_loss(pi, action)
        actor_loss = actor_q_term + actor_bc_loss

    first_loss, first_q1, first_q2 = phibe_loss_for_mode(
        agent, "first_order", state, action, reward, next_state
    )
    full_loss, full_q1, full_q2 = phibe_loss_for_mode(
        agent, "full_second_order", state, action, reward, next_state
    )

    td_value = max(scalar(td_loss), 1e-12)
    first_value = max(scalar(first_loss), 1e-12)
    drift_value = max(float(full_q1["drift_term_abs"]), 1e-12)

    return {
        "batch": batch_index,
        "td_loss": scalar(td_loss),
        "td_q1_loss": scalar(td_q1),
        "td_q2_loss": scalar(td_q2),
        "target_q_mean": scalar(target_q.mean()),
        "target_q_abs_mean": scalar(target_q.abs().mean()),
        "current_q_abs_mean": scalar(0.5 * (current_q1.abs().mean() + current_q2.abs().mean())),
        "actor_loss": scalar(actor_loss),
        "actor_bc_loss": scalar(actor_bc_loss),
        "actor_q_term": scalar(actor_q_term),
        "actor_lambda": scalar(actor_lambda),
        "actor_q_mean": scalar(actor_q.mean()),
        "first_phibe_loss": scalar(first_loss),
        "full_phibe_loss": scalar(full_loss),
        "lambda_first_phibe": agent.lambda_phibe * scalar(first_loss),
        "lambda_full_phibe": agent.lambda_phibe * scalar(full_loss),
        "lambda_first_to_td": agent.lambda_phibe * scalar(first_loss) / td_value,
        "lambda_full_to_td": agent.lambda_phibe * scalar(full_loss) / td_value,
        "full_to_first_loss": scalar(full_loss) / first_value,
        "first_residual_abs_q1": first_q1["residual_abs"],
        "full_residual_abs_q1": full_q1["residual_abs"],
        "grad_s_norm_q1": full_q1["grad_s_norm"],
        "hess_vec_norm_q1": full_q1["hess_diag_norm"],
        "drift_abs_q1": full_q1["drift_term_abs"],
        "diffusion_abs_q1": full_q1["diffusion_term_abs"],
        "diffusion_to_drift_q1": float(full_q1["diffusion_term_abs"]) / drift_value,
        "drift_abs_q2": full_q2["drift_term_abs"],
        "diffusion_abs_q2": full_q2["diffusion_term_abs"],
    }


def summarize(rows):
    numeric_keys = [key for key in rows[0] if key not in ("seed", "batch")]
    groups = {"all": rows}
    for seed in sorted({row["seed"] for row in rows}):
        groups[f"seed{seed}"] = [row for row in rows if row["seed"] == seed]

    out = []
    for group, group_rows in groups.items():
        for key in numeric_keys:
            values = np.array([float(row[key]) for row in group_rows], dtype=np.float64)
            out.append(
                {
                    "group": group,
                    "metric": key,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return out


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_key_summary(summary_rows):
    keys = [
        "td_loss",
        "actor_bc_loss",
        "actor_q_term",
        "first_phibe_loss",
        "full_phibe_loss",
        "lambda_full_to_td",
        "diffusion_abs_q1",
        "drift_abs_q1",
        "diffusion_to_drift_q1",
        "full_to_first_loss",
    ]
    lookup = {(row["group"], row["metric"]): row for row in summary_rows}
    print("\nKey diagnostics, all seeds:")
    print(f"{'metric':<24}{'mean':>14}{'p50':>14}{'p90':>14}")
    print("-" * 66)
    for key in keys:
        row = lookup[("all", key)]
        print(f"{key:<24}{row['mean']:>14.4g}{row['p50']:>14.4g}{row['p90']:>14.4g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="adult#1")
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--training-timesteps", type=int, default=int(1e5))
    parser.add_argument("--tag", default="phibe_stage3_full_second_order_lam1e-4_norm_alpha1p75")
    parser.add_argument("--lambda-phibe", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.75)
    parser.add_argument("--critic-activation", default="softplus")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--raw-out", default="./Results/complete_phibe_terms_diagnostic_raw.csv")
    parser.add_argument("--summary-out", default="./Results/complete_phibe_terms_diagnostic_summary.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_envs()
    patient_params = get_params()[args.patient]
    train_seeds = parse_int_list(args.train_seeds)

    rows = []
    for train_seed in train_seeds:
        print(f"Preparing Complete PhiBE seed {train_seed}")
        params = make_params(patient_params, args)
        agent = td3_bc_phibe(init_seed=train_seed, patient_params=patient_params, params=params)
        prepare_agent(agent, patient_params, args.training_timesteps)
        load_weights(agent, patient_params, train_seed, args.tag)

        for batch_index in range(args.batches):
            row = collect_one_batch(agent, batch_index)
            row.update({"seed": train_seed})
            rows.append(row)
            print(
                f"seed={train_seed} batch={batch_index + 1}/{args.batches} "
                f"td={row['td_loss']:.3g} full_phi={row['full_phibe_loss']:.3g} "
                f"lambda_phi/td={row['lambda_full_to_td']:.3g} "
                f"diff/drift={row['diffusion_to_drift_q1']:.3g}",
                flush=True,
            )

    raw_fields = ["seed", "batch"] + [key for key in rows[0] if key not in ("seed", "batch")]
    write_csv(args.raw_out, rows, raw_fields)
    summary_rows = summarize(rows)
    write_csv(args.summary_out, summary_rows)
    print_key_summary(summary_rows)
    print(f"\nSaved raw diagnostics: {args.raw_out}")
    print(f"Saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()
