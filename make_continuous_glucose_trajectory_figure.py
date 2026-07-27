"""
Create a continuous-time glucose trajectory figure comparing PID, TD3-BC,
and PhiBE-TD3-BC on Adult#1.

The x-axis is real time in hours, not simulation step index. By default this
uses the latest full-Hessian PhiBE weights and the corrected multi-dt state
construction in utils/evaluation.py.
"""

import argparse
import csv
import os

import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_multidt_stage2 import build_baseline, build_phibe, register_envs
from utils import get_params, set_env_sample_time, test_algorithm


def metrics(bg):
    arr = np.asarray(bg, dtype=np.float64)
    return {
        "tir": float(np.mean((arr >= 70.0) & (arr <= 180.0)) * 100.0),
        "tbr": float(np.mean(arr < 70.0) * 100.0),
        "tar": float(np.mean(arr > 180.0) * 100.0),
        "mean_bg": float(np.mean(arr)),
    }


def rollout(agent, patient_params, test_seed, dt, days):
    env = gym.make(patient_params["env_name"])
    actual_dt = float(set_env_sample_time(env, dt))
    max_timesteps = int(round(days * 24 * 60 / actual_dt))
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
        pid_run=False,
        params=agent.params,
        sample_time=actual_dt,
    )
    return {
        "actual_dt": actual_dt,
        "rl_bg": list(rl_bg),
        "pid_bg": list(pid_bg),
        "rl_reward": rl_reward,
        "pid_reward": pid_reward,
    }


def write_trajectory_csv(path, time_hours, pid_bg, td3_bg, phibe_bg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["time_hours", "pid_bg", "td3_bc_bg", "phibe_bg"],
        )
        writer.writeheader()
        for idx, time_hour in enumerate(time_hours):
            writer.writerow(
                {
                    "time_hours": float(time_hour),
                    "pid_bg": float(pid_bg[idx]),
                    "td3_bc_bg": float(td3_bg[idx]),
                    "phibe_bg": float(phibe_bg[idx]),
                }
            )


def make_figure(output_png, output_pdf, time_hours, pid_bg, td3_bg, phibe_bg):
    pid_m = metrics(pid_bg)
    td3_m = metrics(td3_bg)
    phibe_m = metrics(phibe_bg)

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=180)
    ax.axhspan(70, 180, color="#dff2df", alpha=0.85, label="Target range 70 - 180 mg/dl")
    ax.axhline(70, color="#b74141", linewidth=1.0, alpha=0.85)
    ax.axhline(180, color="#b74141", linewidth=1.0, alpha=0.85)

    ax.plot(
        time_hours,
        pid_bg,
        color="#7a7a7a",
        linewidth=1.35,
        label=f"PID (TIR {pid_m['tir']:.1f}%)",
    )
    ax.plot(
        time_hours,
        td3_bg,
        color="#1f77b4",
        linewidth=1.45,
        label=f"TD3-BC (TIR {td3_m['tir']:.1f}%)",
    )
    ax.plot(
        time_hours,
        phibe_bg,
        color="#d62728",
        linewidth=1.45,
        label=f"PhiBE (TIR {phibe_m['tir']:.1f}%)",
    )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Blood glucose (mg/dL)")
    ax.set_xlim(float(time_hours[0]), float(time_hours[-1]))
    ax.set_ylim(40, 320)
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.grid(True, axis="x", color="#eeeeee", linewidth=0.5, alpha=0.55)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="adult#1")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--days", type=float, default=2.0)
    parser.add_argument("--training-timesteps", type=int, default=int(1e5))
    parser.add_argument("--phibe-tag", default="phibe_stage4_full_second_order_lam1e-2_norm_alpha1p25")
    parser.add_argument(
        "--phibe-mode",
        default="full_second_order",
        choices=["first_order", "second_order", "full_second_order", "none"],
    )
    parser.add_argument("--lambda-phibe", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--output-dir", default="./Figures")
    args = parser.parse_args()

    register_envs()
    patient_params = get_params()[args.patient]
    baseline = build_baseline(patient_params, args.train_seed, args.training_timesteps)
    phibe = build_phibe(
        patient_params,
        args.train_seed,
        args.training_timesteps,
        args.phibe_tag,
        args.phibe_mode,
        args.lambda_phibe,
        args.alpha,
    )

    baseline_out = rollout(baseline, patient_params, args.test_seed, args.dt, args.days)
    phibe_out = rollout(phibe, patient_params, args.test_seed, args.dt, args.days)
    actual_dt = baseline_out["actual_dt"]

    pid_bg = np.asarray(baseline_out["pid_bg"], dtype=np.float64)
    td3_bg = np.asarray(baseline_out["rl_bg"], dtype=np.float64)
    phibe_bg = np.asarray(phibe_out["rl_bg"], dtype=np.float64)
    min_len = min(len(pid_bg), len(td3_bg), len(phibe_bg))
    pid_bg = pid_bg[:min_len]
    td3_bg = td3_bg[:min_len]
    phibe_bg = phibe_bg[:min_len]
    time_hours = np.arange(min_len, dtype=np.float64) * actual_dt / 60.0

    stem = (
        f"adult1_continuous_glucose_trajectory_dt{actual_dt:g}"
        f"_train{args.train_seed}_test{args.test_seed}_{args.days:g}days"
    )
    output_png = os.path.join(args.output_dir, stem + ".png")
    output_pdf = os.path.join(args.output_dir, stem + ".pdf")
    output_csv = os.path.join(args.output_dir, stem + ".csv")

    make_figure(output_png, output_pdf, time_hours, pid_bg, td3_bg, phibe_bg)
    write_trajectory_csv(output_csv, time_hours, pid_bg, td3_bg, phibe_bg)

    print("Saved figure:")
    print(output_png)
    print(output_pdf)
    print("Saved trajectory data:")
    print(output_csv)
    for name, bg in [("PID", pid_bg), ("TD3-BC", td3_bg), ("Full-Hessian PhiBE-TD3-BC", phibe_bg)]:
        m = metrics(bg)
        print(
            f"{name}: TIR={m['tir']:.2f}% TBR={m['tbr']:.2f}% "
            f"TAR={m['tar']:.2f}% meanBG={m['mean_bg']:.1f}"
        )


if __name__ == "__main__":
    main()
