"""
CHECK THE PhiBE LOSS (professor's directive, step 1).

The professor pointed out that the PhiBE critic loss is NOT a standard RL
loss: the regression target itself contains DERIVATIVES of Q wrt the input
(grad_s Q, grad_a Q, and a Hessian term), evaluated on the buffer's
Delta-input (s'-s, a'-a). Standard RL targets depend only on Q's VALUE.

Before deciding between
    (1) change the function approximation  (=> rebuild the whole framework), or
    (2) find a better optimization solver for this derivative-bearing loss,
we first MEASURE how this non-standard loss misbehaves on a NEURAL critic.

This script trains Path B with a plain 256x256 ReLU critic for FULL 100k
steps with diagnostics on, and with input-gradient clipping effectively
DISABLED (path_b_grad_clip huge) so we observe the TRUE runaway rather than
the capped value. Each PhiBE target component is logged every 1000 steps.

It records, over training:
    grad_s_norm / grad_a_norm / grad_s_max   (pre-clip; the derivative terms)
    comp_Q / comp_ds_gradsQ / comp_da_gradaQ / comp_2nd  (target pieces)
    critic_loss / target_Q_mean / target_Q_absmax / current_Q_absmean

Key expected evidence for the (1)-vs-(2) decision:
  * If grad_s Q grows unboundedly  -> the derivative term in the target is the
    instability source; ordinary optimizers cannot tame an unbounded target.
  * comp_2nd (the Hessian term) is ~0 throughout, because a ReLU net has zero
    second derivative -> the professor's 2nd-order term does nothing under the
    current function approximation. That alone is an argument for (1).

Usage:
    python diagnose_phibe_loss.py            # FULL 100k (default)
    python diagnose_phibe_loss.py --steps 50000
"""
import argparse
import csv
import math

import numpy as np

from utils import get_params, create_env
from TD3_BC_ct import td3_bc_ct


def register_envs():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000,
                    help="training steps (FULL = 100000)")
    ap.add_argument("--log-freq", type=int, default=1000)
    ap.add_argument("--out", default="phibe_loss_diag.csv")
    args = ap.parse_args()

    register_envs()

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
        "training_timesteps": int(args.steps),
        "device": "cpu",
        "rnn": None,
        "dt": DT,
        "beta": BETA,

        # ===== PhiBE / Path B on a NEURAL critic (the thing we diagnose) =====
        "use_path_b": True,
        "use_hjb": False,
        "critic_type": "neural",       # 256x256 ReLU -> 2nd-order term == 0
        "use_spectral_norm": False,    # no stabiliser: we want the raw behaviour
        "normalize_reward": True,      # keep |Q| in the sane regime

        # Disable input-grad clipping so we SEE the true runaway, not the cap.
        "path_b_grad_clip": 1e9,

        # ===== turn diagnostics on =====
        "diag_phibe": True,
        "diag_log_freq": args.log_freq,
    }

    print("CHECK PhiBE LOSS  (neural critic, Path B)")
    print(f"  steps = {args.steps:,} | log every {args.log_freq} | "
          f"clip = OFF (raw) | critic = neural ReLU")
    print(f"  dt={DT} beta={BETA:.3e}  exp(-beta*dt)={math.exp(-BETA*DT):.4f}\n")

    agent = td3_bc_ct(init_seed=0, patient_params=patient_params, params=params)
    agent.train_model()

    hist = agent._diag_history
    if not hist:
        print("No diagnostics captured (diag_phibe off?).")
        return

    # ---- write CSV ----
    keys = ["t", "grad_s_norm", "grad_a_norm", "grad_s_max",
            "comp_Q", "comp_ds_gradsQ", "comp_da_gradaQ", "comp_2nd",
            "critic_loss", "target_Q_mean", "target_Q_absmax",
            "current_Q_absmean"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for snap in hist:
            w.writerow({k: snap.get(k, "") for k in keys})
    print(f"\nWrote {len(hist)} snapshots -> {args.out}")

    # ---- console summary: first vs last, growth factors ----
    def g(snap, k):
        v = snap.get(k, float("nan"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    first, last = hist[0], hist[-1]
    print("\n" + "=" * 70)
    print(f"{'metric':<20}{'@start':>14}{'@end':>14}{'growth x':>14}")
    print("-" * 70)
    for k in keys[1:]:
        a, b = g(first, k), g(last, k)
        growth = (b / a) if (a not in (0.0,) and a == a and b == b) else float("nan")
        print(f"{k:<20}{a:>14.4g}{b:>14.4g}{growth:>14.2f}")
    print("=" * 70)

    # ---- automatic reading of the evidence ----
    gs_a, gs_b = g(first, "grad_s_norm"), g(last, "grad_s_norm")
    c2_max = max(g(s, "comp_2nd") for s in hist)
    cl_a, cl_b = g(first, "critic_loss"), g(last, "critic_loss")
    print("\nREADING THE EVIDENCE:")
    if gs_b == gs_b and gs_a not in (0.0,) and gs_b / gs_a > 3:
        print(f"  * grad_s Q grew {gs_b/gs_a:.1f}x ({gs_a:.2f} -> {gs_b:.2f}): the "
              "derivative term in the PhiBE target is UNBOUNDED. A standard "
              "optimizer regresses onto a target that itself diverges.")
    else:
        print(f"  * grad_s Q stayed bounded ({gs_a:.2f} -> {gs_b:.2f}).")
    print(f"  * max |2nd-order term| over all of training = {c2_max:.3e}  "
          "(ReLU critic has zero curvature -> the professor's 2nd-order term "
          "is INERT under this function approximation).")
    print(f"  * critic_loss: {cl_a:.3g} -> {cl_b:.3g}.")
    print("\nINTERPRETATION for the (1) vs (2) decision:")
    print("  - 2nd-order term inert + grad_s Q unbounded under a ReLU net is")
    print("    evidence that the FUNCTION APPROXIMATION is the bottleneck (-> 1).")
    print("  - If grad_s Q is the ONLY thing diverging and the rest is healthy,")
    print("    a derivative-aware solver/penalty might rescue it without a")
    print("    framework change (-> 2). Inspect the CSV trend to decide.")


if __name__ == "__main__":
    main()
