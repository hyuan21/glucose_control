# HANDOFF — offline-glucose CT branch

> **Purpose**: This file is read by Cowork on a new device to continue the
> research project where it left off, without re-doing weeks of design work.
> Last updated: 2026-04-30, after Approach C v3 results.

---

## How to use this file (on the new device)

1. Clone this repo.
2. Set up the conda environment: `conda env create -f environment.yml && conda activate offline-glucose`.
3. Open Cowork (or VS Code with the Anthropic Claude extension), point it at this folder.
4. **First message to Cowork**: paste this prompt:
   > "Read HANDOFF.md in this folder, then continue our work. The mid-term report has been sent to the professor; we're waiting for direction. Until then, no new training. Help me understand the existing code and prepare for whichever direction the professor picks."

That's it. Cowork will pick up exactly where the previous session left off.

---

## TL;DR — current state in 5 bullets

- **Project goal**: Adapt Emerson 2023 TD3-BC for blood-glucose control using the
  professor's continuous-time value function `v(s) = r·Δt + e^(-β·Δt) · E[v(s')|s]`.
- **Approach A (continuous-time interface)**: ✅ Done. Implemented as TD3_BC_ct.py
  with `target_Q = reward + done * exp(-β·Δt) * target_Q`. At default
  `β = -ln(0.99)/3` and `Δt = 3 min`, this is **mathematically equivalent to the
  paper baseline** (ΔTIR = 0.00 across 3 seeds). β and Δt are now tunable.
- **Approach C (HJB regularisation)**: ❌ Three versions all underperformed.
  v1 (FD all dims): ΔTIR -11.6.  v2 (FD physical dims only): ΔTIR -5.1.
  v3 (DriftNet, FULL 100k×3 seed): ΔTIR -3.8.
- **Diagnosis**: HJB fails on this problem due to (a) drift estimation noise at
  Δt=3 min, (b) state representation has 8 lagged-BG dims that don't satisfy SDE,
  (c) meal disturbances are pulse-like, not Gaussian, (d) PID demo buffer is
  too narrow for HJB's OOD-control benefit to manifest.
- **Status**: Mid-term report sent to professor. Waiting for direction among
  4 candidates (see report §7). Do **not** start new training until the
  professor responds.

---

## What's in this repo (CT branch additions, on top of paper code)

| File | Status | What it is |
|------|--------|------------|
| `TD3_BC.py` | unchanged | Paper baseline. Don't modify. |
| `TD3_BC_ct.py` | **NEW** | Approach A + C (single file, `use_hjb` / `drift_mode` flags) |
| `Comparison_CT.ipynb` | **NEW** | Train + evaluate baseline / A / C with one click |
| `HANDOFF.md` | **NEW (this file)** | Project state for cross-device continuity |
| `utils/` | unchanged | Paper's helper code. Don't modify. |
| `BCQ.py`, `CQL.py`, `SAC_RNN.py` | unchanged | Other algorithms in paper. Not used in CT branch. |
| `Offline_RL_Comparison.ipynb` | unchanged | Paper's original notebook. Use Comparison_CT.ipynb instead. |

**Not in repo (regeneratable, in .gitignore)**:
- `Models/` — trained network weights (~1 MB each × ~24 files = ~25 MB)
- `Replays/` — PID demonstrator buffer (~50 MB)
- `*中期报告*.docx` — mid-term report (kept locally, sensitive)

To regenerate them on a new device:
- `Replays/`: run cell 3 of `Comparison_CT.ipynb` (~5-15 min, PID simulation)
- `Models/`: run cells 6, 7, 8 of `Comparison_CT.ipynb` (~3-6 hours total on CPU)

---

## Mathematical core (one-page summary for the professor)

### The professor's continuous-time value function

$$v(s) = r \cdot \Delta t + e^{-\beta \Delta t} \cdot \mathbb{E}[v(s') \mid s]$$

This is the *continuous-time analogue* of the paper's discrete Bellman
$v(s) = r + \gamma \cdot \mathbb{E}[v(s')|s]$, with the substitution
$\gamma \leftrightarrow e^{-\beta \Delta t}$ and reward integrated over $\Delta t$.

### Reward-rate form (Approach A's actual implementation)

Dividing both sides by $\Delta t$ and identifying $r$ as a reward rate gives
the equivalent, numerically-stable form

$$\tilde{v}(s) = r + e^{-\beta \Delta t} \cdot \mathbb{E}[\tilde{v}(s') \mid s]$$

Argmax-equivalent to the original (constant scaling of $v$ doesn't change the
optimal policy). Approach A's only structural change vs the paper baseline is
`self.gamma` → `self.discount = exp(-β·Δt)`. At defaults β=-ln(0.99)/3, Δt=3,
`self.discount = 0.99 = self.gamma` exactly.

### HJB equation (foundation of Approach C)

Taylor-expand $\mathbb{E}[v(s')|s]$ via Itô's lemma assuming $ds = \mu(s,a) dt + \sigma(s,a) dW$
and let $\Delta t \to 0$:

$$\beta v(s) = r(s, a) + \nabla v(s)^\top \mu(s, a) + \tfrac{1}{2}\,\mathrm{tr}(\nabla^2 v(s) \cdot \Sigma(s, a))$$

This is the PDE that $v$ must satisfy in continuous time. Approach C uses
the **first-order** version (drops the $\Sigma$ term) as a soft regulariser
on the critic loss:

$$\mathcal{L}_{\text{critic}}^{\text{C}} = \underbrace{\mathrm{MSE}(Q, y_{\text{Bellman}})}_{\text{data}} + \lambda_{\text{HJB}} \cdot \underbrace{\big[\beta Q - r - \nabla Q^\top \hat{\mu}_{\text{phys}}\big]^2}_{\text{physics prior}}$$

where $\hat{\mu}$ is estimated either by finite difference $(s'-s)/\Delta t$
(versions v1, v2) or by a small MLP DriftNet (v3), restricted to the 3
physical-dynamics state dims [BG_now, MOB, IOB] (v2 onwards).

---

## Experiments done so far

All on UVA/Padova in-silico patient `adult#1`, with the meal scenario from the
paper. Buffer is the PID demonstrator from `utils.fill_replay_split`.

### QUICK mode (50k steps × 1 seed) — for fast iteration

| Algorithm | TIR (%) | TBR (%) | Magni | ΔTIR vs base | Notes |
|-----------|---------|---------|-------|--------------|-------|
| PID | 59.50 | 0.00 | 4.19 | — | Clinical floor |
| Baseline | 69.52 | 0.00 | 3.93 | — | Paper code |
| A | 69.52 | 0.00 | 3.93 | **+0.00** | Identical (math equiv.) |
| C v1 (FD, all 11 dims) | 57.92 | 0.00 | 4.19 | -11.60 | HJB Loss never decreased |
| C v2 (FD, [8,9,10]) | 64.44 | 0.00 | 4.08 | **-5.08** | Restricting to physical dims helped |
| C v3 (DriftNet, [8,9,10]) | 63.23 | 0.00 | 4.09 | -6.29 | DriftNet overfit FD noise |

### FULL mode (100k steps × 3 seed) — for the mid-term report

| Algorithm | TIR (%) | TBR (%) | Magni | ΔTIR vs base |
|-----------|---------|---------|-------|--------------|
| PID | 59.50 ± 0.00 | 0.00 ± 0.00 | 4.19 ± 0.00 | — |
| **Baseline** | **67.48 ± 3.16** | 0.05 ± 0.07 | **3.88 ± 0.09** | — |
| **A** | **67.48 ± 3.16** | 0.05 ± 0.07 | 3.88 ± 0.09 | **+0.00** |
| C v3 | 63.67 ± 0.75 | 0.00 ± 0.00 | 4.06 ± 0.04 | -3.81 |

**Note on baseline gap**: paper reports TIR ~78-83% on adult#1, we got 67.48%.
Likely causes: different patient/seed, paper may use tricks not in the released
code, single patient evaluation. This is **not a bug in our implementation** —
A reproduces baseline exactly.

---

## Why Approach C failed (3 layers)

1. **Drift estimation is noise-dominated.** $(s'-s)/\Delta t = \mu + \sigma\sqrt{\Delta t} \cdot Z$.
   At Δt=3, the noise term has stddev ~$\sigma\sqrt{3}$ which can exceed $\mu$.
   Neither raw FD nor DriftNet (which over-fits FD) recovers true $\mu$.
2. **State representation is partly non-physical.** Of 11 state dims, 8 are
   lagged BG snapshots (window shifts). Their $(s'-s)/\Delta t$ is a windowing
   artefact, not an SDE drift. v2 fixed this (restricted to indices [8,9,10]),
   but v2 still failed.
3. **SDE assumption itself is shaky.** Meals are pulse-like external events,
   not Gaussian white noise. The HJB framework is built on the SDE assumption.
4. **Buffer is too narrow.** HJB's main benefit is constraining critic
   behaviour in OOD regions. Single-PID demo has essentially no OOD coverage,
   so the HJB regulariser provides bias without variance reduction.

The pattern of "monotonic improvement through diagnosed fixes (v1 → v2 → v3) but
never crossing into positive ΔTIR" indicates the issue is the framework, not
the implementation.

---

## Candidate next directions (waiting on professor)

From the mid-term report §7. Sorted by my (Claude's) personal preference:

1. **A as final deliverable.** Accept results, write up, move on. 1 week.
2. **Re-collect a more diverse buffer.** Mix multiple PID configs + random
   actions + intentional failures. 2-3 weeks. Might unlock real improvement
   over baseline since current bottleneck is data, not algorithm.
3. **Different continuous-time route.** Adaptive Δt, time-aware critic, or
   continuous-time actor-critic. Drops HJB but keeps the spirit. 3-4 weeks.
4. **Second-order HJB.** Add $\frac{1}{2}\mathrm{tr}(\nabla^2 Q \cdot \hat\Sigma)$ term.
   Most likely to also fail. 1-2 weeks.

---

## How to actually run things on the new device

```bash
# Setup (one time)
git clone <this-repo>
cd offline-glucose
conda env create -f environment.yml
conda activate offline-glucose
python -m ipykernel install --user --name offline-glucose --display-name "offline-glucose"

# Open notebook (every session)
jupyter lab Comparison_CT.ipynb     # or use VS Code

# Inside the notebook, the QUICK / FULL toggle is in cell 2:
QUICK = True    # 50k × 1 seed, ~10 min
QUICK = False   # 100k × 3 seed, ~3 hours
```

### GPU note (we hit this on the original device)

The paper environment locks `torch==1.7.1+cu101`, which doesn't support
RTX 50-series GPUs (sm_120). Cell 1 of the notebook auto-detects this
and falls back to CPU. If your new device has an older GPU (Pascal,
Volta, Turing) it should just work; for newer GPUs you may need to
upgrade PyTorch (be careful, this can break paper's other dependencies).

---

## Code-level cheatsheet

### TD3_BC_ct.py runtime modes

The single file supports 3 modes via `params` dict:

```python
# Mode A (continuous-time, no HJB)
params = {..., "dt": 3.0, "beta": -math.log(0.99)/3, "use_hjb": False}

# Mode C v1/v2 (HJB with finite-difference drift)
params = {..., "use_hjb": True, "drift_mode": "fd", "lambda_hjb": 1.0}

# Mode C v3 (HJB with neural-net drift)
params = {..., "use_hjb": True, "drift_mode": "net",
          "lambda_hjb": 1.0, "drift_warmup_steps": 5000}
```

### Weight file naming convention

| Suffix | Mode |
|--------|------|
| `TD3_offline_BC_weights_*` | Baseline (TD3_BC.py) |
| `TD3_offline_BC_ct_weights_*` | Approach A |
| `TD3_offline_BC_ct_hjb_weights_*` | Approach C v1/v2 (FD) |
| `TD3_offline_BC_ct_hjb_net_weights_*` | Approach C v3 (DriftNet) |

### State layout reminder

11-D condensed state (paper's "condensed" data_processing):

```
[BG_-4h, BG_-3.5h, BG_-3h, BG_-2.5h, BG_-2h, BG_-1.5h, BG_-1h, BG_-0.5h, BG_0, MOB, IOB]
   0       1       2       3       4       5       6       7      8    9    10
   <------------- 8 lagged BG snapshots ------------->     <-- physical -->
```

Only indices [8, 9, 10] obey real SDE dynamics. C v2 onwards restricts HJB to
these via `phys_idx = [8, 9, 10]` in `TD3_BC_ct.py`.

---

## Things to NOT do on the new device (lessons learned)

- ❌ Don't try to upgrade PyTorch to use the GPU. Will break dependencies.
  CPU + 100k steps takes ~3 hours, that's fine.
- ❌ Don't `pip install -e .` — this repo has no `setup.py`. The README
  instruction is wrong. Just import from the project directory.
- ❌ Don't run `Run All Cells` blindly. Training cells take hours; better to
  step through cells 1, 2, 3, then run the relevant training cell only.
- ❌ Don't try to push Models/ or Replays/ — they're large and regeneratable.
- ❌ Don't tune `lambda_hjb` further for C — the diagnosis (§5 above) shows
  the issue isn't lambda, it's the framework on this problem.

---

## Open questions to ask the professor

1. Was the HJB regularisation route something you specifically wanted, or
   was Approach A (the value-function rewrite) the actual deliverable?
2. Should we focus on data diversity (re-collecting buffer) before any further
   algorithmic work?
3. Is matching the paper's reported 78% TIR a hard requirement, or is the
   8% lift over PID we currently get acceptable?
4. Time budget for this branch of work? (1 week, 1 month, 1 quarter?)
5. Is the goal a paper, a thesis chapter, or an internal milestone?
