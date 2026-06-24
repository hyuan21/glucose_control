# Meeting Summary — 2026-05-25

> 给教授（Yuhua Zhu）汇报本周（5/22–5/25）工作的结构化提纲。
>
> 重点：(1) 实验做了什么，(2) 怎么失败的，(3) 一个意外的正向发现，(4) 我们想跟教授请教的方向。

---

## 1. 数学起点

**教授指示 2 邮件原文**:
> "use the same trick from off-policy learning for the Q defined by the Bellman equation in RL, but apply it to this Q function."

我们的实现 (Path B)：把 TD3-BC 的离散 Bellman target 换成教授连续时间 Q 公式的二阶泰勒展开作为 critic target 主体（不是辅助正则）。

**起点公式**（教授给的）：
$$
Q(s_t, a_t) = r \cdot \Delta t + e^{-\beta \Delta t} \cdot \mathbb{E}\big[Q(s_{t+1}, a_{t+1}) \mid s_t, a_t\big]
$$

**泰勒展开后的 per-sample target**：
$$
\text{target}_i = r_i + e^{-\beta\Delta t}\Big[Q_\text{target}(s_i, a_i) + \Delta s_i^\top \nabla_s Q + \Delta a_i^\top \nabla_a Q + \tfrac{1}{2}\Delta s_i^\top \nabla_s^2 Q \, \Delta s_i\Big]
$$

reward-rate 约定 (HANDOFF §5.2) 保留；TD3-BC 整体架构（target net、replay buffer、actor-critic、BC 项）不动；只换 critic target 一行。

**off-policy 性质保持**：所有 $\mathbb{E}[\cdot]$ 用 buffer 单 sample 估计，target 只依赖 $(s_i, a_i, s_i', r_i)$，不依赖 behavior policy。

数学严格性已通过单元测试验证（`test_path_b_math.py`：HVP 对照真 Hessian、形状一致、target critic 不收梯度等）。

---

## 2. 教授论文里的 open problem (§3.3.2)

教授论文 page 20 第二条 remark：
> *"As in classical RL, convergence guarantees are generally unavailable under nonlinear function approximation. In practice, stability can be improved by decoupling the target from the online network..."*

教授 §5 conclusion 把 "extending our methodology to large-scale problems via deep neural approximators" 列为 future work。

**这正是我们撞到的问题**：用 neural critic 实现 Path B 时，$\|\nabla_s Q\|$ 在训练中单调增长，target 数值失控，actor 学坏。Yuhua Lemma 3.1 在 linear basis 下用的 c_1, c_2, c_3 常数有界——这对 neural network **没有**自然成立。

详细定位文档：见 `RESEARCH_POSITION.md`。

---

## 3. 全部尝试的实验

所有实验：`adult#1`，3 min 采样，50k QUICK 单 seed (除非另注)，PID demo buffer。

### 3.1 baseline 复现
| 配置 | TIR | 备注 |
|---|---|---|
| Baseline FULL 100k×3 seed | **67.48% ± 3.16%** | HANDOFF §6.2（旧数据，已知）|
| Baseline 50k×1 同 seed | 63.65% | 这周复现做对照 |
| **Baseline + reward normalization (50k×1)** | **74.29%** | ⚠️ **后面会重点讨论这个** |

### 3.2 Path B 各种变体（按尝试顺序）

| 配置 | TIR | 训练稳定性 | 主要诊断 |
|---|---|---|---|
| Path B neural critic 50k | 59.69% | ❌ critic loss 30x 暴涨 | ∇_s Q 无界增长 |
| Path B neural critic FULL 100k | **5.46%** | ❌ runaway 完成态 | Q 学坏，actor 跟着崩 |
| Path B + input-grad clip=10 | 54.19% | ⚠️ 撞 cap | clip 削掉了真信号 |
| Path B + clip=10 + reward normalize | 41.29% | ⚠️ 撞 cap | normalize 让 \|Q\| 小，clip 相对太严 |
| Path B + clip=0.5 + normalize | 14.04% | ⚠️ 撞 cap | clip 太紧 |
| **Path B + spectral_norm (Lip=1) + normalize** | **56.96%** | ✅ **完全稳定** | Lipschitz 太紧，Q 表达力受限 |
| Path B + spectral_norm (sn_scale=5) + normalize | 36.54% | ⚠️ | 放宽方向错了 |
| Path B + linear_basis (Yuhua §4.1 alignment) + normalize | 3.71% | ⚠️ | actor loss = -2.4 跑飞 |
| Path B + linear_basis + normalize + fixed_lambda | **2.19%** | ❌ actor loss -817 | 即使 fix lambda 还是发散 |

### 3.3 Path B 的成绩天花板

**所有训练稳定的 Path B 配置**：最高 TIR = **56.96%**（spectral_norm + normalize）。

**比 baseline 50k 的 63.65% 低 7%**，比 baseline + normalize 的 74.29% 低 17%，**比 PID 的 59.50% 还低**。

---

## 4. 关键发现（不止是失败）

### 4.1 意外的大新闻：reward normalization 让 baseline 跳到 **74.29%**

baseline 多年来一直在 ~67% 区间（论文报告 adult 70%）。我们这周顺手把 buffer 上的 reward 做了 z-score 归一化（保留 reward shape，只改尺度），TIR 一下涨了 **10.6%**——首次超过 paper baseline。

reward 原始量级：BG=144 给 r=-0.07，BG=70 给 r=-25，terminal -1e5。归一化后量级是 -1~1。这恰好是 D4RL 那种 setting 下 TD3-BC alpha=2.5 想要的合理 |Q| 量级。**HANDOFF §3 一直未解的 gap 可能就是这件事**。

### 4.2 Spectral Norm 直接解决了 Path B 的 stability open problem

加 spectral_norm 后：
- $\|\nabla_s Q\|$ 全程稳定在 ~0.88，**不再单调发散**
- critic loss 1.7-2.7 区间震荡，不爆炸
- 这是教授论文 §3.3.2 future work 里 "DNN stability" 的一个具体可行解决方案

但代价：Lip=1 太严，Q 表达力受限，TIR 上限被压在 ~57%。

### 4.3 第一次让二阶项真正起作用

ReLU 神经网络下 $\nabla_s^2 Q \equiv 0$——所以 neural Path B 实际上是"一阶 Path B"。
linear basis critic 下 $\|\nabla_s^2 Q \cdot \Delta s\|$ 在 0.1-0.7 之间，**首次非零**。
教授泰勒展开里的二阶项第一次真正参与训练。

---

## 5. 失败的核心诊断

**最深的事实**：在 single-patient + PID-demo + Magni reward 这个 setting 下——

| 算法层 | 行为 |
|---|---|
| Baseline (离散 Bellman, target 只依赖 Q 的**值**) | 稳定，~67-74% |
| Path B (target 依赖 Q 的**值 + 梯度 + 二阶导**) | 需要额外约束才稳定；约束后表达力又不够；TIR ceiling ~57% |

**Path B 的根本困难**：target 公式里出现 $\nabla_s Q$ 和 $\nabla_s^2 Q$——它们的尺度 / 稳定性比 Q 值本身更难控制。我们尝试的所有"控制方式"（clip / spec_norm / linear basis）都让 Q 表达力下降，无法达到 baseline 水平。

**还有几个我们怀疑但没机会验证的 contributing factors**:
- buffer 单一性（HANDOFF §6.7 你之前的洞察）—— PID 是确定性策略，buffer 在 action space 几乎一条线，critic 在 OOD 无信号
- Magni reward 在 BG≈70 附近曲率极大 + terminal -1e5 cliff——Q 不光滑，泰勒展开假设的局部光滑性受挑战
- $\Delta s$ 可以很大（餐后 BG 跳 30+ mg/dL），一阶泰勒近似误差大

---

## 6. 想跟教授请教的问题

**问题 1**：连续时间 Q 公式作为 critic target 这条路（Path B），在我们的实测里**ceiling ~57%**，低于 baseline 67%。**您对这件事的解读**？

候选解读 a：实验设定问题——单 patient + PID-only buffer + 我们具体的 reward 形式，确实让 Path B 这条路不显优势。换 setting 可能就行（多 patient、多策略 buffer、光滑 reward）。

候选解读 b：算法/工程层缺一个我们没找到的稳定化技巧。比如教授论文 §3.3.2 的 nonlinear extension hint，您是否有更具体的 stabilization 想法？

候选解读 c：连续时间公式应该用在别的位置（不是 critic target 主体）——比如作为 OOD 诊断、跨 patient 迁移、actor 改进的 $\nabla_a Q$ 信号？

**问题 2**：reward normalization 让 baseline 跳到 74.29%——这件事**研究意义有多大**？是否值得作为一个独立发现报告？

**问题 3**：spectral_norm + Path B 给出的"稳定但表达力受限" trade-off，是否对应您论文 §3.3.2 future work 里的某种 architecture constraint 思路？

**问题 4**：(r, β) pair 这个指示 1 我们一直没正面碰。在我们 baseline 已经 ≥ paper adult 70% 的情况下，您希望我们怎么定位 (r, β)？

---

## 7. 仓库状态

GitHub: https://github.com/hyuan21/glucose_control（待 push 本周新文件）

新增文件（这周做的）：
- `PATH_B_MATH.md` — 数学推导（公式 1-6 链式）
- `RESEARCH_POSITION.md` — 把 Path B 定位成攻击 §3.3.2 open problem
- `MEETING_SUMMARY_2026-05-25.md` — 本文档
- `TD3_BC_ct.py` — 含 Path B、spec_norm、linear_basis 三种 critic、lambda_mode 切换
- 多个 `run_*.py` 实验脚本（baseline_norm / path_b_quick / path_b_full / path_b_specnorm / path_b_specnorm_scale5 / path_b_linbasis）
- `compare_quick.py` — 评估脚本
- `test_path_b_math.py` / `test_linear_basis.py` — 数学层单元测试

所有 baseline / A v2 / C v1-v3 历史实验路径**完全保留可复现**——所有 Path B 改造通过新 flag 开关（`use_path_b`、`normalize_reward`、`use_spectral_norm`、`critic_type`、`lambda_mode`）触发，默认关闭。

---

**一句话总结**：Path B 数学落地正确，证伪了"教授连续时间公式作为 critic target 主体能在我们 setting 下超过 baseline"；意外发现 reward normalization 让 baseline 超过 paper；下一步想跟教授对齐研究方向。
