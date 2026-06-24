# Research Position — 我们到底在做什么

> **最后更新**：2026-05-25
> **目的**：把项目的研究定位锚定清楚，避免会话切换 / 跟教授沟通时反复回到 "Path B 失败、要不要换方向" 的循环。
>
> 这份文档比 PATH_B_MATH.md 更上层——PATH_B_MATH.md 是数学，这份是 **为什么这件事值得做、它在文献里的位置**。

---

## 1. 一句话定位

**我们攻击的是 Zhu 教授的 PhiBE-Q 论文 (Ren & Zhu, 2025) 显式留下的 open problem：让 PhiBE-Q 在 deep neural network critic 下保持稳定。**

---

## 2. 论文里的明确证据——这是真的 open problem

### 2.1 论文给的理论保证全部建立在 linear basis

PhiBE-Q paper §3.2 全部理论（Lemma 3.1, Theorem 3.2, Theorem 3.3）建立在：

$$
Q(s, a) = \Phi(s, a)^\top \theta
$$

——也就是 **linear in θ**。所有 Lipschitz 证明（Lemma 3.1）依赖 basis function 的 c_1, c_2, c_3 常数 bounded——这对 fixed basis function 显然成立，对 neural network **不成立**。

### 2.2 §3.3.2 第二个 remark 明确承认 neural extension 没有保证

逐字引用 page 20：

> "the algorithm naturally extends to nonlinear function approximation, such as Deep Neural Networks (DNNs)... **As in classical RL, convergence guarantees are generally unavailable under nonlinear function approximation. In practice, stability can be improved by decoupling the target from the online network. For example, following (Mnih et al., 2015), one may use a target network**..."

**这是论文里 neural 稳定性的全部讨论**——只有一句话，给的解决方案是 "用 target network"。

### 2.3 §5 conclusion 把 DNN 稳定性列为 future work

逐字引用 page 23：

> "extending our methodology to large-scale problems via deep neural approximators presents both **opportunities and challenges** in expressiveness and **stability**."

**——教授自己说的：DNN 下的 stability 是 future work**。

### 2.4 LQR 实验是 toy 设定

§4.1 实验只用 `Φ(s, a) = (s², a², 1)`——3 个 quadratic basis。**没有 neural network 实验**。

---

## 3. 我们的实验印证了这就是真问题

三次 Path B 实验（全部 50k QUICK seed=0）都印证了 neural critic 下的不稳定：

| 配置 | \|grad_s Q\| 末态 | TIR |
|---|---|---|
| Path B no clip raw reward | 420（单调暴涨）| 59.69% |
| Path B clip=10 raw reward | 10（撞 cap）| 54.19% |
| Path B FULL 100k raw + 错位 clip | 925 | 5.46% |
| Path B clip=10 normalize reward | 10（撞 cap）| 41.29% |
| Path B clip=0.5 normalize reward | 0.5（撞 cap）| 14.04% |

**所有 clip 设定都让 \|grad_s Q\| 撞到 cap**——这意味着 critic 内部有一个**结构性的"$\nabla_s Q$ 应该无界增长"的压力**。clip 只是治标。

**对照 baseline**（不需要 clip）：TIR 63.65%（50k raw）/ 74.29%（50k normalize）——**离散 Bellman target 在 neural critic 下天然稳定**，因为它只依赖 Q 的**值**，不依赖 Q 的**梯度**。

---

## 4. 为什么这件事是有研究价值的

**论文 Yuhua Zhu PhiBE-Q (Ren & Zhu, 2025) 是一个数学完整、但 deep-RL deployment 缺乏稳定性的框架**。我们的工作填补这个 gap。

理论侧面的工作（教授擅长）：
- Lemma 3.1 在 neural network 下需要什么 functional class assumption？
- 在什么 architecture 约束下 Lemma 3.1 仍成立？（猜测：Lipschitz neural network）

应用侧面的工作（我们做的）：
- 在 medical control (T1D glucose) 这种 stakes-high、reward-asymmetric setting 下验证可行性
- 这是教授论文里没碰过的 application domain

**两边合起来是一个完整的 paper**——理论 + 应用。

---

## 5. 候选解决方案

按"理论严谨度 × 工程代价"排：

### 5.1 Spectral Normalization on critic（我们当前选项）

- **机制**：把 Critic 每一层 Linear 用 `torch.nn.utils.spectral_norm` 包装，保证每层 Lipschitz ≤ 1，整个网络 \|∇_s Q\| ≤ 常数
- **理论价值**：直接对应 Lemma 3.1 在 neural network 下需要的 Lipschitz 假设
- **代价**：~5 行代码
- **来源**：WGAN-SN (Miyato et al., 2018)、SAC 等用过

### 5.2 Gradient Penalty (WGAN-GP style)

- **机制**：critic loss 加 $\lambda_\text{gp} \cdot (\|\nabla_s Q\|_2 - 1)^2$，软约束 Lipschitz
- **理论价值**：跟 5.1 同理，但是 soft constraint
- **代价**：~5 行 + 一个超参数
- **来源**：WGAN-GP (Gulrajani et al., 2017)

### 5.3 Baseline-Anchored Path B

- **机制**：target = baseline_target + λ · path_b_correction（连续时间修正项作为 perturbation）
- **理论价值**：保留 baseline 的稳定性，逐步引入 CT 约束
- **代价**：~10 行 + 一个超参数
- **风险**：跟 Approach C 思路接近，C 已经验证失败

### 5.4 Layer Normalization in critic

- **机制**：每层 Linear 后加 LayerNorm——经验性平滑梯度
- **代价**：~5 行
- **理论价值**：相对弱，但 zero hyper-parameter

---

## 6. 这周的具体动作

**第 1 步**：实现 Spectral Normalization 版本的 critic（5.1）
**第 2 步**：跑 Path B + normalize reward + spectral norm critic（50k QUICK seed=0）
**第 3 步**：根据结果决定下一步——
- 如果 TIR ≥ baseline+normalize（74.29%）：spectral norm 成功，进 FULL 3 seed 验证
- 如果 TIR 60-74%：spectral norm 部分有效，考虑加 gradient penalty
- 如果 TIR < 60%：尝试 5.2 或 5.3

**关键比较**：
- Baseline + normalize 74.29%（已知）
- Path B + normalize + spectral norm ???

---

## 7. 给教授的简短陈述（备份)

> "您的 PhiBE-Q 论文在 linear basis 下有完整理论保证（Theorem 3.2/3.3）。§3.3.2 第二 remark 提到 DNN extension 没有 convergence guarantee、§5 conclusion 列为 future work。我们的实验印证了这一点——直接把 Q 换成 deep neural network 后，∇_s Q 在训练中单调增长，TIR 降到 PID 水平甚至更低。这正好是您论文留下的 open problem。
>
> 我们的方法：把 Lemma 3.1 在 linear basis 下用的 Lipschitz 约束显式 enforce 到 neural critic 上——通过 spectral normalization。这相当于把您论文里 c_1, c_2, c_3 这些 basis function 自带的 Lipschitz bound，作为 architecture constraint 强加到 neural network 上。"

---

**这份文档让任何会话能立刻定位"我们到底在做什么"。**
