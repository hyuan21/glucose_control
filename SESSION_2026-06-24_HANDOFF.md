# Session 2026-06-24 — 接续文档（换设备用）

> **给新设备 Claude 的开场白**：
> 你好。我接续之前的科研项目（offline RL 做 type-1 diabetes 血糖控制）。
> 这份文档是**当前唯一有效的进度**。请只读这一份就够了——
> 之前还有 HANDOFF.md / SESSION_2026-05-14.md / SESSION_2026-06-11.md /
> PATH_B_MATH.md / RESEARCH_POSITION.md 等历史文档，但**它们记录的方向已经
> 被推倒，不要再参考、不要再提**（详见下面第 4 节"已废弃，不要碰"）。
> 读完本文档后，用中文跟我确认你的理解，然后等我给下一步指示。
> 讨论用中文；代码注释和给教授的写作用英文。

---

## 1. 一句话现状

我们从 Emerson 2023 的 offline RL baseline（TD3-BC）出发，**只做一处改造：
替换 critic 的 target function**，把教授连续时间公式右边的期望项做泰勒展开，
于是 target/loss 里出现了 PhiBE 的三项 `V`、`∇ₛV`、`∇ₛ²V`。
**当前的研究任务是：找到一个合适的 optimization 方法，去求解这个同时含
这三项的目标函数。** 仅此而已，没有别的支线。

---

## 2. 数学进度（已经推清楚的部分）

### 2.1 起点：教授的连续时间公式

$$
V(s_t) = r\,\Delta t + e^{-\beta\Delta t}\,\mathbb{E}\big[V(s_{t+1}) \mid s_t\big]
$$

（Q 版结构相同：`Q(s_t,a_t) = r·Δt + e^{-βΔt}·E[Q(s_{t+1},a_{t+1})|s_t,a_t]`。）

### 2.2 唯一的改造：对期望项做泰勒展开

把 `V(s_{t+1})` 在 `s_t` 处展开到二阶，记 `Δs = s_{t+1} - s_t`：

$$
V(s_{t+1}) \approx V(s_t) + \Delta s^\top \nabla_s V(s_t)
+ \tfrac12 \Delta s^\top \nabla_s^2 V(s_t)\,\Delta s
$$

取条件期望，定义 drift / diffusion 的样本估计：

$$
\mu = \frac{\mathbb{E}[\Delta s]}{\Delta t},\qquad
\Sigma = \frac{\mathbb{E}[\Delta s\,\Delta s^\top]}{\Delta t}
$$

代回、令 Δt 一阶，得到 **PhiBE 残差核心方程**：

$$
\beta\,V(s) = r + \mu^\top \nabla_s V(s) + \tfrac12\,\mathrm{tr}\!\big(\Sigma\,\nabla_s^2 V(s)\big)
$$

**三项第一次同时出现**：`V`、`∇ₛV`、`∇ₛ²V`。

### 2.3 目标函数（这是我们要 minimize 的对象）

定义每个样本点的残差：

$$
\mathcal{R}_\theta(s) = \beta V_\theta(s) - r
- \mu^\top \nabla_s V_\theta(s)
- \tfrac12 \mathrm{tr}\!\big(\Sigma \nabla_s^2 V_\theta(s)\big)
$$

目标函数：

$$
f(\theta) = \mathbb{E}_{s\sim\mathcal D}\big[\mathcal{R}_\theta(s)^2\big]
$$

三项各自：`V_θ` 标量（系数 β）；`∇ₛV_θ` 向量（跟 μ 内积）；
`∇ₛ²V_θ` 矩阵/Hessian（跟 Σ 做 trace）。
注意 **μ、Σ 是数据估的常数，对 θ 无依赖**；含 θ 的只有这三项。

### 2.4 对比：跟传统 RL 的本质区别（教授强调的点）

- 传统 RL 目标：`min_θ f(V_θ(s))` —— loss 只通过**输出值**依赖 θ。
- PhiBE 目标：`min_θ f(V_θ, ∇ₛV_θ, ∇ₛ²V_θ)` —— loss 通过**值 + 输入一阶导
  + 输入二阶导**依赖 θ。

### 2.5 为什么传统 optimizer 求不了（三层困难，已分析清楚）

1. **混合高阶导**：算 `∇_θ f` 需要 `∇_θ∇ₛV_θ`（混合二阶）和 `∇_θ∇ₛ²V_θ`
   （混合三阶）——传统 RL 工作流里根本没有的对象，每步要 build 高阶计算图。
2. **不受控**：`∇ₛV_θ`（斜率）、`∇ₛ²V_θ`（曲率）没有任何东西直接约束其大小，
   可以任意大 → bootstrapping 时被越推越大 → 发散。
3. **二阶项病态**：ReLU 网络二阶导几乎处处为 0（第三项消失，退化成一阶）；
   光滑激活下二阶导震荡、量级难控。

---

## 3. 下一步该做什么

**下一步任务（教授给的方向）**：找到能 minimize 上述含三项目标的"合适的
optimization 方法"。

> **注意**：上一次会话 Claude 已经在脑子里整理过几条候选解法思路（线性
> 参数化 / 迭代线性化 / 约束优化），但**用户明确说这些思路先不要展开**——
> 要先跟教授对齐"合适的 optimization 到底指什么"再决定。
> 所以新设备 Claude：**不要主动推销任何具体解法**，等用户/教授定方向。

读完本文档后，新设备 Claude 应该：
- 用中文复述对第 2 节数学的理解（三项怎么来的、目标函数长什么样、为什么难）；
- 然后**停下来等用户的下一步指示**，不要自作主张选方法。

---

## 4. 已废弃，不要碰（重要）

以下方向在之前会话里出现过、但**已被用户明确推倒**，新设备 Claude
**不要再参考、不要再提、不要基于它们建议**：

- **Path B 的全部 7 个变体实验**（neural critic runaway、clip、spectral_norm、
  linear_basis、reward normalization 等）—— 已推倒。
- **方法对比四件套**（Galerkin / 高阶有限差分 / PINN policy iteration /
  shift-operator stabilization）—— 上一次会话搜过文献、读过教授新论文
  Optimal-PhiBE 的代码，但**用户说"这些思路先不要提了，太乱"**，全部搁置。
- **HJB 正则化（Approach C）、Approach A 等价性** 等更早的历史 —— 与当前线无关。

当前唯一有效的认知就是第 1-3 节：**Emerson baseline → 只改 target →
泰勒展开出三项 → 找合适 optimization 求这三项**。

---

## 5. 代码与环境状态（仅供参考，当前任务以数学讨论为主）

- 仓库：`github.com/hyuan21/glucose_control`，本地有大量未 commit 改动。
- 环境：本地 conda `offline-glucose`，python 全路径
  `/c/ProgramData/miniconda3/envs/offline-glucose/python.exe`（Git Bash 里
  conda 没初始化，直接用全路径跑）。RTX 5070 Ti 不兼容 PyTorch 1.7，**CPU 跑**。
- 项目目录：`/c/Users/Administrator/Documents/Claude/Projects/Glucouse Control/Research Project/offline-glucose`
  （注意拼写是 "Glucouse"）。
- **当前阶段是数学/方向讨论，暂时不需要动代码。** 等方向跟教授对齐后再说。

---

## 6. 用户偏好（沿用）

- 中文讨论数学和工程；代码注释 + 给教授的写作用英文。
- 喜欢"先讲清机制再下结论"；提问尖锐，要诚实回答不要搪塞。
- 一步一步来，不喜欢一次被灌大量信息，也不喜欢在没确认方向时就堆细节。
- 给一个推荐 + 理由，让用户决定接受/否定；不要每次列一堆选项。
- **当用户说"推倒""不要提了"时，严格执行，不要把废弃内容又带回来。**

---

**结尾**：新设备 Claude，读完请用中文向用户复述第 2 节的数学理解，
然后等指示。祝 travel 顺利。— 2026-06-24 会话 Claude 留
