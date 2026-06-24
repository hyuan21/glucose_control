# Path B — 数学整理

> **本文目的**：把我们这周从教授指示 2 开始一路推到 Path B 的完整数学逻辑整理在一起。
> 不放代码、不放实验数字、不放工程细节——纯数学和推理。
> 给你出差前消化一遍用的。
>
> 下次和教授见面时，可以从这份文件里挑要点做 1 页 memo。

---

## 1. 起点：教授给的公式

教授给的连续时间 value function：

$$
v(s_t) = r \cdot \Delta t + e^{-\beta \Delta t} \cdot \mathbb{E}[v(s_{t+1}) \mid s_t]
$$

这次（指示 2）她给的是 Q 版本：

$$
Q(s_t, a_t) = r \cdot \Delta t + e^{-\beta \Delta t} \cdot \mathbb{E}[Q(s_{t+1}, a_{t+1}) \mid s_t, a_t]
$$

**符号约定**：

- $s_t \in \mathbb{R}^{11}$ 是状态（血糖滑窗 8 维 + BG_now + MOB + IOB）
- $a_t \in \mathbb{R}$ 是动作（basal insulin rate）
- $r$ 是 reward rate（每分钟的 reward，**不是**累积 reward）
- $\beta > 0$ 是连续时间衰减率，单位 1/min
- $\Delta t$ 是采样间隔，CGM 是 3 min
- $a_{t+1} = \pi(s_{t+1})$ 因为我们的 actor 是确定性的（TD3）

**关键观察**：在 $\beta = -\ln(0.99)/3$、$\Delta t = 3$ 下，$e^{-\beta\Delta t} = 0.99$，
和论文 baseline 的 $\gamma$ 数值上完全一致。所以教授公式跟 baseline 在数值层面**等价**——
真正"换"了什么取决于我们怎么处理右边那个 $\mathbb{E}[Q(s_{t+1}, a_{t+1}) | s_t, a_t]$。

---

## 2. Reward-rate 约定（HANDOFF §5.2 的传承）

教授原公式两边都除以 $\Delta t$，记 $\tilde Q := Q / \Delta t$：

$$
\tilde Q(s_t, a_t) = r + e^{-\beta\Delta t} \cdot \mathbb{E}[\tilde Q(s_{t+1}, a_{t+1}) \mid s_t, a_t]
$$

形式完全一样——所有项都成比例除了 $\Delta t$，方程结构不变。
**critic 实际学的是 $\tilde Q$**，本质就是 reward-rate 单位下的 Q。

**这一约定为什么重要**：
- A v1 错误版本曾经把 $r \cdot \Delta t$ 直接写进 target，导致 critic target 数值放大 $\Delta t = 3$ 倍，
  critic loss 放大 9 倍，BC adaptive lambda $\lambda = \alpha / |Q|_\text{mean}$ 被压扁，
  actor 学坏（HANDOFF §6.1）
- A v2 切到 rate 约定后，target 跟 baseline 同尺度，问题消失（HANDOFF §6.2）
- Path B 沿用这个约定。从此**公式里看到的 $r$ 都是 rate，不乘 $\Delta t$**

---

## 3. 教授给的泰勒展开（原话照抄）

教授白板上写的泰勒展开（关于 $\mathbb{E}[Q(s_{t+1}, a_{t+1}) | s_t, a_t]$）：

$$
\mathbb{E}\Big[
Q(s_t, a_t)
+ (s_{t+1} - s_t)^\top \nabla_s Q(s_t, a_t)
+ (a_{t+1} - a_t)^\top \nabla_a Q(s_t, a_t)
+ \tfrac{1}{2}(s_{t+1} - s_t)^\top \nabla_s^2 Q(s_t, a_t)(s_{t+1} - s_t)
+ \cdots
\Big]
$$

**教授停在这里——后面没继续写**。这一节是整篇推导的**关键解读节点**。

### 3.1 教授停在这一步的解读

如果教授继续往下推，"自然"的下一步会是：
- 把 $\Delta a = a_{t+1} - a_t$ 用 $\pi$ 展开成 $J_\pi \cdot \Delta s$（链式法则）
- 写出含 $\nabla_a Q^\top J_\pi$、$J_\pi^\top \nabla_{aa}^2 Q J_\pi$ 等的复杂二阶项

**教授没有走这条路**——会话早期我（Claude）自作主张推了这条路，被你和教授明确否定。
教授的意思是：
- $\Delta s$、$\Delta a$ 是两个**并列**的增量量
- 展开形式只关心 $\nabla_s$、$\nabla_a$、$\nabla_s^2$ 这三个**对应教授白板上写出来的**导数
- 不要再继续展开 $\Delta a$，让它保持原状

### 3.2 简记

记 $\Delta s := s_{t+1} - s_t$，$\Delta a := a_{t+1} - a_t$。教授公式简记为：

$$
\mathbb{E}[Q_{t+1}] \approx Q(s_t, a_t)
+ \mathbb{E}[\Delta s]^\top \nabla_s Q
+ \mathbb{E}[\Delta a]^\top \nabla_a Q
+ \tfrac{1}{2}\mathbb{E}[\Delta s^\top \nabla_s^2 Q \, \Delta s]
$$

所有 $\nabla Q$ 在 $(s_t, a_t)$ 处取值；所有期望都是条件期望 $\mathbb{E}[\cdot | s_t, a_t]$。

---

## 4. Off-policy trick 的应用

教授后续邮件（指示 2 的关键澄清）：

> *"No, use the same trick from off-policy learning for the Q defined by the Bellman equation in RL, but apply it to this Q function. Give it some thoughts."*

**最终解读**（你确定的版本）：

- "the same trick from off-policy learning" = TD3-BC 现有的 off-policy 架构（target net + replay buffer + sample-based TD error）
- "this Q function" = 教授给的连续时间 Q（第 1 节那个公式）
- "apply it" = **把 critic target 一行换成连续时间 Q 公式给出的 target，其他不动**

### 4.1 为什么 trick 能直接用

标准 TD3-BC 的 critic target：

$$
\text{target}^{\text{TD3-BC}} = r + e^{-\beta\Delta t} \cdot Q_\text{target}\big(s', \pi_\text{target}(s')\big)
$$

——这个 target **只依赖 buffer 里的 sample $(s, a, s', r)$ 和 target network**，不依赖
生成数据的 behavior policy。所以是 off-policy 的。

我们把右边的 $Q_\text{target}(s', \pi_\text{target}(s'))$ **换成**教授泰勒展开式给出的近似——
其他不动，**架构层的 off-policy 性质自动保留**：

- 期望 $\mathbb{E}[\Delta s]$、$\mathbb{E}[\Delta a]$、$\mathbb{E}[\Delta s^\top \nabla_s^2 Q \,\Delta s]$
  用 buffer 里的**单个 sample**估
- $a_{t+1}$ 用 target actor 在 $s'$ 上的输出，加 TD3 smoothing noise
- 所有 $\nabla Q$ 用 target critic 算

数据从哪个 behavior 来都不影响 target 公式——这正是 "the same trick"。

### 4.2 用 sample 估每一项

| 数学项 | 用样本估计 |
|---|---|
| $\mathbb{E}[\Delta s \| s_t, a_t]$ | $s' - s$（buffer 直接取） |
| $\mathbb{E}[\Delta a \| s_t, a_t]$ | $\pi_\text{target}(s') - a$（target actor 算） |
| $\mathbb{E}[\Delta s^\top \nabla_s^2 Q \, \Delta s \| s_t, a_t]$ | $\Delta s^\top \nabla_s^2 Q \, \Delta s$（直接用样本 $\Delta s$） |

注意 $\nabla Q$ 等导数在数据点 $(s, a)$ 处取值——这里 $a$ 是 **buffer 里的 action**（行为策略生成的），
不是 $\pi(s)$。展开中心点就是 buffer sample。

---

## 5. Path B 的 critic target 最终形式

把以上代入起点公式：

$$
\boxed{\;\text{target} = r + e^{-\beta\Delta t} \Big[
Q_\text{target}(s, a)
+ \Delta s^\top \nabla_s Q_\text{target}
+ \Delta a^\top \nabla_a Q_\text{target}
+ \tfrac{1}{2}\Delta s^\top \nabla_s^2 Q_\text{target} \, \Delta s
\Big]\;}
$$

其中：
- $\Delta s = s' - s$，$\Delta a = \pi_\text{target}(s') - a$
- 所有 $\nabla Q$ 在 $(s, a)$ 处取值，用 target critic 算
- target critic 通过 TD3 标准 soft update 跟踪 online critic

### 5.1 critic loss

$$
L_\text{critic} = \frac{1}{|B|} \sum_i \Big(Q_\text{online}(s_i, a_i) - \text{target}_i\Big)^2
$$

target 整体 `detach()`，梯度只更新 online critic 参数。

### 5.2 actor update

**actor update 完全不变**——还是 TD3-BC 原版的

$$
L_\text{actor} = -\lambda \cdot Q_\text{online}(s, \pi(s)) + \|\pi(s) - a_\text{demo}\|^2,
\qquad \lambda = \frac{\alpha}{|Q|_\text{mean}}
$$

我们只动了 critic target 一行。

---

## 6. Path B 与已有方案的对照

把所有 critic target 公式放在一起：

| 方案 | critic target |
|---|---|
| **Baseline** (论文，离散 Bellman) | $r + \gamma\,Q_\text{target}(s', \pi_\text{target}(s'))$，$\gamma=0.99$ |
| **Approach A v2** (reward-rate) | $r + e^{-\beta\Delta t}\,Q_\text{target}(s', \pi_\text{target}(s'))$，默认参数下数值等价 baseline |
| **Approach C** (HJB 残差正则) | A 的 target 不变；critic loss 额外加 $\lambda_\text{hjb}\|\beta Q - r - \nabla_s Q\cdot\hat\mu\|^2$ |
| **Path B** | $r + e^{-\beta\Delta t}[Q + \Delta s^\top\nabla_s Q + \Delta a^\top\nabla_a Q + \tfrac{1}{2}\Delta s^\top\nabla_s^2 Q \Delta s]$ |

**关键区别**：
- A 跟 baseline 数学等价（默认参数），换的是参数化方式
- C 是把 HJB 残差当**辅助正则**加到 critic loss，target 还是 A 的形式
- **Path B 是把 target 本身换成泰勒展开作为主体**——不是辅助，是主体

---

## 7. 推导中走过的弯路（避免重复）

记录在 SESSION_2026-05-14.md 第 3 节，简略再列：

**弯路 1**：把 $\Delta a$ 用 $\pi$ 展开成 $J_\pi \Delta s$，引入 actor Jacobian。
**否定理由**：教授白板停在 $\nabla_s^2 Q$，没继续展开 $\Delta a$；
邮件 "No" 明确否定这条路。

**弯路 2**：把指示 2 理解成"实现教授论文 PhiBE-Q 的 Algorithm 1"。
**否定理由**：教授原话 "this is for later"——论文是以后的事，这周不该做。

**弯路 3**：在 v1 实现里加二阶 $\Sigma$、$\hat\Sigma$ 估计、Hessian × Σ 项等论文里的内容。
**否定理由**：教授公式里**不含** $\mu, \Sigma$ 这些 SDE 符号，
只含 $\nabla_s Q$、$\nabla_a Q$、$\nabla_s^2 Q$ 这些**梯度本身**。
我们停在梯度形式，不替换成 drift / 扩散。

---

## 8. 几个值得跟教授讨论的开放问题

这些问题在 SESSION_2026-05-14.md 第 5 节都有展开，本节是要点提炼。

### 8.1 $\nabla_a Q^\top \Delta a$ 这一项的含义

教授白板写了这一项，但没解释它的物理意义。我们的实现里 $\Delta a = \pi_\text{target}(s') - a$，
其中 $a$ 是 buffer 里的 behavior action。这一项可以解读为：
- "如果按 target policy 走，下一步动作会偏离 behavior 多少"
- 它给 critic 提供了一个**间接的 policy improvement 信号**——
  在 behavior 与 target 不一致的方向上扣分

这个解读是否符合教授的意图？还是她有别的想法？

### 8.2 $\nabla_s^2 Q$ 二阶项在 ReLU critic 下退化

ReLU 网络的二阶导数几乎处处为 0。我们的 critic 是 256-256-ReLU 的标准 TD3 critic。
这意味着：**实际训练中二阶项 $\tfrac{1}{2}\Delta s^\top \nabla_s^2 Q \, \Delta s \equiv 0$**。

也就是说，Path B 在 ReLU 下退化成"一阶 Path B"：

$$
\text{target}_{\text{effective}} = r + e^{-\beta\Delta t}\big[Q + \Delta s^\top \nabla_s Q + \Delta a^\top \nabla_a Q\big]
$$

教授让我们展开到二阶，但在我们的网络下二阶不出力。**值得讨论**：
- 是不是要换激活函数（GELU / Tanh / Swish）？
- 还是教授其实知道 ReLU 下二阶=0，只是要我们写到这一阶以保持公式的对称性？

### 8.3 (r, β) 这对参数怎么定（指示 1）

教授指示 1 让我们找一对合理的 $(r, \beta)$。"$r$" 这里是 rate = $e^{-\beta\Delta t}$，
不是 reward。我们目前默认 $\beta = -\ln(0.99)/3 \approx 3.35 \times 10^{-3}$ /min，
$\Delta t = 3$ min。

未解决的问题：
- $\Delta t$ 是固定 3 min 还是也作为搜索变量？
- "运行得更好"具体指 TIR、TBR、Magni risk 还是综合？
- $\beta$ 的物理上下界？教授有没有 prior？

---

## 9. 一句话总结

> 教授给的连续时间 Q 公式做泰勒展开，停在 $\nabla_s^2 Q$ 那一步，
> 把这个展开式作为 critic target 的主体（不是辅助正则），
> 用 buffer sample 估期望，用 target critic 算导数，
> 保持 TD3-BC 的整体 off-policy 架构不动。
> 这是教授邮件 "apply the trick to this Q function" 的最朴素落地。

---

**祝消化顺利。**
