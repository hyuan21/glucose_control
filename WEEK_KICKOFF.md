# Week Kickoff — 教授新指示（2026-04-30 之后）

> **用法**：把这个文件的全部内容**直接粘贴到新会话的第一条消息里**，
> 然后让 Claude 读完后向你复述理解、并告诉你下一步该做什么。
>
> Claude 会从这一刻开始接手本周的研究工作。

---

## 给新会话 Claude 的开场白（直接复制粘贴）

```
你好。我们之前在另一个会话里完成了 Approach A（连续时间 value function 改造，
与论文 baseline 数学等价）和 Approach C（HJB 一阶正则的三个版本，全部失败）。
完整的项目历史和代码状态在 HANDOFF.md 里，请先读它。

读完 HANDOFF.md 后，再读这份 WEEK_KICKOFF.md（你正在看的这一份）。
它包含教授本周给的新指示和我对指示的初步理解。

读完两份文档后请：
(1) 用一段话向我复述你对项目当前状态的理解；
(2) 用一段话向我复述教授本周两条指示的含义；
(3) 列出你认为还需要跟我澄清的具体问题，再开始动手。
```

---

## 我（Claude，前一个会话）对教授指示的初步整理

教授给的两条指示（用户原话）：

**指示 1**：
> 我们通过换算之后教授需要我们寻找一个合理的 pair (r, β) 在使模型运行的更好

用户进一步澄清：**r 不是 reward，而是折扣因子 `r = e^(-β·Δt)`** —— 教授把它叫做 "rate"。

所以指示 1 的含义最可能是：

- 联合搜索 (r, β) 的最优配对
- r 和 β 通过 `r = e^(-β·Δt)` 关联
- 给定 Δt 时两者只有一个自由度，所以教授可能想让我们**解放 Δt** 也作为变量
- 或者教授想用 (r, β) 这两个有物理意义的参数取代论文的 γ=0.99，做 grid search

**待跟用户确认（开干前必须问）**：

1. Δt 是固定为 3 min 还是也可调？（如果 Δt 固定，r 和 β 互为函数，只是单变量搜索）
2. "运行的更好"具体指哪个指标？TIR / TBR / Magni risk / 综合？
3. 搜索范围怎么定？β 的物理上界 / 下界是什么？
4. 用 QUICK 模式（50k × 1 seed，每点 ~10 min）还是 FULL 模式（100k × 3 seed，每点 ~3h）跑 grid？

**指示 2**：
> 对于 function 中的 E[Q(s_(t+1), a_(t+1)) | s_t, a_t = π(s,a)] 这一项做一个泰勒展开，
> 然后对这个展开进行求解

**关键点**：

- 这次教授给的是 **Q 函数**（不是之前那个 v 函数）
- 注意条件期望写作 `E[...| s_t, a_t = π(s,a)]`，意思是"给定当前状态-动作，下一个状态-动作"
- `a_(t+1) = π(s_(t+1))` 是 actor 在下一步选的动作
- 教授要求展开**并求解**

**我（前任 Claude）认为教授可能的"求解"指的有四种可能**：

1. **求 Q* 的闭式解**：从 HJB 推出 optimal value function 的解析表达
2. **求 optimal policy 的闭式解**：从 HJB 出发推 π* 的某种描述
3. **数值解**：把展开后的 PDE 用 finite differences / RBF 之类求解
4. **HJB residual minimisation**：跟之前 Approach C 一样把残差作为 critic 正则项

**待跟用户确认**：

1. 教授是否给了任何提示说"求解的形式"是什么样的？例如他写没写公式？
2. 教授是否在白板上推过中间步骤？我们能拿到照片吗？
3. 教授对 Approach C 失败那部分的反馈是什么？这次的"求解"是否就是 C 路线的延伸？

---

## 数学准备 — Q-version HJB 推导（前任 Claude 提前推给你）

之前我们推的 v-version HJB：

$$\beta v(s) = r(s, a) + \nabla_s v \cdot \mu + \tfrac{1}{2}\,\mathrm{tr}(\nabla^2_s v \cdot \Sigma)$$

这次教授要的是 **Q-version**。从 Q 的连续时间公式开始：

$$Q(s_t, a_t) = r \cdot \Delta t + e^{-\beta \Delta t} \cdot \mathbb{E}\big[Q(s_{t+1}, a_{t+1}) \mid s_t, a_t = \pi(s_t)\big]$$

设 actor 是确定性的 $a_{t+1} = \pi(s_{t+1})$，则期望是关于 $s_{t+1}$ 的（动作完全由策略决定）。

**对 Q 在 $(s_t, \pi(s_t))$ 处做泰勒展开**：

$$
Q(s_{t+1}, \pi(s_{t+1})) \approx Q(s_t, \pi(s_t)) + \nabla_s Q^\top \cdot (s_{t+1} - s_t) + \nabla_a Q^\top \cdot (\pi(s_{t+1}) - \pi(s_t)) + \tfrac{1}{2}(\cdots)
$$

注意 $\pi(s_{t+1}) - \pi(s_t) \approx J_\pi(s_t) \cdot (s_{t+1} - s_t)$，其中 $J_\pi$ 是 actor 的 Jacobian。

代入 SDE $ds = \mu dt + \sigma dW$，对随机部分取期望（用 Itô 引理处理二阶项），最终得：

$$
\beta Q(s, a) = r + (\nabla_s Q + J_\pi^\top \nabla_a Q)^\top \mu + \tfrac{1}{2}\,\mathrm{tr}\big[(\nabla^2_s Q + \text{cross terms} + J_\pi^\top \nabla^2_a Q J_\pi) \cdot \Sigma\big]
$$

**这比 v-HJB 复杂的关键在于多了 actor Jacobian $J_\pi$**。Q 的总变化沿状态演化 = "状态本身的变化" + "策略响应状态变化的变化"。

**这正是教授想让我们推导的 + 求解的内容**。具体是不是这个形式，要等用户/教授确认。

---

## 跟用户提议的本周工作流（Claude 自己决定要不要采纳）

按依赖顺序，先理论后实验：

**第 1-2 天：与用户共同推导指示 2 的 Q-HJB**

- 严格写出 Q 函数的连续时间形式（确认条件期望、Jacobian 项）
- 用 Itô 引理推导一阶 / 二阶展开
- 看能否化简出有意义的"解"（最优策略的闭式表达？最优 Q 函数的某种结构？）
- 用 LaTeX 写出推导，发给教授确认方向对了再继续

**第 3 天：与用户讨论指示 1 的 grid search 设计**

- 确认 r/β/Δt 的关系和搜索空间
- 设计实验矩阵（建议先 QUICK 跑稀疏 grid，再 FULL 跑密集）

**第 4-5 天：实验**

- 运行 grid search，每个 (r, β) 跑 QUICK
- 找出 top-3 候选点，FULL mode 验证
- 出表 + 出图

**第 6-7 天：整合**

- 写本周报告（数学推导 + 实验结果）
- 准备下周与教授的 meeting

---

## 当前代码状态（继承自上个会话）

GitHub: https://github.com/hyuan21/glucose_control
最新 commit: `a68836c` "Expand HANDOFF.md to a complete cross-device conversation log"

**核心文件**：
- `TD3_BC.py` — 论文 baseline（不动）
- `TD3_BC_ct.py` — A + C 的统一实现，参数化 β/Δt/use_hjb/drift_mode
- `Comparison_CT.ipynb` — train + eval + plot 的 notebook
- `HANDOFF.md` — 完整对话历史交接文档（**首先必读**）
- `WEEK_KICKOFF.md` — 本文档
- `utils/` — 论文工具（不动）

**已知约束**（来自 HANDOFF.md §8 已知坑）：
- RTX 5070 Ti 不兼容 PyTorch 1.7.1，必须 CPU 跑
- pip install -e . 会失败，跳过即可
- jupyter notebook 命令不存在，用 jupyter lab 或 VS Code

---

## 给新 Claude 的几条嘱咐

1. **不要把 Q-HJB 推导草草写完就开始写代码**。先跟用户对一遍数学，确保推导对、解释清楚每一步。教授可能这周就要看推导。

2. **不要把指示 1 的 grid search 跑成 100k × 3 seed × 大 grid**。CPU 上一晚上都跑不完。先用 QUICK 50k × 1 seed × 5-9 个点，定位有希望的区域再加密。

3. **承认指示 2 的"求解"语义不明**。第一次跟用户对话就该问清楚教授到底想要哪种形式的解。

4. **保持对用户洞察的尊重**：用户上周提出的"buffer 多样性是性能瓶颈"这个观察，对本周工作仍然适用 —— 即使我们调出了最优 (r, β)，预期收益也有限。在汇报时不要给用户虚假希望。

5. **复用现有代码结构**：TD3_BC_ct.py 的 use_hjb / drift_mode 这套已经能支持很多实验。不要为了"做新东西"而新建一堆文件。

6. **保持中文交流**：用户偏好中文讨论数学和工程问题。代码注释 + 论文风格写作可以英文。

---

## 用户的偏好（来自上个会话观察）

- 不喜欢长篇 markdown 头条款列，但**容忍**它们用在结构化文档里
- 偏好"先讲清楚机制再讲结论"的解释方式
- 提问通常很尖锐（如 "Q 用教授公式 y 用离散公式吗"），需要诚实回答而不是搪塞
- 习惯一边做一边问，不喜欢被一次性灌输大量信息
- 在每次实验后会问"为什么"，要为这个准备好诊断逻辑

---

**祝本周顺利。前任 Claude 留。**
