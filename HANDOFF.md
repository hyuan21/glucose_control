# HANDOFF — offline-glucose continuous-time branch (full conversation log)

> **目的**：在新设备的 Cowork 上恢复跨数天会话的完整研究上下文。
> Cowork 之间不能直接同步对话历史；这份文档承担这个角色。
> 读完这份后，新会话的 Claude 应当等同于参与了之前的所有讨论。
>
> 最后更新：2026-04-30，方案 C v3 实验完成、给教授的中期报告完成、push 到 GitHub 之后。

---

## 0. 在新设备上怎么用这份文档

**第 1 步**：克隆仓库到新设备。

```bash
git clone https://github.com/hyuan21/glucose_control.git
cd glucose_control
conda env create -f environment.yml
conda activate offline-glucose
python -m ipykernel install --user --name offline-glucose --display-name "offline-glucose"
```

**第 2 步**：在 Cowork（或装了 Claude 扩展的 VS Code）里打开这个文件夹。

**第 3 步**：第一句话告诉新会话的 Claude：

> 「请完整阅读 HANDOFF.md 中的全部内容，包括第 0-12 节。读完后用一段话向我复述：(a) 我们做了什么，(b) 当前的状态卡在哪，(c) 你认为下一步该问我什么。然后我们继续。」

Claude 会读这份文档（约 4000 字），然后用它的判断告诉你卡点在哪、下一步建议。

---

## 1. 项目人物 & 关系

- **用户**：Hanzhang Yuan（袁瀚璋），UCLA 学生，邮箱 hyuan21@ucla.edu
- **教授**：未具名。给了一个连续时间 value function 的公式作为研究方向
- **论文作者**：Harry Emerson 等人（hemerson1@github），2023 J Biomed Inform 论文 "Offline Reinforcement Learning for Safer Blood Glucose Control in People with Type 1 Diabetes"
- **Claude（前任会话）**：完成了方案 A 实现、方案 C 三个版本的实现与诊断、写了中期报告、做了所有的代码改动

---

## 2. 项目原始命题（教授给的）

教授把研究方向从 bolus（餐前大剂量胰岛素）切换到 basal（基础胰岛素），目标算法是 TD3-BC，参考 Emerson 2023。教授给了一个**连续时间 value function** 公式：

$$v(s) = r \cdot \Delta t + e^{-\beta \cdot \Delta t} \cdot \mathbb{E}[v(s') \mid s]$$

并要求**对 $\mathbb{E}[v(s') | s]$ 做泰勒展开**。

教授没明说要做 HJB 正则化或者其他什么特定改动。后续把这条公式落地为算法 + 推泰勒展开后的 HJB 残差作为 critic 正则项，**这两步都是用户和 Claude 一起讨论决定的方向**，不是教授明文指示。这一点很重要，决定了我们对结果的解读。

---

## 3. 与原论文（Emerson 2023）的关系

代码基线是 `github.com/hemerson1/offline-glucose`，4 个 commit 自 2022 年起没动过。我们 fork 了这个仓库的内容到用户自己的 `github.com/hyuan21/glucose_control`，加了三个新文件（见第 4 节）。

论文用的环境：UVA/Padova T1D 仿真器、`adult#1`、PID 作为 demonstrator。

论文 Table 1 报告 TD3-BC 在 adult#1 上 TIR ≈ 78-83%。**我们实测只得到 67.5%**。这 10% 的 gap 至今未完全解释清楚（猜测是论文 patient 选择、reward shaping、或其他没有在 release 代码里的 trick）。

---

## 4. 当前仓库结构

```
glucose_control/                              # ← 用户的 fork
├── TD3_BC.py                                 # 论文原版（baseline，未动）
├── TD3_BC_ct.py                              # ⭐ 新文件 — Approach A + C 的统一实现
├── Comparison_CT.ipynb                       # ⭐ 新 notebook — 训练 + 评估 + 出表 + 出图
├── HANDOFF.md                                # ⭐ 你正在读这份
├── BCQ.py / CQL.py / SAC_RNN.py              # 论文里其他算法，CT 分支不用
├── Offline_RL_Comparison.ipynb               # 论文原 notebook，被 Comparison_CT.ipynb 取代
├── utils/                                    # 论文工具函数（未动）
│   ├── general.py        # PID_action, calculate_risk, calculate_bolus, is_in_range
│   ├── parameters.py     # create_env, get_params  ← state 维度定义在这
│   ├── data_collection.py
│   ├── data_processing.py # unpackage_replay, get_batch  ← 11 维 state 的拼接逻辑
│   ├── evaluation.py     # test_algorithm, create_graph
│   └── pid_grid_search.py
├── environment.yml                           # 论文 conda 环境（PyTorch 1.7.1+cu101）
└── .gitignore                                # ← 已忽略 Models/, Replays/, *中期报告*.docx

不在仓库里（gitignore 排除）：
├── Models/                                   # 训练好的 .pt 权重，约 25 MB（重训可生成）
├── Replays/                                  # PID demonstrator buffer，约 50 MB（重跑 cell 3 可生成）
└── offline-glucose-中期报告.docx              # 给教授的中期报告（敏感，本地保留）
```

---

## 5. 数学基础（讲给教授的版本）

### 5.1 论文离散 Bellman

$$Q(s, a) = r + \gamma \cdot Q(s', \pi(s')), \quad \gamma = 0.99$$

代码：`target_Q = reward + done * self.gamma * target_Q`。

### 5.2 教授连续时间 value function

$$v(s) = r \cdot \Delta t + e^{-\beta \cdot \Delta t} \cdot \mathbb{E}[v(s') | s]$$

把它转换成代码必须解决"reward 是 per-step 还是 per-minute"。我们最终选择 **reward-rate 形式**（v 与 v' 都除以 Δt）：

$$\tilde{v}(s) = r + e^{-\beta \cdot \Delta t} \cdot \mathbb{E}[\tilde{v}(s') | s]$$

代码：`target_Q = reward + done * self.discount * target_Q` 其中 `self.discount = exp(-β·Δt)`。

**默认参数**：β = -ln(0.99)/3 ≈ 3.35e-3 /min, Δt = 3 min → discount = 0.99 = γ（论文）。**所以 A 方案在默认参数下数学上严格等价于 baseline**。

### 5.3 泰勒展开 → 一阶 HJB

假设状态满足 SDE: $ds = \mu(s,a) dt + \sigma(s,a) dW$，对 $\mathbb{E}[v(s')|s]$ 用 Itô 引理展开，代回原式让 Δt → 0：

$$\beta \cdot v(s) = r + \nabla v^\top \mu + \tfrac{1}{2}\text{tr}(\nabla^2 v \cdot \Sigma)$$

我们用一阶版（丢掉 Σ 项）：

$$\beta \cdot v = r + \nabla v^\top \mu$$

作为 critic loss 的软正则项加进去。

### 5.4 物理状态的解释（关键）

11 维 condensed state 的布局是：

```
索引 0-7: 9 维 BG 历史的前 8 个滞后快照（4h 前、3.5h 前、…、0.5h 前）
索引 8:   BG_now（当前血糖）        ← 物理动力学
索引 9:   MOB（餐食 on board）       ← 物理动力学
索引 10:  IOB（胰岛素 on board）      ← 物理动力学
```

**只有 [8, 9, 10] 三维真满足 SDE**。其他 8 维只是滑窗位移，HJB 不该作用在它们上面。

---

## 6. 全部实验记录（按时间顺序）

### 6.1 A v1（错误版本，仅在内部 commit 历史里出现，未保留）

逐字翻译教授公式：`target_Q = reward * dt + done * exp(-β·dt) * target_Q`

**结果**：QUICK 50k × 1 seed，TIR = 64.10，ΔTIR = -3.6%。

**诊断**：reward 多乘 Δt = 3，target_Q 数值放大 3 倍，critic loss 放大 9 倍，BC 项相对权重被压扁，actor 学坏。

### 6.2 A v2（修复版，当前 TD3_BC_ct.py 的默认 mode）

reward-rate 形式：`target_Q = reward + done * exp(-β·dt) * target_Q`

**结果（FULL 100k × 3 seed）**：

| Algorithm | TIR (%) | TBR (%) | Magni | ΔTIR vs baseline |
|---|---|---|---|---|
| PID | 59.50 ± 0.00 | 0.00 ± 0.00 | 4.19 ± 0.00 | — |
| **Baseline** | **67.48 ± 3.16** | 0.05 ± 0.07 | **3.88 ± 0.09** | — |
| **A** | **67.48 ± 3.16** | 0.05 ± 0.07 | 3.88 ± 0.09 | **+0.00** |

完全等价（数学保证：默认参数下 self.discount = self.gamma 浮点位级相同；同 seed、同 buffer、同初始化 → forward/backward 逐位一致）。

### 6.3 C v1（FD drift, 全 11 维 state）

`hjb_residual = β·Q - r - ∇Q · (s'-s)/Δt`，对全 11 维计算。

**结果（QUICK 50k × 1 seed）**：TIR = 57.92, ΔTIR = -11.6%, HJB Loss 30-46 全程不下降。

**诊断**：8 维 BG 滑窗历史的 (s'-s)/Δt 是几何位移、不是物理 drift，污染了 HJB 残差。

### 6.4 C v2（FD drift, 仅物理维度 [8, 9, 10]）

只对 [BG_now, MOB, IOB] 算 ∇Q·μ̂。代码改动 5 行。

**结果（QUICK 50k × 1 seed）**：TIR = 64.44, ΔTIR = -5.1%, HJB Loss 30-50 仍震荡。

**诊断**：维度问题缓解，但 (s'-s)/Δt 在 Δt=3 下被 diffusion 噪声 σ√Δt 严重污染。SNR 太低。

### 6.5 C v3（DriftNet, 物理维度，带 5000 步 warmup）

引入 256-256-3 MLP `μ_φ(s,a)` 拟合 (s'_phys - s_phys)/Δt。前 5000 步只训 DriftNet 不开 HJB。

**结果（FULL 100k × 3 seed）**：TIR = 63.67 ± 0.75, ΔTIR = -3.81, HJB Loss 13-40 震荡, **Drift Loss 极低 0.0003-0.006**。

**反直觉的诊断**：Drift Loss 极低反而是坏信号。256-256 容量足够把 FD 信号（噪声主导）几乎完美拟合，所以"平滑作用"没发生。神经网络学到的也是噪声。

### 6.6 C 失败的最终诊断（4 层）

每改一次 ΔTIR 改善但永远到不了正面（-11.6 → -5.1 → -3.8），强烈说明问题不在实现而在框架：

1. **Drift 估计的内在噪声**：(s'-s)/Δt 被 σ√Δt 主导。无论 FD 还是 NN 都治不了。
2. **状态非完全物理**：v2 已修。但即使纯物理维度，3 维仍偏小。
3. **SDE 假设不成立**：餐食是脉冲式外生扰动，不是高斯白噪声。
4. **Buffer 单一**：HJB 主要好处是约束 OOD critic，但单一 PID demo 几乎没有 OOD 区域。

### 6.7 用户提出的关键洞察（必须保留）

用户在 C v3 之后提出："**是不是 dataset 过于完美？我发现不管论文还是改进算法都在模仿 PID**"。

数学验证：

- TD3-BC 的 actor loss = `-λ·Q + MSE(π, a_demo)`，其中 `λ = α/|Q|.mean ≈ 2.5/1000 ≈ 0.0025`
- BC 项相对权重是 Q 项的 **400 倍**
- actor 主要在学 PID，Q 项的拉力极小
- 这就是为什么 baseline 比 PID 仅多 8% TIR，再训也突破不了

**Buffer 多样性才是性能瓶颈，不是算法**。这个洞察是中期报告 §5 的核心。

---

## 7. 已经做出的关键决策（不要推翻）

按对话顺序记录：

| 决策点 | 选择 | 理由 |
|---|---|---|
| 用哪个仓库 | hemerson1 官方 4 commit 版 | 教授没指定，论文官方 |
| reward 形式 | reward-rate（不乘 dt） | 数值稳定，actor 行为不变 |
| 数据 buffer 重建 | 不重建，复用 PID demo | A v2 数学等价，无需重训数据 |
| 先一阶 HJB 还是二阶 | 一阶 | 实现简单、Σ 难估、研究方法学 |
| Drift 估计 phys_idx | [8, 9, 10] | 状态布局确认，只这 3 维满足 SDE |
| GPU 还是 CPU | CPU | RTX 5070 Ti（sm_120）不兼容 PyTorch 1.7。**不要再尝试升级 PyTorch** |
| QUICK / FULL 模式 | FULL 100k × 3 seed 用于报告 | 验证训练步数不是瓶颈 |
| 给教授交付什么 | 中期报告 docx + 4 方向选择 | 用户选了"先写报告等教授裁定" |

---

## 8. 已知坑（每个都踩过）

| 坑 | 现象 | 解决 |
|---|---|---|
| `pip install -e .` | setup.py 不存在 | 跳过这步，README 写错了 |
| `Cannot re-register id: simglucose-child1-v0` | gym 重复注册 | cell 1 已加 try/except 抓住 |
| Notebook controller DISPOSED | kernel 内存爆 | 重启 VS Code、关浏览器、用 CPU |
| RTX 5070 Ti CUDA available=True 但 sm_120 不兼容 | 训练 silently fallback 到 CPU | cell 1 自动检测，强制 device=cpu |
| Critic loss 涨到几千 | 看着像爆炸 | 实际是 Q 量级正常成长，看 actor loss 是否稳定 |
| Kernel 缓存旧 import | 改完代码 cell 还跑旧版 | 重启 kernel 后跑 |
| jupyter notebook 命令不存在 | 新版 jupyter 拆分了 | 用 `jupyter lab` 或 VS Code |
| Run All 跑掉 baseline 浪费时间 | training 是 SKIPPING 但还 forward 一次 | 文件命名分 tag (`ct`、`ct_hjb`、`ct_hjb_net`) 避免覆盖 |

---

## 9. 给教授的中期报告 — 4 个候选方向（教授尚未答复）

报告文件名：`offline-glucose-中期报告.docx`（在 `Glucose Control/Research Project/` 下，**不在 git 里**，只在用户本地）。

报告 §7 列了 4 个方向，按 Claude 推荐顺序：

| # | 方向 | 耗时 | 风险 | 价值 | 适合场景 |
|---|---|---|---|---|---|
| ① | A 收尾，作为最终交付 | 1 周 | 低 | 低（仅工程贡献） | 教授时间紧 |
| ② | 重新设计 buffer，提升多样性 | 2-3 周 | 中 | 高 | 教授希望深入 offline RL 数据效率 |
| ③ | 放弃 HJB，改其他连续时间路线（自适应 Δt / time-aware critic） | 3-4 周 | 中 | 中-高 | 教授希望保留连续时间主题但绕开 HJB |
| ④ | 升级 C 至二阶 HJB（加 ½tr(∇²Q·Σ̂) 项） | 1-2 周 | 高 | 低 | 教授坚持 HJB 路线 |

**Claude 个人偏好**：① + ② 作为 future work。但**实际方向必须教授定**，不要替他做决定。

---

## 10. 教授可能反问的问题（提前准备）

会话历史里讨论过的尖锐问题：

1. **"为什么 baseline 只有 67% 不是论文的 78%?"** → 回答：单 patient/单 buffer/可能的 reward shaping 差异；A 方案与 baseline 数学等价证明实现没 bug
2. **"HJB 项里为什么用 grad_s Q 而不是 grad_a Q?"** → 因为 HJB 是关于状态的偏微分方程，actor 选 a 是给定的；critic 关于 a 的梯度是 actor loss 用的，不在 HJB 里
3. **"为什么 Q(s,a) 和 target_Q 用同一个公式而不是离散+连续混用？"** → 一个等式自洽性。对话里有详细解释（参见 conversation summary § 关于 reward-rate 那一段）
4. **"为什么不直接调小 α 或 λ_hjb?"** → 调过 1.0；调小相当于退回 A，HJB 项失去意义
5. **"为什么不去仿真器里直接拿真值 μ 和 Σ?"** → 仿真器没暴露接口；真实病人也没有，研究价值在于从数据估
6. **"DriftNet 学得太好（loss=0.0003）反而是坏事？"** → 是的。256-256 容量足以拟合 FD 噪声，没起到平滑作用

---

## 11. 重要文件 / 行号对照（让 Claude 快速找到关键代码）

### TD3_BC_ct.py 关键行

| 行号 | 内容 |
|---|---|
| ~108 | `class DriftNet` — 256-256-3 的 MLP |
| ~145 | `self.use_hjb`, `self.lambda_hjb`, `self.drift_mode`, `self.drift_warmup_steps`, `self.phys_idx = [8, 9, 10]` 定义 |
| ~186 | `save_model` — 三个 tag 的命名规则 |
| ~262 | `target_Q = reward + done * self.discount * target_Q` ← A 方案的灵魂改动 |
| ~289 | HJB block 开始（`if self.use_hjb`）|
| ~324 | `hjb_residual = self.beta * q1_g - reward - drift_term` |
| ~364 | 进度打印（含 HJB Loss 和 Drift Loss）|

### Comparison_CT.ipynb cell 顺序

| Cell ID | 类型 | 内容 |
|---|---|---|
| init-env | code | 环境注册（含 GPU 检测 + 自动 fallback） |
| init-params | code | 参数（含 QUICK/FULL 切换 flag）|
| collect-data | code | buffer 收集（已存在则 skip） |
| train-baseline | code | 训练论文 baseline，3 seed |
| train-ct | code | 训练 A 方案，3 seed |
| (next cell) | code | 训练 C 方案，3 seed（drift_mode="net"） |
| metrics-fn | code | 评估指标计算 |
| test-rollouts | code | 测试 rollout 收集 BG trace |
| results-table | code | 出表 + ΔTIR 比较 |
| plot-curves | code | 出 5 条线对比图 |

---

## 12. 给新设备 Claude 的工作建议

如果你是在新设备上读这份文档的 Claude：

1. **不要重做实验**，除非用户明确要求。100k × 3 seed 实验已经做过两轮，CPU 上每轮 1.5-3 小时
2. **不要继续调 C 方案的 λ_hjb 之类**——会话历史已经证明这条路不可行（4 层诊断）
3. **不要尝试升级 PyTorch 用 GPU**——RTX 5070 Ti 不兼容 1.7.1，环境会破
4. **优先关心教授有没有回复**。如果有，按教授指定的方向推进（4 个方向之一或新方向）
5. **如果教授没回**，可以帮用户：
   - 复读会话历史确认理解
   - 准备口头汇报材料（用户可能要面对面讲解）
   - 探索性地写"方向 ②（重收 buffer）"的设计文档（不实施，只设计）
   - 探索性地写"方向 ③（自适应 Δt）"的算法草图
6. **保持诚实**：C 方案在当前问题上失败的事实不要修饰。如果教授追问，承认局限是对研究负责
7. **保持对用户的尊重**：用户提出的"buffer 多样性"洞察是这个项目最深刻的发现，比所有算法实验都重要。讨论时要把它当成 first-class citizen

---

## 13. 仓库 push 状态

- 仓库地址：https://github.com/hyuan21/glucose_control
- 最新 commit：`e72b6f6` "Add continuous-time TD3-BC (Approach A) and HJB regularisation (Approach C)"
- push 时间：2026-04-30
- 用户名 / 邮箱：Hanzhang Yuan / hyuan21@ucla.edu

新设备 clone 后应该看到：
- `TD3_BC.py`（论文未动）
- `TD3_BC_ct.py`（A + C 实现）
- `Comparison_CT.ipynb`（新 notebook）
- `HANDOFF.md`（这份）
- `utils/`（论文工具，未动）
- 其他论文文件

不应该看到：
- `Models/`（gitignored）
- `Replays/`（gitignored）
- `*中期报告*.docx`（gitignored）

---

**结尾**：如果你是新会话的 Claude，读完了这份。请用一段话告诉用户：(a) 我们做了什么，(b) 当前的状态卡在哪，(c) 你认为下一步该问 ta 什么。然后等用户回应再继续。
