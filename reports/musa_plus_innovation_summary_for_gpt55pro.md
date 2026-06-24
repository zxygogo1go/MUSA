# MUSA+ 当前创新点与实验提升梳理

用途：这份材料用于发给 GPT-5.5 Pro，请它在不重复现有工作的前提下，帮助凝练或设计第三个论文创新点。

当前方法名称可暂写为：**M05+MUSA+ adaptive Stage3 safe**。它是在 M05+MUSA Stage1/Stage2 backbone 后新增一个面向小器官残差失配的自适应局部 refinement 阶段。

---

## 1. 当前整体方法定位

原始 MUSA/M05+MUSA 的两阶段配准可以较好处理头颈部 CT 的全局结构、骨性结构和大器官对齐，但在视神经、晶状体、垂体、视交叉、耳蜗等小器官上仍存在明显 residual mismatch。

当前 MUSA+ 的核心思路不是推翻 MUSA，而是：

1. 保留 M05+MUSA 的 Stage1/Stage2 作为稳定 backbone；
2. 在 Stage2 之后新增一个轻量 Stage3 local residual refinement network；
3. 只在 small-OAR ROI 附近预测 residual DVF，用于补偿 Stage2 后的小器官局部失配；
4. 根据每个 pair 的 Stage2 后残差难度动态调节 refinement 强度和正则；
5. 加入 large-OAR / bone preservation、Jacobian safety 和 no-harm model selection，避免为了小器官提升而破坏整体解剖结构。

因此当前最稳的论文叙事是：

> MUSA+ 是一个以 MUSA two-stage registration 为基础的 small-organ-aware adaptive residual refinement framework。它针对 Stage2 后小器官 residual mismatch，利用 pair difficulty 和 anatomy-conditioned regularization 实现局部、可控的 deformation refinement。

---

## 2. 已有创新点 1：Small-OAR-aware Stage3 Local Residual Refinement

### 2.1 解决的问题

Stage2 后大结构和骨通常已经较稳定，但小器官由于体积小、边界弱、标签稀疏、局部形变复杂，Dice 仍然偏低。直接增强全局配准容易带来大器官损伤和 folding，因此需要一个局部、小器官感知的 refinement 阶段。

### 2.2 技术实现

Stage1 和 Stage2 仍使用 M05 DualPRNet-v1：

- Stage1：r2/coarse resolution，全局粗配准；
- Stage2：r1/full resolution，基于 Stage1 结果继续做 residual registration，得到 MUSA Stage2 DVF。

新增 Stage3 使用 `LocalResidualUNet`，是一个轻量 3D U-Net，只预测 residual local DVF：

- 输入通道数：7
  - fixed CT
  - Stage2 warped moving CT
  - fixed small-OAR mask
  - Stage2 warped moving small-OAR mask
  - Stage2 DVF magnitude
  - fixed bone mask
  - Stage2 warped moving bone mask
- 输出通道数：3，对应 x/y/z 三个方向的 residual DVF；
- 网络结构：
  - ConvBlock = Conv3d + InstanceNorm3d + LeakyReLU + Conv3d + InstanceNorm3d + LeakyReLU；
  - Encoder: 7->8, 8->16, 16->32；
  - Decoder: ConvTranspose3d upsampling + skip connection；
  - Output: Conv3d 8->3；
  - 约 8.7 万参数，远小于重新训练一个完整配准 backbone。
- 最终变形场：
  - `final_dvf = stage2_dvf + roi_gate * residual_scale * stage3_residual_dvf`

关键点：Stage3 不是重新做全局配准，而是在 Stage2 的稳定结果上做局部 residual correction。

### 2.3 与 MUSA 相比新增了什么

相比 MUSA，新增内容包括：

- 一个独立的 Stage3 轻量 3D U-Net residual refinement network；
- small-OAR ROI gate；
- small-OAR mask、bone mask、Stage2 DVF magnitude 等显式解剖/形变条件输入；
- residual DVF 与 Stage2 DVF 的局部合成机制；
- Stage3 专用 loss，包括 local image MSE、small-OAR Dice、weighted smoothness、weighted magnitude、Jacobian safety、large/bone preservation。

### 2.4 实验提升

SegRap held-out test，M05+MUSA Stage2 vs M05+MUSA+ Stage3 safe：

| 指标 | Stage2 | Stage3 safe | 提升 |
|---|---:|---:|---:|
| All-OAR Dice | 0.5442 | 0.6559 | +0.1117 |
| Small-OAR Dice | 0.3452 | 0.6015 | +0.2563 |
| Large-OAR Dice | 0.6307 | 0.6306 | -0.0001 |
| Bone Dice | 0.7888 | 0.7894 | +0.0006 |
| Stage3 wins | - | all-OAR 5/5, small-OAR 5/5 | 稳定提升 |

SegRap per-label 小器官最大收益：

| 标签 | Stage2 | Stage3 | 提升 |
|---|---:|---:|---:|
| OpticNerve_R | 0.3463 | 0.9769 | +0.6306 |
| Lens_L | 0.3869 | 0.9680 | +0.5811 |
| Chiasm | 0.2722 | 0.8532 | +0.5811 |
| OpticNerve_L | 0.3766 | 0.9520 | +0.5754 |
| Lens_R | 0.3949 | 0.8941 | +0.4992 |
| MiddleEar_L | 0.4988 | 0.8458 | +0.3470 |

HaN-Seg held-out test，M05+MUSA Stage2 vs M05+MUSA+ Stage3 safe：

| 指标 | Stage2 | Stage3 safe | 提升 |
|---|---:|---:|---:|
| All-OAR Dice | 0.3213 | 0.4417 | +0.1204 |
| Small-OAR Dice | 0.1304 | 0.5827 | +0.4522 |
| Large-OAR Dice | 0.3940 | 0.3880 | -0.0061 |
| Bone Dice | 0.7104 | 0.7105 | +0.0001 |
| Stage3 wins | - | all-OAR 5/5, small-OAR 5/5 | 跨数据集仍稳定提升 |

HaN-Seg per-label 小器官最大收益：

| 标签 | Stage2 | Stage3 | 提升 |
|---|---:|---:|---:|
| OAR_Pituitary | 0.1949 | 0.8844 | +0.6895 |
| OAR_OpticNrv_L | 0.0613 | 0.6951 | +0.6337 |
| OAR_OpticNrv_R | 0.1351 | 0.7506 | +0.6155 |
| OAR_Glnd_Lacrimal_L | 0.0770 | 0.5717 | +0.4947 |
| OAR_Glnd_Lacrimal_R | 0.2353 | 0.6664 | +0.4311 |
| OAR_Cochlea_L | 0.1754 | 0.5701 | +0.3946 |

### 2.5 可作为论文表述的结论

Stage3 local residual refinement 是当前最核心、最直观的创新点。它在两个头颈部纵向配准数据集上都显著提高 small-OAR Dice，并带动 overall OAR Dice 提升。SegRap 上 large-OAR 和 bone 基本不受影响；HaN-Seg 上 small-OAR 提升更大，但也暴露出局部形变更激进的问题。

---

## 3. 已有创新点 2：Pair-Difficulty-Aware Adaptive Control + Anatomy-Conditioned Regularization

### 3.1 解决的问题

不同病例 pair 的配准难度不同：

- 有些 pair Stage2 后已经比较好，只需要很弱的局部修正；
- 有些 pair 小器官残差大，需要更强 residual；
- 如果所有病例使用同一 refinement 强度和同一正则，容易出现：
  - easy pair 过修正；
  - hard pair 修不动；
  - 小器官提升与大器官/骨保持之间 trade-off 不稳定；
  - 局部 folding 增加。

因此当前方法引入 pair-difficulty-aware 控制。

### 3.2 difficulty score 如何计算

Stage3 使用 Stage2 后的 residual difficulty，而不是初始图像难度：

`difficulty = 0.45 * (1 - small Dice after Stage2) + 0.20 * (1 - bone Dice after Stage2) + 0.25 * local image MSE + 0.10 * Stage2 flow p95 score`

含义：

- small-OAR overlap 越差，难度越高；
- bone overlap 越差，难度越高；
- Stage2 后局部 CT 残差越大，难度越高；
- Stage2 flow p95 越大，说明已有变形更强，难度越高。

### 3.3 动态调整了哪些权重或超参数

它不是单纯的 dynamic loss weighting，而是同时动态控制 refinement 空间范围、残差幅度、loss 权重和解剖正则图。

实际参与 difficulty-aware 调整的项：

| 项 | 公式/机制 | 作用 | 当前 safe 实验说明 |
|---|---|---|---|
| `residual_scale` | `scale_min + d * (scale_max - scale_min)` | 控制 Stage3 residual DVF 幅度，easy pair 少动，hard pair 多修正 | 使用 0.25 -> 1.00 |
| `lambda_small` | `lambda_small_min + d * (lambda_small_max - lambda_small_min)` | hard pair 增大小器官 Dice loss 权重 | 使用 1.0 -> 3.0 |
| `lambda_smooth` | `lambda_smooth + lambda_smooth_extra * d` | hard pair 在允许更强修正的同时增加 smoothness | 使用 0.10 + 0.20*d |
| ROI radius | `radius_min + d * (radius_max - radius_min)` | 控制 small-OAR refinement 空间范围 | 机制支持动态；最终 safe run 为降低风险设置 3/3，因此实际固定为 3 |
| anatomy smooth map | bone、boundary、ROI 根据 difficulty 加权 | 骨/边界处更强平滑约束，控制局部形变 | 参与 weighted gradient loss |
| anatomy magnitude map | ROI inside/outside/bone 根据 difficulty 加权 | ROI 内允许必要 residual，ROI 外和骨处限制 residual 泄漏 | 参与 weighted magnitude loss |

另外还有安全相关项，它们不一定随 difficulty 动态变化，但构成 safe 版本的关键约束：

| 安全项 | 作用 |
|---|---|
| `lambda_jacobian` + `jacobian_roi_weight` | 惩罚局部 folding，尤其 small-OAR ROI 内 Jacobian 异常 |
| `lambda_preserve_large` | 约束 Stage3 后 large-OAR 不偏离 Stage2 |
| `lambda_preserve_bone` | 约束骨性结构不被 Stage3 破坏 |
| no-harm best-policy | validation 选模型时惩罚 ROI Jacobian、large/bone degradation 和过强 residual |

### 3.4 实验提升与证据

SegRap 上这个 adaptive/safe 设计的结果比较理想：

- small-OAR Dice 提升 +0.2563；
- all-OAR Dice 提升 +0.1117；
- large-OAR Dice 几乎不变：-0.0001；
- per-label large-OAR 没有出现 >0.02 或 >0.05 的明显下降；
- bone Dice 稳定：+0.0006；
- Stage3 residual ROI p95 约 1.8988，说明新增形变幅度相对克制；
- ROI non-positive Jacobian ratio 约 0.0069，global non-positive Jacobian ratio 约 5.39e-05。

HaN-Seg 上说明这个机制具备跨数据集迁移性，但安全项仍不足：

- small-OAR Dice 提升 +0.4522；
- all-OAR Dice 提升 +0.1204；
- bone Dice 基本不变：+0.0001；
- large-OAR Dice 平均下降 -0.0061；
- large-OAR worst-label delta 平均 -0.0922；
- degradation 主要集中在 orbital labels：OAR_Eye_PL、OAR_Eye_PR、OAR_Eye_AL；
- ROI non-positive Jacobian ratio 上升到约 0.0400；
- Stage3 residual ROI p95 约 4.3048，明显高于 SegRap。

### 3.5 可作为论文表述的结论

Pair-difficulty-aware adaptive control 是第二个创新点。它将每个 pair 的 Stage2 后残差状态转化为局部 refinement 的强度、loss 权重和正则图控制，使 Stage3 不再使用固定 refinement 策略。SegRap 上它能在明显提升小器官的同时保持大结构和骨稳定；HaN-Seg 上说明该思路能迁移，但也暴露出需要更强 topology / large-OAR preservation 的空间。

---

## 4. 当前两个创新点各自对应的提升总结

| 创新点 | 主要作用 | SegRap 证据 | HaN-Seg 证据 | 当前不足 |
|---|---|---|---|---|
| Small-OAR-aware Stage3 local residual refinement | 修正 Stage2 后小器官 residual mismatch | small-OAR +0.2563, all-OAR +0.1117, 5/5 pair 提升 | small-OAR +0.4522, all-OAR +0.1204, 5/5 pair 提升 | 个别极小结构仍不稳定，如 IAC/Cochlea |
| Pair-difficulty-aware adaptive control + anatomy-conditioned regularization | 按 pair 难度控制 residual 强度、小器官 loss、smoothness 和解剖正则 | large-OAR -0.0001, bone +0.0006, no large label drop >0.02 | bone +0.0001，说明骨保持稳定 | HaN-Seg orbital large-OAR drop 和 ROI folding 偏高 |

需要注意：目前还没有完整 ablation 能把每一个子模块的贡献严格拆开。因此“各自提升”更适合表述为机制对应的实验证据，而不是严格因果归因。后续若写论文，需要补充 ablation：

- Stage2 only；
- Stage3 fixed strength；
- Stage3 + dynamic residual scale；
- Stage3 + dynamic loss；
- Stage3 + anatomy maps；
- Stage3 + safe/no-harm selection；
- full method。

---

## 5. 当前问题与第三创新点的空位

已有两个创新点已经比较清楚：

1. 小器官感知的 Stage3 residual refinement；
2. pair-difficulty-aware adaptive control 与 anatomy-conditioned safe regularization。

现在还缺一个更有论文价值的第三创新点。这个第三点最好不要只是“又加一个 loss”，而应该能回应当前实验暴露的问题：

### 5.1 当前主要短板

SegRap：

- 整体较好；
- small-OAR 提升明显；
- large-OAR 和 bone 基本 no-harm；
- ROI folding 有少量增加但可接受；
- 极小结构如 IAC、Cochlea_R 仍存在 0 Dice 或下降。

HaN-Seg：

- small-OAR 提升非常大；
- all-OAR 也提升；
- bone 稳定；
- 但是 Stage3 residual 更强，ROI non-positive Jacobian ratio 达到约 4.0%；
- orbital large-OAR 出现局部下降，尤其 OAR_Eye_PL、OAR_Eye_PR、OAR_Eye_AL；
- 说明跨数据集时 fixed adaptive rule 仍可能过激。

### 5.2 第三个创新点可以考虑的方向

请 GPT-5.5 Pro 帮忙思考时，可以重点让它围绕以下方向提出更强、更像论文创新点的方案：

1. **Topology-aware / diffeomorphic-safe local refinement**
   - 目标：解决 HaN-Seg 上 ROI folding 偏高的问题；
   - 可能方向：Jacobian-aware residual projection、stationary velocity field residual、inverse-consistency、cycle consistency、folding-aware uncertainty gating；
   - 要求：不能显著牺牲 small-OAR Dice。

2. **Neighbor-risk-aware orbital preservation**
   - 目标：解决 optic/lacrimal/chiasm ROI 附近大器官眼球结构下降；
   - 可能方向：自动识别 small-OAR ROI 周围的 high-risk neighboring large-OAR，加入 spillover penalty、distance-transform barrier、label-adjacency graph regularization；
   - 要求：比普通 `lambda_preserve_large` 更具体，能解释为什么保护 orbital structures。

3. **Dataset/domain-adaptive refinement scheduler**
   - 目标：解决 SegRap 与 HaN-Seg 难度、Stage2 baseline、residual p95 不同导致同一 adaptive rule 迁移后过激；
   - 可能方向：根据 dataset/domain statistics 校准 residual-scale-max、Jacobian weight、large preservation weight；或者学习一个 confidence/safety controller；
   - 要求：体现“不同纵向配准数据集需要不同 refinement policy”。

4. **Reliability-aware small-structure refinement**
   - 目标：解决 IAC、Cochlea 等极小结构受重采样、mask 稀疏和标签缺失影响大；
   - 可能方向：label volume-aware weighting、structure reliability score、multi-slice consistency、tiny-label uncertainty filtering；
   - 要求：不要让不可靠标签过度驱动 DVF。

5. **Head-and-neck large-motion-aware evaluation/visualization framework**
   - 目标：体现头颈部大幅度运动特点；
   - 可能方向：姿态差异、骨轮廓、DVF magnitude、deformation grid/vector、小器官 overlay 的论文级可视化与指标；
   - 注意：如果只作为可视化，创新性可能不够；若与训练中的 large-motion-aware controller 结合，可能更强。

---

## 6. 建议直接发给 GPT-5.5 Pro 的 prompt

下面这段可以直接复制：

```text
我在做头颈部 CT 纵向 deformable registration，baseline 是 M05+MUSA Stage1/Stage2。现在我提出了 MUSA+ adaptive Stage3 safe。

已有两个创新点：

1. Small-OAR-aware Stage3 local residual refinement：
   在冻结的 M05+MUSA Stage1/Stage2 后新增一个轻量 3D U-Net LocalResidualUNet，输入 fixed CT、Stage2 warped moving CT、fixed/warped small-OAR mask、Stage2 DVF magnitude、fixed/warped bone mask，输出 3-channel local residual DVF。最终 final_dvf = stage2_dvf + roi_gate * residual_scale * residual_dvf。它不是重新全局配准，而是只在 small-OAR ROI 附近修正 Stage2 后 residual mismatch。

2. Pair-difficulty-aware adaptive control + anatomy-conditioned regularization：
   先根据 Stage2 后 residual difficulty 估计每个 pair 的难度：difficulty = 0.45*(1-small Dice) + 0.20*(1-bone Dice) + 0.25*local image MSE + 0.10*Stage2 flow p95。然后动态调整 residual_scale、lambda_small、lambda_smooth、ROI radius 机制和 anatomy-conditioned smooth/magnitude maps。safe 版本还加入 Jacobian penalty、large-OAR/bone preservation 和 no-harm model selection。

实验结果：

SegRap held-out test:
- All-OAR Dice: 0.5442 -> 0.6559, +0.1117
- Small-OAR Dice: 0.3452 -> 0.6015, +0.2563
- Large-OAR Dice: 0.6307 -> 0.6306, -0.0001
- Bone Dice: 0.7888 -> 0.7894, +0.0006
- Small-OAR 和 all-OAR 都是 5/5 pair 提升；large-OAR 没有 >0.02 的明显下降。

HaN-Seg held-out test:
- All-OAR Dice: 0.3213 -> 0.4417, +0.1204
- Small-OAR Dice: 0.1304 -> 0.5827, +0.4522
- Large-OAR Dice: 0.3940 -> 0.3880, -0.0061
- Bone Dice: 0.7104 -> 0.7105, +0.0001
- Small-OAR 和 all-OAR 也是 5/5 pair 提升，但 ROI non-positive Jacobian ratio 约 4.0%，large-OAR worst-label delta 平均 -0.0922，下降主要集中在 orbital labels，例如 OAR_Eye_PL/OAR_Eye_PR/OAR_Eye_AL。

我现在需要想第三个论文创新点。请你不要重复上面两个创新点，也不要只说“加一个 loss”。请围绕当前结果的短板，帮我设计一个更像论文贡献的第三创新点，并说明：

1. 这个创新点要解决什么具体问题；
2. 技术实现可以怎么做；
3. 它与前两个创新点的边界是什么；
4. 应该做哪些 ablation 和指标证明它有效；
5. 怎么把它写成论文中的 contribution statement。

我尤其关心：
- HaN-Seg 上 ROI folding 偏高；
- orbital large-OAR 在 small-OAR refinement 附近下降；
- 不同数据集难度不同，同一 refinement policy 可能不适合所有 pair；
- 极小结构如 IAC/Cochlea 的标签稀疏与不稳定问题。
```

---

## 7. 可引用的结果来源

SegRap:

- `outputs/paper_split_compare_stage2_stage3_safe_test_full/stage2_vs_stage3_summary.csv`
- `outputs/paper_split_compare_stage2_stage3_safe_test_full/stage2_vs_stage3_by_pair.csv`
- `outputs/paper_split_compare_stage2_stage3_safe_test_per_label/per_label_summary.csv`
- `outputs/paper_split_compare_stage2_stage3_safe_test_per_label/per_label_group_summary.csv`

HaN-Seg:

- `outputs_hanseg/paper_split_compare_stage2_stage3_safe_test_full/stage2_vs_stage3_summary.csv`
- `outputs_hanseg/paper_split_compare_stage2_stage3_safe_test_full/stage2_vs_stage3_by_pair.csv`
- `outputs_hanseg/paper_split_compare_stage2_stage3_safe_test_per_label/per_label_summary.csv`
- `outputs_hanseg/paper_split_compare_stage2_stage3_safe_test_per_label/per_label_group_summary.csv`

代码：

- Stage3 network: `musa/registration_models/musa_plus/local_refinement.py`
- Stage3 training and losses: `scripts/train/train_musa_plus_stage3.py`
- MUSA+ utilities: `musa/utils_musa_plus.py`
- Stage3 evaluation: `scripts/infer/eval_musa_plus_prepared_pairs.py`
