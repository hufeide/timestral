# Timestral 训练模块技术报告

- 范围：`src/training/` 全部代码，及其与 `src/timestral/` 推理栈的一致性
- 环境：Windows / Python 3.12 / PyTorch 2.11.0+cu128 / CUDA 可用
- 基准 checkpoint：`train_example_out/last`（`small_config`，`block_size=2`）
- 结论日期：2026-10-22

---

## 1. 结论

### 1.1 模型本体：完全一致

训练代码**不定义任何网络结构**，而是直接导入并训练 `timestral.model.TimestralModel`：

```python
from timestral.model import TimestralModel
```

因此以下内容在训练与推理之间是同一份代码、同一套参数：

- patch 嵌入、RoPE、Time/Variate 交替注意力、SwiGLU、RMSNorm
- 因果逐 patch 归一化 `scale()`
- 单调分位数头
- `predict_block()` 与 `_predict_context_only()`
- 前向掩码（因果、变量轴防火墙、双流拆分）

### 1.2 无协变量前向路径：逐元素一致

训练用 `Trainer._decode_train` 复刻推理的 `TimestralForecaster._block_ar_decode`。在相同模型、相同输入下实测：

| 场景 | max\|diff\| |
|---|---|
| 干净上下文，4 个 patch | 0.000e+00 |
| horizon = 129 步（非整数 patch） | 0.000e+00 |
| horizon = 1 步 | 0.000e+00 |
| 上下文中段缺失 | 0.000e+00 |
| 序列起始左填充缺失 | 0.000e+00 |
| context 超过 `max_context` 截断 | 0.000e+00 |
| `detach_feedback` True/False | 0.000e+00 |

这里的“一致”指前向数值完全一致，不是近似。

### 1.3 训练与推理的差异（已对齐或刻意设计）

| 项目 | 训练 | 推理 | 说明 |
|---|---|---|---|
| 协变量路径 | past/known 数值 + sampler 内类别目标编码 | `forecast()` 支持 past/known/类别协变量 | 数值协变量与 `V = 1` 类别编码逐元素一致；`V > 1` 推理需以数值传入，见 §7.1 |
| 对称平均 `symmetric_averaging` | `sign_augment` 以 50% 概率对样本（含协变量行）取负 | `symmetric_averaging=True` 默认开启 | 负号分支已进入训练分布，见 §7.2 |
| 非负截断 / real cap | `_postprocess` 在 loss 前应用同样的 clamp | `clip_nonnegative` / `quantile_real_cap_k` 默认开启 | 不引入参数；边界外的分位数梯度为零，见 §7.2 |
| 损失函数 | pinball（raw space，按 anchor scale 归一） | 不存在 | 训练专属 |
| feedback 梯度 | 可选 `detach_feedback` | 无梯度概念 | 只影响显存/梯度，不改变数值 |
| horizon 形状 | `horizon_steps` 可设为非 patch 整数倍 | `prediction_length` 任意，尾块截断 | 尾块形状已进入训练采样，见 §7.4 |

一句话：**架构 100% 相同；无协变量、数值协变量、类别协变量三条计算图均逐元素一致，推理侧的默认输出后处理与非整 patch horizon 也已进入训练路径。**

---

## 2. 模块结构

| 文件 | 职责 | 关键符号 |
|---|---|---|
| `src/training/dataset.py` | 滚动窗口采样、类别协变量目标编码 | `RollingWindowDataset`、`collate_windows`、`_encode_categorical` |
| `src/training/loss.py` | 分位数目标函数 | `pinball_loss`、`crps_from_quantiles` |
| `src/training/model_utils.py` | 配置预设与 checkpoint IO | `small_config`、`base_config`、`save_model`、`save_training_state`、`load_training_state`、`load_for_training` |
| `src/training/trainer.py` | 训练循环 | `Trainer.pack`、`_decode_train`、`forward_loss`、`_postprocess`、`fit`、`evaluate`、`save_checkpoint`、`_resume` |
| `src/training/example_train.py` | 可运行示例 | `make_synthetic`、`main` |

依赖方向是单向的：`training → timestral`，`timestral` 不导入任何训练代码。

---

## 3. 端到端数据流

```
(B, V, T) 原始上下文 + 可选 past/known 协变量
   │ pack() → TimestralForecaster._assemble_rows()（协变量路径，与推理同一份代码；无协变量时只有零填充的目标行）
   ▼
(R, T) rows + (R, T) bool mask + (R,) group_ids + (R,) role_ids + known_future
   R = B·(V_target + C_past + C_known)；无协变量时 R = B·V，role_ids = None
   │ TimestralModel.predict_block(rows, mask, gids, block, role_ids, kf, kfm)
   │   scale() → patch 嵌入 → [Enc ; Dec] → Transformer → 分位数头
   ▼
(R, block·ps, Q)  raw-space 分位数
   │ TimestralForecaster._refine_block(q, rows, mask, target_rows)
   ▼
(R, block·ps, Q)
   │ feedback：目标行取中位数（极端值替换）；known 行取真实未来；past 行填空
   ▼
context 追加 feedback → 下一 block（直到 horizon 结束）
   │ 拼接所有 block，截到 H
   ▼
(R, H, Q)
   │ 只保留 role_ids == ROLE_TARGET
   │ _postprocess：_cap_quantiles + clip_nonnegative（与 forecast 同一实现）
   ▼
(R_t, H, Q)  vs  (R_t, H) target
   │ pinball_loss(..., scale=scale0, mask=isfinite(target))
   ▼
scalar loss → backward
```

### 3.1 关键形状

| 阶段 | 张量 | 形状 |
|---|---|---|
| 输入 | context | `(B, V, T)` |
| 打包 | rows / mask / group_ids | `(R, T)` / `(R, T)` / `(R,)`，无协变量时 `R = B·V` |
| 打包（协变量） | role_ids / known_future / known_future_mask | `(R,)` / `(R, H)` / `(R, H)`，`R = B·(V + C_past + C_known)` |
| 归一化 | loc / scale | `(R, T)`，patch 内常量 |
| 编码 | x_enc | `(R, ll, D)` |
| 解码查询 | x_dec | `(R, block, D)` |
| 单块输出 | quantiles | `(R, block·ps, Q)` |
| 完整 horizon | packed | `(R, H, Q)`，取目标行后 `(R_t, H, Q)` |
| 损失 | scalar | `()` |

### 3.2 采样语义

`RollingWindowDataset.__getitem__(idx)` 用 `(seed, worker_id, idx)` 作为熵源：

- 不同 `idx` → 不同序列/起点；
- 不同 DataLoader worker → 不同流；
- 相同 `idx` → 可复现（断点续训重放一致）。

`__len__` 返回 `1 << 30`（虚拟无限数据集），因此：

- 训练循环按 `total_steps` 停止，不按 epoch；
- 验证必须限制 batch 数（`evaluate(max_batches=8)`）；
- `len(dataloader)` 没有实际意义。

---

## 4. 训练循环

`Trainer.fit()` 的每个 step：

1. **设置学习率**：`_update_lr()` 在 forward 之前调用，第 1 步即为 `lr0/warmup`。
2. **前向**：`pack → _decode_train → _refine_block → 拼接`，全程 raw space。
3. **后处理**：按 `role_ids` 只保留目标行，再走 `_postprocess`（`_cap_quantiles` + `clip_nonnegative`）。
4. **损失**：`pinball_loss(packed, tgt, levels, scale0, isfinite(tgt))`。
5. **反向**：bf16 autocast 下前向，损失在 fp32 计算；梯度裁剪后 `optimizer.step()`。
6. **EMA**：`ema_decay > 0` 时更新滑动平均。
7. **验证/存档**：按 `val_every`、`checkpoint_every` 周期执行。

### 4.1 解码循环与推理的对应关系

训练侧（`trainer.py::_decode_train`）：

```python
block = min(self.per_block_patches, remaining)
q = self.model.predict_block(rows, mask, group_ids, block, role_ids, kf_block, kfm_block)
q = self.forecaster._refine_block(q, rows, mask, target_rows)  # 只重框目标行
feedback = replace_extreme_values(q[..., median_index], threshold)
if role_ids is not None:            # known 行喂真实未来，past 行填空
    feedback = torch.where(known_rows[:, None], kf_block, feedback)
    feedback = torch.where(past_rows[:, None], torch.zeros_like(feedback), feedback)
rows = torch.cat([rows, feedback], dim=-1)
if rows.shape[-1] > max_context:
    keep = max(ps, (max_context // ps) * ps)
```

推理侧（`forecaster.py::_block_ar_decode`）与上段逐行相同：同样的 `remaining`/`block` 计算、同样的 `predict_block` 调用、同样的 `_refine_block`、同样的 `replace_extreme_values`、同样的 `keep` 截断公式。训练侧唯一的额外交付是 `target_rows` 掩码——目标行在 `_assemble_rows` 中排在最前，`_refine_block` 只对它们做 frame 变换。`role_ids=None` 时该段整体跳过，即纯目标路径。

### 4.2 损失设计

- **raw space**：分位数和 target 都不在归一化空间比较，避免 asinh 逆变换误差进入目标函数。
- **按序列加权**：除以最后一个上下文 patch 的 `scale`，使不同量级的序列贡献相近。
- **缺失值**：`isfinite(target)` 作为 mask，缺失位置不计入分子也不计入分母（`torch.where`，不是乘法，避免 `0 * NaN`）。
- **锚点选择**：`scale0` 取反馈前的上下文；后续 block 的分位数已处于各自的 advancing frame，分母仍是首块锚点，因此它是**每序列静态权重**，不是逐步 frame 权重。

### 4.3 学习率

```
s <= w : lr = lr0 * s / w
s >  w : lr = lr0 * 0.5 * (1 + cos(pi * (s - w) / (T - w)))
```

由于 `_update_lr` 在 `optimizer.step()` 之前执行，实际生效序列为 `3e-5, 6e-5, 9e-5, ...`（`lr0=3e-4, w=10`）。

### 4.4 检查点

`save_checkpoint(dir)` 写三样东西：

```
config.json         TimestralConfig（含 block_size，推理侧读取）
model.safetensors   权重（启用 EMA 时写 EMA 权重）
training_state.pt   optimizer / step / RNG
```

`resume_from=dir` 会依次恢复权重、优化器状态、step、RNG；`evaluate()` 与 `save_checkpoint()` 共用 `_ema_weights()`，保证“选 best 的权重”和“落盘的权重”是同一套。

---

## 5. 一致性审计

### 5.1 逐项对照

| 检查项 | 训练实现 | 推理实现 | 判定 |
|---|---|---|---|
| 网络定义 | `timestral.model.TimestralModel` | 同左 | 同一份代码 |
| 行组装 | `Trainer.pack` 直接调用 `TimestralForecaster._assemble_rows` | `_forecast_raw` 调用同一方法 | 同一份代码 |
| 单块预测 | `model.predict_block(..., role_ids, kf, kfm)` | 同 | 一致（target-only 时 `role_ids=None`，走同一分支条件） |
| 分位数反归一化 | `_refine_block(q, rows, mask, target_rows)` | 同 | 一致（target-only 时 `target_rows=None`） |
| 块间反馈 | 中位数 + 极端值替换；known 行喂真实未来，past 行喂空 | 同 | 一致 |
| 上下文截断 | `max(ps, (max_context // ps) * ps)` | 同 | 一致 |
| 缺失值处理 | 协变量路径交给 `_assemble_rows` 做 ffill；无协变量路径在缺失位保留 0 | `_as_context` + `_assemble_rows` 做 ffill | 数值一致（缺失位由 mask 屏蔽，`scaled` 均为 0） |
| 位置编码 | 模型内部按 group 首观测 patch 计算 | 同 | 一致 |
| block 大小 | 强制 `per_block_patches == config.block_size` | `per_shot = min(per_shot, block_size)` | 已强制对齐 |
| 对称平均 | `sign_augment` 50% 取负 | 默认开启 | 负号样本进入训练分布 |
| 非负截断 / real cap | `_postprocess`（复用 `_cap_quantiles` / `nonnegative_context`） | 默认开启 | 同一后处理路径 |
| 协变量 | past/known 数值 + sampler 内类别编码 | past/known/类别协变量 | 数值与 `V = 1` 一致；`V > 1` 需以数值传入 |
| horizon 形状 | `horizon_steps` 任意步 | `prediction_length` 任意 | 尾块形状可训练 |

### 5.2 数值等价实验

同一模型实例、`eval()`、`block_size=2`，比较 `Trainer._decode_train` 与 `TimestralForecaster._block_ar_decode`；horizon 取 128（两个 block，触发反馈）、129（非整 patch）和 1 步：

```
clean ctx, 4 patches             max|diff| = 0.000e+00
clean ctx, 129 steps             max|diff| = 0.000e+00
clean ctx, 1 step                max|diff| = 0.000e+00
missing ctx values               max|diff| = 0.000e+00
leading missing (left pad)       max|diff| = 0.000e+00
max_context=64 truncation        max|diff| = 0.000e+00
detach_feedback True/False       max|diff| = 0.000e+00
```

协变量路径（`V=2` 目标 + 1 past + 1 known，known 内含 NaN，horizon=32 跨两个 block）：

```
assembled rows equal (train vs infer)  True
train vs _block_ar_decode              max|diff| = 0.000e+00
train vs forecast()                    max|diff| = 0.000e+00
target-only 路径回归                   max|diff| = 0.000e+00
```

类别协变量编码（sampler 产出 vs `forecast` 的 `_encode_categorical`，`V = 1`）：

```
torch.equal / rtol=0, atol=0                   True
```

`V > 1` 的 sampler 编码 vs 独立复刻的"观测变量均值池化 + ffill"参考实现：

```
_encoding_target + _encode_rows 复刻比对      rtol=0, atol=0
与原始 code 不同                                True（确认不是透传）
```

非整 patch horizon（`horizon_steps=100`，`ps=32`，末尾尾块 4 步）：

```
Trainer._decode_train vs forecast(100)  max|diff| = 0.000e+00
```

输出后处理（`quantile_real_cap_k` + `clip_nonnegative` 同时开启）：

```
Trainer._postprocess vs forecast()      max|diff| = 0.000e+00
被 clamp 的分位占比                     0.444（宽分位模型，触发充分）
```

`detach_feedback` 只改变 autograd 图，不改变前向数值；这一点由同一模型、同一随机输入实测确认。

数值比较的前提是**同一 batch 布局**：训练把 `R = B·V_all` 行一次送进模型，`forecast` 的 batch 也是 `B` 个序列，两者只有在样本数相同时矩阵归约顺序才相同，`max|diff|` 才是精确的 0；若逐样本比较，会看到约 `1e-5` 的 float32 归约差异，这不是逻辑不一致。

全路径一致性矩阵（`tests/test_training.py::test_training_forward_matches_inference`，11 个配置，`ps=8`、`block_size=2`）：

| 配置 | 协变量 | 目标 | 该配置额外覆盖 |
|---|---|---|---|
| `no_cov` | 无 | V=1 | 基线 |
| `no_cov_missing` | 无 | V=1 | 起始 2 步 + 中段 5 步 + 单点缺失 |
| `no_cov_v2` | 无 | V=2 | 一条变量中段缺失，变量轴防火墙 |
| `no_cov_partial_horizon` | 无 | V=1 | `H = 2·ps + 3` |
| `past_cov` | past×2（含 NaN） | V=1 | past-only 行角色与填空反馈 |
| `known_cov` | known×2（0/1 指示行 + 连续行，ctx 与 future 均有 NaN） | V=1 | 二元归一化分支 + known-future 真值反馈 |
| `past_known_v2` | past×1 + known×1（0/1） | V=2 | 三类行同时在组内，跨两个 block 反馈 |
| `known_partial_horizon` | known×1 | V=1 | `H = 2·ps + 5`，尾块截断 + 协变量 |
| `max_context` | past×1 + known×1 | V=1 | 80 步上下文截到 `max_context=32` |
| `observation_mask` | 无 | V=1 | `forecast(observation_mask=...)` 与训练侧 NaN 编码等价 |
| `clamps_bite` | past×1 + known×1 | V=1 | 宽分位模型，cap + clip 实际生效 |

每个配置都断言以下内容与推理逐一相等（`rtol=0, atol=0`）：`mask`、`group_ids`、`role_ids`、`known_future` / `known_future_mask`、`n_tgt`；`model.scale()` 的 `loc` / `scale`；`_decode_train` vs `_block_ar_decode`；`_postprocess` vs 推理的 cap + clip；最终输出 vs `forecast(..., symmetric_averaging=False)`。

唯一的行值差异出现在**无协变量**配置：训练 `pack` 在缺失位置保留 0，推理 `_assemble_rows` 做前向填充。两者带同一 `mask`，`apply_affine_asinh_masked` 把这些位置置 0，所以 `loc` / `scale` 及全部下游数值仍逐元素相等（用例已断言）。

---

## 6. 缺陷修复与能力补齐记录

| # | 问题 | 修复 | 验证证据 |
|---|---|---|---|
| 1 | `__getitem__` 每次重建 RNG，全 batch 只有 1 条不同样本 | 以 `(seed, worker_id, idx)` 作为熵源 | batch 内 8/8 不同；同 idx 可复现 |
| 2 | 训练 `per_block=2` 而 `config.block_size=32`，推理按 32 解码 | `Trainer` 断言两者相等；示例用 `replace(config, block_size=PER_BLOCK)` | 不匹配时抛 `ValueError` |
| 3 | `loss * mask` 遇 NaN 目标变 NaN | 改 `torch.where`；`forward_loss` 传 `isfinite(tgt)` | 含 NaN 目标 loss 有限（0.3700） |
| 4 | `_update_lr` 在 `step()` 之后，首步用峰值 LR | 移到 forward 之前 | 实测 `3e-5, 6e-5, 9e-5, 1.2e-4, 1.5e-4, 1.8e-4` |
| 5 | 验证用实时权重、落盘用 EMA 权重 | 共用 `_ema_weights()` 上下文管理器 | 保存后实时权重未被污染 |
| 6 | `resume_from` 不加载权重 | `_resume` 先 `safetensors.load_model` | 磁盘张量与模型逐项相等，step=20 |
| 7 | 类别协变量没有训练通路 | `RollingWindowDataset` 接收类别码并在 sampler 内 `target_encode`（`V > 1` 先按观测变量均值池化） | `V = 1` 时与 `forecast._encode_categorical` 逐元素相等（`rtol=0`） |
| 8 | 推理默认的 cap/clip 未进入训练 | `Trainer._postprocess` 复用 `_cap_quantiles` / `nonnegative_context` | 与 `forecast` `max\|diff\| = 0` |
| 9 | horizon 只能是 patch 整数倍 | `RollingWindowDataset(horizon_steps=...)` | `horizon_steps=100` 与 `forecast(100)` `max\|diff\| = 0` |

第 7–9 项不是 bug，而是"训练覆盖的能力面小于推理 API"的补齐；实现与验证细节见 §5.2、§7.1、§7.2、§7.4。

附带清理：`or True` 死条件、`pack()` 未用参数、`save_model` 同名遮蔽、示例未使用的 `np_rng`、`dataset.py` 未使用的 `os`。

---

## 7. 已知限制与扩展点

### 7.1 协变量：数值与类别已全部接入

此前 `predict_block` 只从 `role_ids=None` 分支进入，协变量分支从未执行。现在的接法：

1. `Trainer.pack` 直接调用 `TimestralForecaster._assemble_rows`（推理的私有方法，训练侧原样复用；未改动推理代码），目标行、past-only 行、known-future 行的顺序、`role_ids`、`known_future` / mask 与推理**同一份代码**生成。
2. `_decode_train` 把 `role_ids`、`known_future` 传给 `predict_block`，反馈阶段按角色分流：known 行喂真实未来，past-only 行喂 0 且 mask=False，目标行喂中位数；`_refine_block` 只对目标行做 refine frame 变换。
3. `forward_loss` 按 `role_ids == ROLE_TARGET` 只保留目标行（`packed`、`scale0`、`rows`、`mask` 同步过滤），协变量行不进入 pinball。
4. `RollingWindowDataset` / `collate_windows` 支持 `past_covariates`、`known_covariates`，窗口切片与目标窗口对齐（known 覆盖 context + horizon）。

因此 `_predict_window`、`_known_cov_scaling`、变量轴防火墙、known-future 编码全部收到梯度。

**类别协变量已在 sampler 内建**。`RollingWindowDataset` 的 `categorical_past_covariates` / `categorical_known_covariates` 接收 `(C, L)` 的类别码，`_encode_categorical` 在窗口切出之后：

1. 对目标 context 做前向填充（`_ffill`），得到编码用的 target；
2. 逐通道调用 `timestral.covariate_encoding.target_encode`，与 `forecast` 的编码是同一个函数；
3. known 协变量按 `split=context_len` 分开编码，horizon 部分用 context 统计量（未见过类别回退到序列均值）；
4. `target` 与 code 同时缺失的位置保持 NaN（padding，而不是一个类别）；
5. 编码行 `_concat_rows` 追加在数值协变量行之后，行序与 `forecast` 的 `_cat_rows` 一致。

编码发生在 DataLoader 侧（采样时、CPU、逐序列 NumPy，`num_workers=0` 时就在主进程），不进入 `pack`，也不进梯度图。实测 `V = 1` 时 sampler 输出与推理侧编码逐元素相等（`rtol=0, atol=0`）。

**多变量目标（`V > 1`）**：推理侧的 `_encoding_target` 只支持单变量目标，`V > 1` 时 `categorical_*` 参数会原样把 code 当数值传入（推理代码未改动）。训练侧在 sampler 内按"每一步已观测变量的均值"池化后再编码；因此 `V > 1` 的推理应把 sampler 产出的编码行通过 `past_covariates` / `known_covariates` 传入，而不是 `categorical_*`。`tests/test_training.py` 用两条用例分别锁定：`V = 1` 时 sampler 与 `forecast` 逐元素相等（`rtol=0`），`V > 1` 时按池化规则编码而非透传。

### 7.2 对称平均与截断类后处理均已覆盖

`symmetric_averaging=True` 会对 `-context` 再跑一遍并反射平均。`sign_augment=True`（默认）以 50% 概率把样本连同全部协变量行取负，使负号分支进入训练分布。镜像样本是标签同时取负的对称增强，而 `asinh((x-loc)/scale)` 在取负下严格反对称（`loc` 反号、`scale` 不变），所以两者处在同一归一化流形上。

两个默认后处理现在也进入了训练与验证的同一路径：

- `clip_nonnegative`：`nonnegative_context(rows, mask)` 判定该序列已观测上下文是否全非负，是则在 0 处 `clamp_min`；
- `quantile_real_cap_k`：`_cap_quantiles` 按 `[min - k*s, max + k*s]` 截断，`s` 取最后一个 context patch 的 scale。

`Trainer._postprocess` 在 `packed` 上按行（每行即一个序列）执行，参数 `clip_nonnegative=True` / `quantile_real_cap_k=1e4` 与 `TimestralForecaster` 默认值一致。它不引入任何参数，但落在边界外的分位数在 clamp 之后梯度为零，即训练不会把容量花在部署时会被截掉的值上；同时它改变了部署输出，因此走同一路径后，**训练 loss 与验证指标和 `forecast` 处于同一口径**，此前"开启后验证指标与训练 loss 不可直接比较"的问题不再存在。实测 `_postprocess` 与 `forecast()` 的 `max|diff| = 0.000e+00`。

注意两点：

- `_cap_quantiles` 依赖 context 的 min/max/scale，因此它只在目标行上可用——协变量行在进入 `_postprocess` 之前已被 `role_ids` 过滤掉了。它在取负下上下界互换、对每个 pass 独立成立，所以训练的逐 pass 应用与推理的逐 pass 应用等价。
- `clip_nonnegative` 在推理里只作用于平均后的原始符号，训练对镜像样本不做 0 截断（`nonnegative_context(-ctx)` 为假），这正是镜像 pass 的行为；"先平均再截断"与"逐 pass 截断"不可交换，训练用逐样本逼近，差异只落在 0 附近。

### 7.3 显存与速度

- `detach_feedback=True` 时显存与 horizon 基本无关；设为 `False` 后会保留每个 block 的计算图。
- `_refine_block` 内部使用 float64 做 Welford 折叠，GPU 上吞吐低于 fp32；horizon 越长越明显。
- 每步 forward 除了模型内部的 `scale()`，还有两次额外的因果归一化：一次在 `forward_loss` 里取 `scale0`，一次在 `_postprocess → _cap_quantiles` 里取 anchor scale。二者都只依赖 context，是纯开销；horizon 很长时训练速度会明显低于纯推理路径。
- EMA 评估/保存：常驻一份 `self.ema`，`_ema_weights` 切换时再克隆一份备份，峰值约为参数量的 3 倍。

### 7.4 数据接口

- 数据集是虚拟无限长度，任何按 epoch 的调度器都不适用。
- 目标缺失位置不参与 loss；如果整个 batch 的 horizon 全缺失，分母被 clamp 到 1，loss 为 0，梯度为 0。
- `horizon_steps` 可覆盖 `horizon_patches * patch_size`，允许非 patch 整数倍的 horizon（如 129）：最后一块不足 `patch_size`，解码后被截断，正是 `forecast(prediction_length=129)` 的尾块形状。训练与推理的数值在 `horizon_steps=19, ps=8`（`tests/test_training.py`）与 `horizon_steps=100, ps=32` 下均实测 `max|diff| = 0.000e+00`。
- 多变量目标（`V > 1`）的类别协变量按已观测变量均值编码（训练侧）；此类序列在推理时应把编码行作为数值协变量传入，见 §7.1。

---

## 8. 复现与验证

### 8.1 运行示例

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe src\training\example_train.py
```

当前示例配置：16 条合成序列（每条 `2×600`，按 8:2 切成训练/验证）、`patch_size=32`、context 8 patch（256 步）、horizon 4 patch（128 步）、`block_size=2`、batch 8、20 步、`use_amp=True`（CUDA 上 bf16）、`ema_decay=0`；带一列 known-future 时间协变量，`sign_augment=True`，`clip_nonnegative` / `quantile_real_cap_k=1e4` 默认开启。示例本身未启用类别协变量与 `horizon_steps`，两者按 §7.1 / §7.4 的方式传入 `RollingWindowDataset` 即可。

### 8.2 已验证结果

| 项目 | 结果 |
|---|---|
| 参数量 | 11.1M（`small_config`）/ 209.31M（`base_config`） |
| 设备 | `cuda`（RTX 3060 Laptop） |
| 示例 20 步 | train 0.3507 → 0.2795；val 0.4018（step 10）→ 0.3772（step 20） |
| checkpoint → `from_pretrained` → `forecast` | `(2, 2, 128, 21)`，全有限 |
| 推理实际 block | `block_size=2`，`per_shot=2` |
| 类别编码 sampler vs `forecast`（`V = 1`） | `rtol=0` 逐元素相等 |
| 非整 patch horizon（100 步） | `max\|diff\| = 0.000e+00` |
| `_postprocess` vs `forecast` | `max\|diff\| = 0.000e+00`，clamp 占比 0.444 |
| 组合探针（类别 + 后处理 + 97 步） | loss 2.0058，梯度有限；`max\|diff\| = 0.0` |
| 全路径矩阵（11 个配置，见 §5.2） | 掩码 / 行角色 / known-future 块 / `loc` / `scale` / decode / postprocess / `forecast` 全部 `max\|diff\| = 0.000e+00` |
| `pytest tests/` | 19 passed |

### 8.3 回归测试

1. 采样多样性：同一 batch 的不同行不完全相同。
2. NaN 目标：loss 必须有限。
3. block 契约：`per_block_patches != config.block_size` 必须报错。
4. 训练/推理一致性：11 个配置的掩码、行角色、known-future 块、因果归一化、decode、后处理与 `forecast` 逐元素相等（`test_training_forward_matches_inference`，见 §5.2）。
5. 类别编码：`V = 1` 时 sampler 产出与 `forecast._encode_categorical` 逐元素相等；`V > 1` 时按观测变量均值池化编码（不再透传）。
6. 后处理：`_postprocess` 与 `forecast(cap + clip)` 逐元素相等（含在矩阵的 `clamps_bite` 配置内）。
7. 非整 patch horizon：`horizon_steps` 非 `patch_size` 整数倍时与 `forecast(prediction_length)` 相等（含在矩阵的 `no_cov_partial_horizon` / `known_partial_horizon` 配置内）。

第 1–3 项在 `tests/test_training.py`；第 4–7 项由 `test_training_forward_matches_inference` 的 11 个参数化配置覆盖；类别编码另由 `test_sampler_categorical_encoding_matches_forecast`、`test_sampler_pools_variates_for_categorical_encoding` 覆盖。

---

## 9. 代码索引

| 位置 | 内容 |
|---|---|
| `src/training/dataset.py:22-227` | `RollingWindowDataset`、类别编码、采样与 collate |
| `src/training/loss.py:12-49` | `pinball_loss`、mask 处理 |
| `src/training/model_utils.py:39-69` | `small_config` / `base_config` |
| `src/training/model_utils.py:71-121` | checkpoint 保存/加载 |
| `src/training/trainer.py:43-49` | `_split_batch`（2 元 / 4 元 batch） |
| `src/training/trainer.py:52-164` | `Trainer.__init__`、block 契约、EMA 初始化 |
| `src/training/trainer.py:166-185` | `_resume`（权重 + 优化器 + RNG） |
| `src/training/trainer.py:188-246` | `pack`（委托 `_assemble_rows`） |
| `src/training/trainer.py:249-303` | `_decode_train`（含协变量反馈分流） |
| `src/training/trainer.py:306-352` | `forward_loss`（含后处理） |
| `src/training/trainer.py:354-373` | `_postprocess`（cap + clip，复用推理实现） |
| `src/training/trainer.py:375-389` | `_negate_random_samples`（符号增强） |
| `src/training/trainer.py:392-401` | LR 调度 |
| `src/training/trainer.py:419-441` | `_ema_weights` |
| `src/training/trainer.py:443-460` | `evaluate` |
| `src/training/trainer.py:462-521` | `fit` |
| `src/training/trainer.py:523-537` | `save_checkpoint` |
| `src/training/example_train.py:64-136` | `_time_index` 与可运行示例 `main` |
| `src/timestral/covariate_encoding.py:17-66` | `target_encode`（sampler 与推理共用） |
| `src/timestral/model.py:122-132` | `scale`（因果 patch 归一化） |
| `src/timestral/model.py:135-152` | `_known_cov_scaling`（known 行整行统计） |
| `src/timestral/model.py:164-202` | `predict_block` 与分支 |
| `src/timestral/model.py:204-230` | `_predict_context_only` |
| `src/timestral/model.py:231-273` | `_predict_window`（协变量分支） |
| `src/timestral/forecaster.py:161-231` | `forecast` 公共入口 |
| `src/timestral/forecaster.py:233-244` | `_as_context`（缺失值零填充） |
| `src/timestral/forecaster.py:323-330` | `_cap_quantiles`（real cap 后处理） |
| `src/timestral/forecaster.py:332-394` | `_assemble_rows`（训练/推理共用行组装） |
| `src/timestral/forecaster.py:396-456` | `_block_ar_decode` |
| `src/timestral/forecaster.py:458-483` | `_refine_block`（advancing frame） |
