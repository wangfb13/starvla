# LeRobot v0.5.1 Sparky X5 改动清单与使用说明

本文档用于记录 `/mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5` 相对原始工程的本地改动、当前训练路径的关键行为，以及一些容易混淆的使用细节。

## 1. 适用范围

当前说明主要针对以下使用场景：

- `pi05 / pi0.5` 策略训练与微调
- `Sparky X5` 双臂数据
- 使用 `rotation6D` 表示旋转
- 使用 `use_relative_actions=true`
- 使用 `canonicalize_bimanual_rot6d=true`

如果后续训练配置不再满足上述条件，部分结论需要重新核对。

## 2. 当前工程主要改动

### 2.1 PI0.5 / PI05 双臂 canonical rot6d 支持

当前工程在 `pi05` 路径中增加了双臂 canonical 表示与 `rot6d` 支持，主要代码位置：

- `src/lerobot/policies/pi05/canonical_bimanual.py`
- `src/lerobot/policies/pi05/processor_pi05.py`
- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/factory.py`
- `src/lerobot/scripts/lerobot_train.py`

主要能力：

- 支持从原始 `quat(xyzw)` 转为 canonical `rot6d`
- 支持双臂 canonical action/state 映射
- 支持缺失臂用 mask 和 `missing_value` 处理
- 支持训练时按当前数据动态构造 canonical stats

### 2.2 相对动作与 canonical stats 在线构造

当训练满足以下条件时：

- `--policy.type=pi05`
- `--policy.use_relative_actions=true`
- `--policy.canonicalize_bimanual_rot6d=true`

训练脚本会在启动时调用：

- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/policies/pi05/canonical_bimanual.py::build_canonical_bimanual_stats`

直接从当前数据集的原始 `state/action` 读取数据，在线构造：

- canonical `observation.state` stats
- canonical `action` stats
- `use_relative_actions=true` 时的相对动作 stats

这意味着：

- 对于这条训练路径，`stats.json` 不是唯一统计来源
- 训练前不一定必须手工执行 `recompute_stats`
- 但如果你希望磁盘上的 `meta/stats.json` 本身就是 canonical/relative 口径，仍然可以离线重算

### 2.3 rot6d 在相对动作中默认排除

在 `canonicalize_bimanual_rot6d=true` 时，工程会自动把 `rot6d` 加入 `relative_exclude_joints`。

关键位置：

- `src/lerobot/policies/factory.py`
- `src/lerobot/policies/pi05/canonical_bimanual.py::extend_relative_exclude_joints`
- `src/lerobot/processor/relative_action_processor.py`

当前默认行为：

- `x/y/z` 走 relative
- `rot6d` 保持 absolute
- `gripper` 保持 absolute

这是刻意设计，不是偶然结果。原因是当前 relative processor 只支持逐维 `action - state`，这适合位置，不适合 `rot6d` 这种旋转流形表示。

## 3. 当前推荐训练配置

典型训练脚本见：

- `shfile/dlc_torchrun_train_pi05.sh`
- `shfile/dsw_train_pi05.sh`

当前推荐关键参数：

```bash
--policy.type=pi05
--policy.use_relative_actions=true
--policy.relative_exclude_joints='["gripper"]'
--policy.canonicalize_bimanual_rot6d=true
--policy.canonical_missing_value=9930.0
```

注意：

- 虽然脚本里只显式写了 `["gripper"]`
- 只要 `canonicalize_bimanual_rot6d=true`
- 代码会自动把 `rot6d` 也加入排除列表

## 4. 数据集元信息使用规则

### 4.1 `info.json` 的作用

`meta/info.json` 主要定义：

- 数据集 feature key
- feature dtype
- feature shape
- `action.names`
- `observation.state.names`

这些信息会直接影响：

- policy 输入输出 feature 推断
- state/action 名称映射
- canonical bimanual 的字段解析
- relative action mask 的维度含义

### 4.2 `stats.json` 的作用

`meta/stats.json` 主要用于：

- normalizer / unnormalizer
- state/action 的 min/max/mean/std/quantiles

如果 `use_relative_actions=true`，normalizer 实际需要的是相对动作分布的 stats，而不是绝对动作分布的 stats。

### 4.3 修改 `info.json` 后是否必须改 `stats.json`

结论：

- 如果只是删掉不用的 feature，`stats.json` 最好同步删掉对应 key
- 如果改了 `state/action` 的维度、顺序、语义或旋转表示，`stats.json` 应重算

不推荐只手改 `info.json` 而不处理 `stats.json`。

## 5. 训练时是否必须保持数据 key 与预训练数据完全一致

结论：不需要完全一致，但必须保持语义兼容。

### 5.1 可以不完全一致的部分

- 相机观测 key 可以不同
- 观测 key 可以通过 `rename_map` 适配

### 5.2 最好不要乱改的部分

- `observation.state`
- `action`

更重要的是：

- `action.names` 的顺序和语义要正确
- `observation.state.names` 的顺序和语义要正确

因为 `pi05` 的 relative action 和 canonical rot6d 逻辑都依赖这些名字做映射，而不是只看 shape。

## 6. `recompute_stats` 的作用与当前使用建议

函数位置：

- `src/lerobot/datasets/dataset_tools.py::recompute_stats`

它的主要作用：

- 重新扫描数据集
- 计算数值特征统计
- 更新 `meta/stats.json`

如果 `relative_action=True`，它会：

- 按 `chunk_size` 遍历合法 action chunk
- 计算 `action - current_state` 后的相对动作统计

### 6.1 当前工程里什么时候建议手工跑

建议手工跑 `recompute_stats` 的场景：

- 没有走 `canonicalize_bimanual_rot6d=true`
- 你修改了磁盘数据表示，希望 `stats.json` 永久正确
- 你希望其他脚本也直接复用更新后的 stats

### 6.2 当前工程里什么时候可以不先跑

对于当前推荐训练路径：

- `pi05`
- `use_relative_actions=true`
- `canonicalize_bimanual_rot6d=true`

可以不在训练前手工跑 `recompute_stats`，因为训练脚本会在线构造 canonical/relative stats 给 processor 使用。

## 7. rot6d 在 relative action 中应使用 absolute 还是 relative

当前建议：`rot6d` 用 `absolute`。

原因：

- 当前 relative processor 的实现是逐维减法
- `rot6d` 不是欧式位置向量
- 直接做 `rot6d_action - rot6d_state` 不等价于真实相对旋转

所以当前工程的正确做法是：

- 平移 `xyz` 使用 relative
- 旋转 `rot6d` 使用 absolute
- `gripper` 使用 absolute

如果后续确实要让旋转也使用“相对表示”，应新增专门的旋转相对 processor，而不是复用现有逐维减法。

## 8. 数据集改相机模态时的建议

例如从：

- `dag911262r00025a_lerobot_0429_193253_6_deformation`

切到：

- `dag911262r00025a_lerobot_0429_193253_6_deformation_cam_left_only`

建议直接使用已经处理好的新数据集目录，而不是在原目录手工修改 `info.json`。

推荐原因：

- `cam_left_only` 版通常已经同步处理过 `info.json`
- 也应同步处理 `stats.json`
- 手工修改原目录更容易造成元信息与真实 parquet 内容不一致

## 9. checkpoint 体积说明

以 `pi05` checkpoint 为例：

- `pretrained_model/model.safetensors` 很大，主要因为保存的是完整 `PaliGemma + Gemma expert + action head`
- 不是只保存了一个小 adapter
- `training_state/optimizer_state.safetensors` 往往比模型权重更大

常见现象：

- 模型权重大约数 GB 到十 GB
- 优化器状态可进一步占十几 GB

因此 checkpoint 大并不一定异常，更多是完整模型与优化器状态共同造成的。

## 10. 常见判断表

### 10.1 是否需要手工 `recompute_stats`

- 普通 `pi0.5/pi05` + `use_relative_actions=true`：建议需要
- 当前 `pi05 + canonicalize_bimanual_rot6d=true` 训练路径：训练本身不强制需要
- 修改了 `stats.json` 供其他流程长期复用：建议需要

### 10.2 是否要让 rot6d 参与 relative

- 当前工程：不要
- 当前推荐做法：排除 rot6d，相对化只作用在位置

### 10.3 是否必须保持预训练数据 key 完全一致

结论：不需要“名字逐字完全一致”，但必须保证训练实际读取到的 key、shape、names、语义兼容。

- 相机 key 可以不同，但要能通过 `rename_map` 或数据集 feature 配置对齐
- `observation.state` / `action` 的维度语义、`names` 顺序不能乱
- 对于 `pi05 + canonicalize_bimanual_rot6d`，双臂字段名应满足 canonical 解析逻辑

## 11. 多数据集 root 训练支持

当前工程已支持只配置 `dataset.root` 来训练多数据集。

### 11.1 支持的两种 `dataset.root` 形式

- `dataset.root` 直接指向一个完整 LeRobot 数据集目录
- `dataset.root` 指向一个父目录，下面每个子目录都是独立 LeRobot 数据集

例如：

```bash
--dataset.root=/mnt/tacumi_data/lerobot_datasetV30_train/extract_dataset/train_0512
```

如果 `train_0512` 下面存在多个子目录，且每个子目录都包含：

- `meta/info.json`
- `meta/stats.json`
- `data/`

训练时会自动走多数据集加载逻辑，不再强制要求显式传 `--dataset.repo_id=...`。

### 11.2 当前多数据集模式的行为

- 会自动扫描 `dataset.root` 下所有合法子数据集
- 训练时按顺序拼接成一个 `MultiLeRobotDataset`
- 会聚合 `meta.stats`
- 会聚合 `meta.episodes`
- 会构造兼容训练脚本的 `dataset.meta`

因此当前训练入口里依赖的这些逻辑都能继续工作：

- `make_policy(..., ds_meta=dataset.meta)`
- `dataset.meta.stats`
- `dataset.meta.episodes["dataset_from_index"]`
- `dataset.meta.episodes["dataset_to_index"]`

### 11.3 与 rot6d / relative pose 的兼容性

这次多数据集支持同时兼容了：

- `pi05`
- `canonicalize_bimanual_rot6d=true`
- `use_relative_actions=true`

具体兼容点：

- canonical rot6d stats 重算现在会遍历所有子数据集，而不是只假设存在单个 `dataset.hf_dataset`
- `rot6d` 自动加入 `relative_exclude_joints` 的逻辑没有改
- 也就是说当前仍然是：
  - `xyz` relative
  - `rot6d` absolute
  - `gripper` absolute

### 11.4 当前限制

当前多数据集 root 模式下，暂不建议同时使用：

- `dataset.streaming=true`
- `dataset.episodes=[...]`

这两个场景目前没有一起做兼容，代码会直接报明确错误，而不是静默走错逻辑。

### 11.5 推荐用法

如果你的目录结构就是：

```text
train_0512/
  dataset_a/
  dataset_b/
  dataset_c/
```

那么训练时推荐直接写：

```bash
src/lerobot/scripts/lerobot_train.py \
  --dataset.root=/mnt/tacumi_data/lerobot_datasetV30_train/extract_dataset/train_0512 \
  --policy.type=pi05 \
  ...
```

不再需要额外把这些子数据集合并成一个大目录后再训练。

## 12. 建议的本地改动检查清单

- [ ] 训练脚本确认使用 `--policy.canonicalize_bimanual_rot6d=true`
- [ ] 训练脚本确认 `--policy.use_relative_actions=true`
- [ ] 确认当前数据集的 `action.names` 与 `observation.state.names` 顺序正确
- [ ] 确认 `info.json` 与真实 parquet 列一致
- [ ] 如裁剪模态，确认 `stats.json` 已同步处理
- [ ] 如不走在线 canonical stats，提前执行 `recompute_stats`
- [ ] 确认 `rot6d` 未被错误地当作逐维 relative 处理
- [ ] 确认 checkpoint 保存频率与磁盘空间匹配

## 13. 关键代码位置速查

- 训练入口：`src/lerobot/scripts/lerobot_train.py`
- policy feature 构造：`src/lerobot/policies/factory.py`
- PI05 processor：`src/lerobot/policies/pi05/processor_pi05.py`
- canonical rot6d 逻辑：`src/lerobot/policies/pi05/canonical_bimanual.py`
- relative action processor：`src/lerobot/processor/relative_action_processor.py`
- stats 重算：`src/lerobot/datasets/dataset_tools.py`
- 相对动作 stats：`src/lerobot/datasets/compute_stats.py`

## 14. 后续维护建议

如果继续在当前工程上迭代，建议每次改动都更新本文档，至少补齐以下内容：

- 改了哪些训练参数默认值
- 改了哪些 dataset meta 规则
- 改了哪些 state/action 表示
- 是否影响 `stats.json` 的生成口径
- 是否影响 checkpoint 加载兼容性
