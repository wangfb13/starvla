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

### 2.3 rot6d / gripper 的 chunk relative 口径

当前推荐训练脚本已经启用 `relative_action_keyframes='["0:3","10:13"]'`，canonical `xyz + rot6d` block 会先走 SE(3)-aware relative：

```text
rel_pos = R_state^T @ (target_pos - state_pos)
rel_rot = R_state^T @ R_target
```

`gripper` 不在 keyframe 内，默认通过普通 chunk relative 处理，也就是连续量 `action_gripper - state_gripper`。如果希望夹爪保持 absolute，可以在启动脚本中覆盖：

```bash
RELATIVE_EXCLUDE_JOINTS='["gripper"]'
```

在 `canonicalize_bimanual_rot6d=true` 的旧在线 canonicalize 路径中，工程仍会自动把 `rot6d` 加入 `relative_exclude_joints`，这是为了防止没有设置 keyframes 时误用逐维 `rot6d_action - rot6d_state`。

关键位置：

- `src/lerobot/policies/factory.py`
- `src/lerobot/policies/pi05/canonical_bimanual.py::extend_relative_exclude_joints`
- `src/lerobot/processor/relative_action_processor.py`

当前推荐脚本行为：

- `xyz + rot6d` 走 SE(3)-aware chunk relative
- `gripper` 走普通连续量 chunk relative
- `relative_exclude_joints=[]`

这样既避免了 `rot6d` 的错误逐维相减，又能让所有 action 维度参与 chunk relative 训练。

### 2.4 PI05 的 state 输入现已可配置

当前工程为 `pi05 / pi0.5` 增加了：

- `--policy.use_state_input=true|false`

对应代码位置：

- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/pi05/processor_pi05.py`
- `src/lerobot/policies/factory.py`
- `tests/policies/pi0_pi05/test_pi05.py`

当前行为分为两层：

- `use_state_input=true` 时：
  - 会读取 `observation.state`
  - 会先把 state 归一化后离散化
  - 再拼进 prompt，格式类似 `Task: ... , State: ...; Action:`
- `use_state_input=false` 时：
  - 不再把 state 拼进语言 prompt
  - prompt 退化为纯任务文本，格式类似 `Task: ...` 后接 `Action:`

但要特别注意，`use_state_input=false` 只表示“不把 state 当作语言输入拼进模型 prompt”，不等价于“训练/推理全流程都不再需要 `observation.state`”。

当前代码里，`pi05` 是否仍然需要 `observation.state`，由下面三个开关共同决定：

- `use_state_input`
- `use_relative_actions`
- `canonicalize_bimanual_rot6d`

也就是说，只有在以下条件同时满足时，当前工程才允许彻底不提供 `observation.state`：

- `--policy.use_state_input=false`
- `--policy.use_relative_actions=false`
- `--policy.canonicalize_bimanual_rot6d=false`

如果下面任意一个开关仍为 `true`：

- `--policy.use_relative_actions=true`
- `--policy.canonicalize_bimanual_rot6d=true`

那么当前工程仍会把 `observation.state` 视为必需输入 feature，即使它不会被拼进 prompt。

另外，这次也补了一个兼容性修复：当从已有 checkpoint / pretrained processor 加载时，`factory.py` 会把当前配置里的 `use_state_input` 覆盖写回 `pi05_prepare_state_tokenizer_processor_step`，避免旧 processor 配置把 prompt 行为锁死在保存时的状态。

### 2.5 新增 `pi05_full` policy 目录与策略入口

当前工程已新增一个独立的：

- `src/lerobot/policies/pi05_full/`

它是基于当前仓库里的 `pi05` 目录复制出来的独立 policy 包，已经包含：

- `configuration_pi05.py`
- `modeling_pi05.py`
- `processor_pi05.py`
- `canonical_bimanual.py`
- `README.md`

这次改动的初衷是先在当前工程里预留一个可独立演进的 `pi05_full` 策略入口，避免后续继续直接改 `pi05` 本体。当前在这个基础上，已经额外迁入了 `vtla_share` 里与 `user_prompt / subtask` 相关的“数据读取、prompt 组织、token 化、模型取 language token”这条最小可用链路。

当前已经完成的接线包括：

- `PreTrainedConfig.register_subclass("pi05_full")`
- `PI05FullConfig`
- `PI05FullPolicy`
- `make_pi05_full_pre_post_processors`
- `lerobot.policies.factory.get_policy_class("pi05_full")`
- `lerobot.policies.factory.make_policy_config("pi05_full")`
- `lerobot.policies.factory.make_pre_post_processors(...)`

因此当前已经可以使用：

```bash
--policy.type=pi05_full
```

来实例化这个新策略。

当前这个 `pi05_full` 不再只是“纯复制版的当前 `pi05`”，而是已经比 `pi05` 多了一层 `user_prompt / subtask` 处理能力。具体新增行为如下：

- `TokenizerProcessorStep` 已支持读取 `complementary_data.user_prompt`
- `TokenizerProcessorStep` 已支持读取 `complementary_data.subtask`
- `subtask` token 化后会自动补 `EOS token`
- `pi05_full` 的预处理会优先读取 `user_prompt`
- 如果没有 `user_prompt`，会自动 fallback 到 `task`
- `pi05_full` 的预处理会优先读取 `next_subtask`
- 如果没有 `next_subtask`，会依次 fallback 到：`subtask`、`next_subtask_text_sidecar`、`subtask_text_sidecar`
- 如果以上字段都没有，并且开启默认 fallback，会退化为使用 `user_prompt / task` 本身生成 `subtask`
- 单样本推理路径下，`AddBatchDimensionComplementaryDataStep` 也已补充支持：`user_prompt`、`subtask`、`next_subtask`、`next_subtask_text_sidecar`、`subtask_text_sidecar`

模型侧当前也做了一个很重要的接线：

- `pi05_full` 在训练和推理时，会优先使用 `observation.user_prompt.tokens`
- 如果没有 `user_prompt tokens`，则自动回退到原来的 `observation.language.tokens`
- `pi05_full` 训练时已经会把 `observation.subtask.tokens` 接入 prefix
- `subtask` token 在 prefix 中采用“可看上下文、对子任务自身保持 causal”的 attention 方式
- `pi05_full` 当前已经支持 `subtask` 的辅助交叉熵训练 loss
- `pi05_full` 推理时已经支持显式自回归生成下一个 `subtask`
- 总训练 loss 现在由：
  - 动作 flow loss
  - `loss_weight_subtask_ce * subtask_ce_loss`
 共同组成

这意味着：

- `user_prompt` 现在已经会真实影响 `pi05_full` 的训练输入
- 不再只是“把字段读出来但模型没用”
- `subtask` 现在也已经不只是“被 token 化存起来”
- 它已经会真实参与训练反向传播
- `pi05_full` 在推理时，如果没有可用的显式 `subtask`，已经会先生成 `subtask` 再采样动作

但也要明确当前边界：

- 这次已经迁入了 `subtask prediction loss` 和“推理时显式自回归生成下一个 `subtask`”这两部分
- 但还没有把 `vtla_share` 那套完整的 FAST action token 支路一起迁入
- 当前 `subtask` loss 是作为 `pi05_full` 的辅助训练目标存在
- 目前新增的推理生成逻辑只挂在 `pi05_full` 上，不会影响当前仓库里的其他 policy

当前 `pi05_full` 推理期关于 `subtask` 的优先级和缓存规则如下：

- 如果 batch 中提供了显式 `subtask tokens`，并且 `pi05_full.subtask_is_fallback=false`，则直接使用这份显式 `subtask`
- 如果没有显式 `subtask`，或者该 `subtask` 只是由 `task/user_prompt` fallback 出来的占位文本，则会触发自回归生成
- 生成结果会缓存在 policy 内部
- 当 `user_prompt/task` 发生变化时，缓存会自动失效并重新生成
- 当调用 `policy.reset()` 时，缓存会被清空
- 当距离上一次生成的时间超过 `--policy.subtask_regeneration_interval` 时，会重新生成
- 如果把 `--policy.subtask_regeneration_interval=0`，则表示每次推理都重新生成

当前与这部分推理行为直接相关的新配置项有：

- `--policy.max_decoding_steps`
- `--policy.temperature`
- `--policy.subtask_regeneration_interval`

另外，这次复制目录后还处理了一个很关键的注册冲突问题。由于 `pi05` 和 `pi05_full` 都会向 `ProcessorStepRegistry` 注册 processor step，如果继续沿用原来的 step 名称，导入时会因为重名而直接报错。因此 `pi05_full` 路径下的 step 注册名已经单独改为：

- `pi05_full_canonicalize_bimanual_rot6d`
- `pi05_full_apply_masks_processor_step`
- `pi05_full_prepare_state_tokenizer_processor_step`

同时，`factory.py` 中加载 pretrained processor 的 override key 也已同步切换到这组 `pi05_full_*` 名字，避免和原 `pi05` 互相覆盖。

另外又补了一层向后兼容：如果 `pi05_full` 训练继续复用旧的 `pi05_base` pretrained directory，而其中保存的 `policy_preprocessor.json` 仍然使用老 step 名 `pi05_prepare_state_tokenizer_processor_step`，当前实现不会再强行加载这套 legacy processor。`factory.py` 会先检查已保存的 processor step keys；一旦发现这是旧 `pi05` processor，就保留 pretrained 模型权重加载，但改为按当前 `pi05_full` config 重新构建 pre/post processor。这样 `use_relative_actions=false` 的 absolute pose 训练也能顺利启动，不会再因为 override key 和 saved processor step name 不匹配而在初始化阶段报 `KeyError`。

当前测试也已经从最小注册测试扩展为覆盖 `pi05_full user_prompt/subtask` 逻辑的回归测试：

- `tests/policies/pi0_pi05/test_pi05_full.py`

当前覆盖点包括：

- `make_policy_config("pi05_full")`
- `get_policy_class("pi05_full")`
- `user_prompt` 优先级
- `subtask` 多级 fallback
- `subtask fallback` 标记透传
- `subtask` token 的 `EOS` 补齐
- `subtask` prefix attention 的 causal 约束
- `subtask` loss 与总训练 loss 的聚合
- `pi05_full` 模型优先使用 `user_prompt tokens`
- `make_pi05_full_pre_post_processors(...)` 在 `use_state_input=false` 下对新字段的兼容性
- `reset()` 清理 subtask cache
- 显式 `subtask` 优先于推理生成
- fallback `subtask` 触发推理生成
- `subtask_regeneration_interval` 控制缓存复用

在本机上，已使用环境：

- `/root/miniconda3/envs/lerobot_v0.5.1`

做过实际回归验证，确认：

- `make_policy_config("pi05_full")` 返回 `PI05FullConfig`
- `cfg.type == "pi05_full"`
- `get_policy_class("pi05_full")` 返回 `PI05FullPolicy`
- `user_prompt / subtask` 预处理逻辑工作正常
- `pi05_full` 会优先消费 `user_prompt tokens`
- `subtask` 已参与训练 loss 计算
- `pi05_full` 推理时会在需要时自回归生成 `subtask`
- 显式 `subtask` 与 fallback `subtask` 的推理分流逻辑工作正常
- `subtask` 推理缓存与 interval 刷新逻辑工作正常

本机当前已在环境：

- `/root/miniconda3/envs/lerobot_v0.5.1`

下直接执行过 `pytest` 回归；本轮与 `pi05_full user_prompt/subtask` 相关的关键用例已通过。

### 2.6 `subtasks.parquet` sidecar 数据兼容增强

这次还补了一块很实用的兼容逻辑，目的是让 `pi05_full` 可以直接消费“只有 `meta/subtasks.parquet`，但 `data` 主表里没有 `subtask_index` 字段”的数据集。

当前支持的行为是：

- 如果数据本身带 `subtask_index`，仍然优先走原来的索引读取逻辑
- 如果没有 `subtask_index`，但存在 `meta/subtasks.parquet`
- 且 `subtasks.parquet` 里至少有 `id` 和 `frames` 两列
- 则 `DatasetReader` 会按 `id` 中的 `ep{episode}` 信息和 `frames=[start, end]` 自动建立每个 episode 的子任务区间映射
- 命中区间后，会把对应文本注入样本：
  - `subtask`
  - `next_subtask`
  - `subtask_text_sidecar`
  - `next_subtask_text_sidecar`

同时，为了让这些字段能一路传到 `pi05_full` 预处理，当前还同步补了两层透传：

- `LeRobotDatasetMetadata.canonicalize_item(...)` 现在会保留：
  - `next_subtask`
  - `user_prompt`
  - `subtask_text_sidecar`
  - `next_subtask_text_sidecar`
- `batch_to_transition(...)` / complementary data 现在也会继续保留这些 language sidecar 字段

这意味着现在的数据兼容范围更大了：

- 不要求数据转换阶段必须先把 `subtask_index` 写进 parquet 主表
- 只要 `meta/subtasks.parquet` 标注完整，`pi05_full` 就能在训练侧读到 `subtask / next_subtask`
- 上层 `TokenizerProcessorStep` 的多级 fallback 也能直接消费这些 sidecar 文本

本次已用真实数据做过验证：

- 数据路径：
  - `/mnt/cpfs/datasets/pi05_groupA_train_roots_symlink/004__dag911262r00025a_lerobot_0411_055753_train_ready`
- 实际确认：
  - `storage_features` 中没有 `subtask_index`
  - 但 `dataset[idx]` 已可正确返回 sidecar 解析出的 `subtask` 和 `next_subtask`
- 实际样例包括：
  - `idx 0`:
    `subtask = Close the gripper to secure the red cup firmly`
    `next_subtask = Move the cup steadily toward the paper plate`
  - `idx 312`:
    `subtask = Lower the red cup onto the paper plate`
    `next_subtask = Lift the red cup with spoon and chopsticks inside`

这部分本地也补了对应回归测试：

- `tests/datasets/test_subtask_dataset.py`
- `tests/processor/test_converters.py`

注意：

- `subtasks.parquet` 中的 `frames` 在真实数据里可能是 `numpy.ndarray`，不是纯 Python `list`
- 这次已经兼容了 `list / tuple / np.ndarray`
- 如果未来 sidecar 的 `id` 命名不再包含 `ep{episode}` 片段，则这条自动映射逻辑需要跟着调整

目前这块验证主要是通过：

- sidecar-only 单测
- converter 透传单测
- 真实数据集抽样打印

来完成的。完整视频链路的全量回归在当前环境里仍受 `torchcodec / ffmpeg / libstdc++` 依赖限制，但不影响这次 `subtask sidecar -> pi05_full 输入` 这条主路径确认。

## 3. 当前推荐训练配置

典型训练脚本见：

- `shfile/dsw_train_pi05.sh`：旧在线 canonical pi05 训练入口。
- `shfile/dsw_training101_pi05_umi_groupA_18.5h.sh`：当前 sidecar pi05 训练入口。
- `shfile/dlc_torchrun_train_pi05full.sh`：当前 DLC 多机 sidecar pi05_full 训练入口。

当前推荐关键参数：

```bash
--policy.type=pi05
--policy.use_relative_actions=true
--policy.relative_action_keyframes='["0:3","10:13"]'
--policy.relative_exclude_joints='[]'
--policy.canonicalize_bimanual_rot6d=true
--policy.canonical_missing_value=9930.0
```

注意：

- 虽然脚本里显式写了 `relative_exclude_joints=[]`
- `relative_action_keyframes` 会先处理 canonical `xyz + rot6d`
- 代码仍保留自动把 `rot6d` 加入排除列表的保护逻辑
- keyframe dims 会优先按 SE(3) relative 处理，不会回到逐维相减
- `gripper` 默认不排除，因此会作为连续量做 chunk relative
- 上面这条推荐配置默认仍然需要 `observation.state`
- 即使你额外设置 `--policy.use_state_input=false`，只要 relative/canonical 任一开着，当前实现里也不能把 `observation.state` 从数据里彻底去掉

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

当前建议：`rot6d` 用 SE(3)-aware relative，不要做逐维相减。

原因：

- 普通 relative processor 的实现是逐维减法
- `rot6d` 不是欧式位置向量
- 直接做 `rot6d_action - rot6d_state` 不等价于真实相对旋转
- 现在已经增加 `relative_action_keyframes`，可以对 `xyz + rot6d` block 做真实相对位姿变换

所以当前工程的正确做法是：

- 平移 `xyz` 使用 SE(3)-aware relative
- 旋转 `rot6d` 使用 SE(3)-aware relative
- `gripper` 默认使用普通连续量 relative

关键是必须通过 `relative_action_keyframes` 指定 `xyz + rot6d` block，不能让 `rot6d` 复用旧的逐维减法。

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
- `rot6d` 自动加入 `relative_exclude_joints` 的保护逻辑仍保留
- 推荐脚本会额外设置 `relative_action_keyframes='["0:3","10:13"]'`
- 当前推荐口径是：
  - `xyz + rot6d` 走 SE(3)-aware relative
  - `gripper` 走普通连续量 relative

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


## 15. umi数据多文件夹训练方法
  - step1：先挑选用来训练的数据，将所有数据的绝对目录一行一个写到一个txt文件中，如/shfile/groupA_roots.txt
  - step2：执行下面的sh文件，创建数据的软链接
    cd /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5
    sh create_dataset_symlinks_from_txt.sh groupA_roots.txt /mnt/cpfs/datasets/pi05_groupA_train_roots_symlink
  - 训练时指定新创建的软链接目录，如--dataset.root=/mnt/cpfs/datasets/pi05_groupA_train_roots_symlink

## 16. UMI数据坐标对齐与action_gap_frames功能

### 16.1 为什么需要先做坐标对齐

UMI 数据中的 `observation.state` 和 `action` 都是在设备开机后由 VSLAM 建图得到的世界坐标系下记录的。由于每次开机或每个回合建立的世界坐标系可能不同，不同轨迹之间的坐标原点和朝向不一定一致。如果直接混在一起训练，模型会看到多套不同的世界坐标系，学习难度会明显增加。

这里需要特别注意位置和姿态的含义不完全一样。位置 `x/y/z` 可以比较直观地以第一帧作为零点，让每条轨迹都从 `(0, 0, 0)` 附近开始；但姿态中的 `roll/pitch` 不能简单理解为也应该从 0 开始。UMI 的 `roll/pitch` 是根据设备初始姿态和重力方向计算出来的，它们反映的是手持设备相对重力的倾斜状态。如果视频开始时设备本身就是倾斜的，比如手持时已经倾斜了 30 度，那么 `pose_data.csv` 第一帧里对应的角度也会接近 30 度，而不是 0 度。

因此，坐标对齐时更准确的理解是：对每条轨迹建立一个以第一帧为参考的局部坐标系。位置可以从第一帧归零；姿态则需要根据第一帧姿态做相对变换，保留和重力对齐后的物理含义，避免把设备初始倾斜误当成模型应该学习的任务差异。处理后，每条轨迹都变成“从第一帧出发的局部绝对轨迹”，不同回合之间就可以放到统一口径下训练。

这种处理不会限制后续训练方式。后续既可以直接使用“以第一帧为零点的局部绝对坐标”，也可以在此基础上继续构造相对动作、gap frame 动作或其他坐标表示，具体方式根据任务需求选择。

### 16.2 action_gap_frames的作用

`action_gap_frames` 用来控制“用当前帧状态/图像预测未来第几帧开始的 action”。例如 `action_gap_frames=6` 且 `chunk_size=50` 时，当前样本对应的 action chunk 会从当前帧之后第 6 帧开始取，范围相当于 `[t+6, t+7, ..., t+55]`。

这个功能主要用于处理 observation 和 action 之间的时间对齐问题。对于 UMI 数据，`state` 和 `action` 往往来自同一时刻的轨迹记录，并不天然包含机器人“下发动作到执行完成”的延时；如果希望模型从当前观测预测稍后会执行到的位置，就可以通过 `action_gap_frames` 把监督信号往未来移动。对于当前真机 LeRobot 数据，记录到的 `state/action` 已经考虑了控制延时，因此训练加载器会根据 `robot_type=spark3_pose_gripper` 自动把该子数据集的 action gap 设为 0。

### 16.3 训练配置建议

  - 单一 gap 设置：使用 `--policy.action_gap_frames=<gap>`。
  - UMI + 真机混训时，可以直接设置 UMI 所需的全局 gap，例如 `--policy.action_gap_frames=6`；真机数据会自动补 `action_gap_dataset_overrides=<dataset_index>:0`，UMI 保持全局 gap。
  - 多数据集混训且需要手动覆盖某个数据源 gap：使用 `--policy.action_gap_dataset_overrides=<dataset_index>:<gap>`。显式配置优先级高于自动推断。
  - 如果为了对齐 UMI 数据，需要用当前真实末端位姿生成 action，建议在数据预处理阶段显式生成和 `state` 同口径的 `action`，再由 `action_gap_frames` 决定训练时取未来多少帧作为监督。
  - 修改 gap 后需要重新确认 `delta_timestamps`、`chunk_size` 和离线评估脚本是否使用同一套配置，避免训练和评估取到不同帧。

## 17. UMI train_ready单手和双手任务key值不统一问题解决

### 17.1 问题背景

UMI train_ready 数据里，左手单臂、右手单臂、双手任务的视觉和触觉 key 并不完全一致。典型问题包括：双手数据的鱼眼相机叫 `observation.images.left_cam_left` / `observation.images.right_cam_right`，而单手数据使用 `observation.images.cam_left` / `observation.images.cam_right`；右手单臂数据中的触觉 key 仍然使用了类似左手单臂的名字，如果直接混训，模型可能把右手触觉误认为左手触觉。

### 17.2 统一后的canonical key

混训时统一暴露下面 6 个视觉/触觉 key：

```text
observation.images.cam_left
observation.images.cam_right
observation.depth_deformation.tactile_left_left
observation.depth_deformation.tactile_left_right
observation.depth_deformation.tactile_right_left
observation.depth_deformation.tactile_right_right
```

具体映射规则：

  - 双手相机 `observation.images.left_cam_left` 会映射为 `observation.images.cam_left`。
  - 双手相机 `observation.images.right_cam_right` 会映射为 `observation.images.cam_right`。
  - 右手单臂触觉旧 key 会重命名到右手 canonical key，不再复制保留旧 key，避免同一份右手触觉同时被当成左手触觉。
  - 左手单臂缺失右相机/右触觉时，会补空 tensor，并通过 `*_is_missing=True` 标记。
  - 右手单臂缺失左相机/左触觉时，会补空 tensor，并通过 `*_is_missing=True` 标记。
  - 双手数据 6 路 key 都应为 `*_is_missing=False`。

### 17.3 读取逻辑上的关键约束

不要直接修改原始数据集文件，也不要要求所有子数据集在磁盘上提前改成统一 key。正确逻辑是：每个子数据集仍然按照它自己原始的 `info.json` / `reader_features` 去读取数据；只有在 item 已经从子数据集读出来之后，`MultiLeRobotDataset` 才把 key 映射成统一 canonical key。

`delta_timestamps` 也遵循同样原则：训练配置侧可以使用统一 canonical key，但传给每个子 reader 时会反向映射回该子数据集真实存在的原始 key。例如双手数据读取时仍使用 `left_cam_left/right_cam_right`，右手单臂读取时仍使用它原始的触觉 key。

### 17.4 当前验证结果

已使用实际数据 `/mnt/cpfs/datasets/pi05_groupA_train_roots_symlink` 验证：

```text
repo_count: 15
total_frames: 1993863
total_episodes: 2332
```

抽样验证结果：

```text
左手单臂: cam_left=False, cam_right=True,  left tactile=False, right tactile=True
双手任务: cam_left=False, cam_right=False, left tactile=False, right tactile=False
右手单臂: cam_left=True,  cam_right=False, left tactile=True,  right tactile=False
```

其中 `False` 表示该模态真实存在，`True` 表示该模态缺失并由空 tensor 填充。三类样本已验证可以正常 batch collate：

```text
observation.images.cam_left batch_shape=(3, 3, 224, 224)
observation.images.cam_right batch_shape=(3, 3, 224, 224)
all tactile batch_shape=(3, 3, 96, 128)
```

相关代码位置：

  - key 统一逻辑：`src/lerobot/datasets/multi_dataset.py`
  - 空视觉/触觉 tensor shape 逻辑：`src/lerobot/datasets/dataset_metadata.py`
  - 单元测试：`tests/datasets/test_datasets.py`

## 18. PI05/PI05_FULL canonical sidecar 训练入口

### 18.1 预处理输出

`scripts/build_pi05_canonical_sidecar.py` 会为每个数据集目录生成一个 sidecar 子目录，默认结构如下：

```text
{sidecar_root}/{dataset_root.name}/framewise_guidance.parquet
{sidecar_root}/{dataset_root.name}/stats.json
{sidecar_root}/{dataset_root.name}/stats_relative_action_chunk{N}_gap{G}.json
{sidecar_root}/{dataset_root.name}/summary.json
```

其中 `framewise_guidance.parquet` 覆盖训练时读取到的 `observation.state` 和 `action`：

- `observation.state`: canonical 22D，左手 10D + 右手 10D + 左右手有效位。
- `action`: canonical 20D，左手 10D + 右手 10D。
- 每只手 10D 为 `xyz + rot6d(6) + gripper`。
- `state_mask` / `action_valid_mask` 会在加载时映射成 `observation.state_mask` / `observation.action_mask`，用于 pi05/pi05_full mask step。

UMI 数据在预处理阶段会先按每个 episode 第一帧有效 state pose 做相对位姿，再转成 rot6d；同时会对 UMI 原始 state/action gripper 先裁剪到 `[-0.436, 0.0]`，再统一除以 `-0.436` 归一化到 `[0, 1]`，也就是 `0 -> 0`、`-0.436 -> 1`。真机数据保持原始坐标系，只做 canonical + quat 转 rot6d，gripper 也保持原始 framewise 数值。

### 18.2 训练参数

现在训练入口支持直接指定 roots txt，不再要求所有数据先放在同一个父目录下：

```bash
--dataset.roots_file=/mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/groupA_roots.txt \
--dataset.sidecar_root=/mnt/cpfs/repo/pi05_canonical_sidecars_groupA
```

`dataset.roots_file` 一行一个本地 LeRobot 数据集根目录，可以是不同父目录下的路径，但最后一级目录名必须唯一。sidecar 会按 `{sidecar_root}/{dataset_root.name}/framewise_guidance.parquet` 自动解析。

单数据集调试时也可以直接指定一个 parquet：

```bash
--dataset.root=/path/to/one_lerobot_dataset \
--dataset.sidecar_path=/path/to/sidecar/framewise_guidance.parquet
```

这种模式下 stats 会从 parquet 同目录自动解析：未开启 `use_relative_actions` 时读取 `stats.json`，开启时优先读取 `stats_relative_action_chunk{N}_gap{G}.json`。如需完全手动指定，也可以使用 `dataset.stats_path` 或 `dataset.relative_stats_path` 覆盖。

使用 canonical sidecar 训练时，sidecar 已经提供 22D/20D canonical state/action，因此需要关闭在线 canonicalize：

```bash
--policy.canonicalize_bimanual_rot6d=false \
--policy.max_state_dim=22 \
--policy.max_action_dim=20
```

如果训练开启：

```bash
--policy.use_relative_actions=true
```

混合训练时 action chunk 的有效 gap 按数据集类型决定：

- UMI 数据：`robot_type` 为 `ugripper_left` / `ugripper_right` / `dual_ugripper`，使用全局 `policy.action_gap_frames`。
- 真机数据：`robot_type=spark3_pose_gripper`，默认使用 `gap=0`，因为记录的 `state/action` 已经考虑了延时。
- 如果显式传入 `policy.action_gap_dataset_overrides`，显式值优先。

normalization stats 会使用与有效 gap 对齐的 relative action 口径：开启 `use_relative_actions` 时，加载器会优先读取与训练 `chunk_size` 和有效 action gap 一致的 `stats_relative_action_chunk{N}_gap{G}.json`，并只使用其中的 `observation.state` / `action` 作为 normalization stats。如果匹配文件不存在，会基于 `framewise_guidance.parquet` 动态计算 relative action stats，不会退回普通 `stats.json`。因此真机数据会使用 `gap=0` 的 relative stats 口径，UMI 数据会使用全局 gap 的 relative stats 口径。

当前 sidecar canonical 训练默认使用 SE(3)-aware relative action keyframes：

```bash
--policy.relative_action_keyframes='["0:3","10:13"]'
--policy.relative_exclude_joints='[]'
```

其中 `0:3` 表示左手 `xyz` 从 0 开始、`rot6d` 从 3 开始，`10:13` 表示右手 `xyz` 从 10 开始、`rot6d` 从 13 开始。keyframe 内部会做：

```text
rel_pos = R_state^T @ (target_pos - state_pos)
rel_rot = R_state^T @ R_target
```

因此 `rot6d` 不再走旧的逐维相减；`gripper` 不在 keyframe 内，默认通过 `relative_exclude_joints=[]` 进入普通连续量 chunk relative。如果要让夹爪保持 absolute，可手动覆盖 `RELATIVE_EXCLUDE_JOINTS='["gripper"]'`。

当 sidecar stats 成功加载或动态计算完成时，pi05/pi05_full 会跳过训练启动阶段耗时的 `build_canonical_bimanual_stats` 全量扫描，直接使用 sidecar 口径的 stats，从而避免此前 `canonicalize_bimanual_rot6d=true + use_relative_actions=true` 启动很久的问题。

### 18.3 示例脚本

已更新：

```text
shfile/dsw_training101_pi05full_umi_groupA_18.5h.sh
shfile/dsw_training101_pi05_umi_groupA_18.5h.sh
```

脚本顶部可修改：

```bash
DATASET_ROOTS_FILE="/mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/groupA_roots.txt"
CANONICAL_SIDECAR_ROOT="/mnt/cpfs/repo/pi05_canonical_sidecars_groupA"
```

如果实际 sidecar 输出目录不同，只需要改 `CANONICAL_SIDECAR_ROOT`。

`pi05` 版本和 `pi05_full` 版本使用同一套 canonical sidecar 数据入口：

```bash
bash shfile/dsw_training101_pi05_umi_groupA_18.5h.sh
```

它会读取：

```bash
--dataset.roots_file="${ROOT_DIR}/scripts/groupA_roots.txt"
--dataset.sidecar_root="/mnt/cpfs/datasets/pi05_canonical_sidecars_umi"
--policy.type=pi05
```

sidecar 中的 `observation.state` / `action` / mask 会覆盖原始数据中的 state/action，维度仍然是 22D state 和 20D action。`pi05_full` 使用 sidecar 的子任务/文本增强字段；`pi05` 可以忽略这些额外字段，继续走自己的 task prompt + action flow 训练路径。

### 18.4 当前 pi05/pi05_full sidecar 训练脚本注意事项

当前推荐的 groupA canonical sidecar 训练入口包括：

- `shfile/dsw_training101_pi05full_umi_groupA_18.5h.sh`
- `shfile/dsw_training101_pi05_umi_groupA_18.5h.sh`
- `shfile/dlc_torchrun_train_pi05full.sh`

脚本默认行为包括：

- 使用 `/root/miniconda3/envs/lerobot_v0.5.1` 环境中的 `accelerate`
- 使用 `scripts/groupA_roots.txt` 作为多数据集路径列表
- 使用 `/mnt/cpfs/datasets/pi05_canonical_sidecars_umi` 作为默认 sidecar 根目录
- 设置 `--policy.type=pi05_full` 或 `--policy.type=pi05`
- 设置 `--policy.use_state_input=false`
- 设置 `--policy.use_relative_actions=true`
- 设置 `--policy.relative_action_keyframes='["0:3","10:13"]'`
- 设置 `--policy.relative_exclude_joints='[]'`
- 设置 `--policy.canonicalize_bimanual_rot6d=false`
- 设置 `--policy.max_state_dim=22`
- 设置 `--policy.max_action_dim=20`

脚本也做了两点运行稳定性处理：

- 如果误用 `sh xxx.sh` 启动，会自动切回 `bash` 执行。
- 日志会通过 `tee` 完整保存到 `logs/train/{JOB_NAME}.log`，默认非 resume 运行会先把已有同名日志移动成 `.bak`。

DLC 多机脚本不再在脚本内部为 `NPROC_PER_NODE` / `WORLD_SIZE` / `RANK` / `MASTER_ADDR` / `MASTER_PORT` 设置本地 fallback。这些变量必须由 DLC 平台注入；如果缺失，脚本会在启动 `torchrun` 前直接报错退出，避免变量缺失时静默降级成单机或错误拓扑。本地 smoke test 请改用 DSW 单机脚本，或显式 export 这些变量后再运行。

当前不建议只通过命令行追加 `--policy.max_action_dim=32` 来切换 32D，因为脚本内部目前显式写了 `--policy.max_action_dim=20`，且 sidecar 的 action/stats 是 20D canonical 口径。如果未来要使用 32D，更稳妥的做法是单独实现“20D 有效 action + 12D padding + action mask=0 + 推理截回 20D”的完整适配，而不是只改一个参数。

### 18.5 20D action 与 pi05_base 预训练权重加载

`lerobot/pi05_base` 的 action head 是 32D：

```text
model.action_in_proj.weight: [1024, 32]
model.action_out_proj.weight: [32, 1024]
model.action_out_proj.bias: [32]
```

当前 canonical sidecar 使用 20D action，因此加载 `lerobot/pi05_base` 时这 3 个 action projection 参数会因为 shape 不一致被跳过并重新初始化。当前代码已经把这类 mismatch 处理为可接受的 warm-start 行为：

- 兼容 shape 的预训练权重会正常加载
- 仅跳过 `action_in_proj/action_out_proj` 这 3 个参数
- 日志中看到 `Skipping 3 state dict keys with incompatible shapes` 属于预期
- 日志中看到 `Compatible pretrained keys loaded successfully; shape-mismatched keys were reinitialized.` 表示加载状态正常

这和 `/mnt/tacumi_data/tmp/lerobot_vtla_share` 中许多 `MAX_ACTION_DIM=20` 训练脚本的处理思路一致。当前优先建议沿用 20D 路线跑正式训练，先保证 sidecar、relative stats、mask、checkpoint 保存链路稳定。

已完成一次 10 step smoke test：

```text
output_dir=/mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/outputs/test_pi0.5fullsh
log_file=/mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/logs/train/test_pi0.5fullsh.log
```

验证结果：

- `dataset.num_frames=1796744`
- `dataset.num_episodes=2047`
- sidecar relative stats 可动态计算
- 预训练权重除 3 个 action projection 外正常加载
- `train_exit=0`
- checkpoint 保存到 `checkpoints/000010`
