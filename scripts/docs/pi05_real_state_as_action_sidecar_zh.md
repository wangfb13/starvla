# 真机 State 复用为 Action 的 Sidecar 使用说明

这份文档说明如何使用：

```text
scripts/build_pi05_real_state_as_action_sidecar.py
```

这个脚本专门处理真机数据，核心逻辑和普通 canonical sidecar 不一样：

- sidecar 里的 `observation.state` 来自原始数据的 `observation.state`
- sidecar 里的 `action` 采用混合来源：末端位姿来自原始 `observation.state`，gripper 来自原始 `action`
- 支持通过参数直接 mask 掉左臂或右臂

也就是说，这个脚本不会直接复用原始数据里的 `action` 全量来构造训练目标，而是走一套：

```text
raw observation.state -> canonical state
raw observation.state pose + raw action gripper -> canonical action
```

适合你想做“用当前状态本身作为监督目标”这类真机实验。

## 1. 输出内容

对每个输入数据集 root，脚本会输出：

- `framewise_guidance.parquet`
- `stats.json`
- `summary.json`
- `stats_relative_action_chunk{N}_gap{G}.json`
  - 只有设置了 `--relative-stats-chunk-size > 0` 才会生成
- `build_manifest.json`
  - 写在 `output_root` 根目录下

输出目录结构仍然和现有 sidecar 保持一致：

```text
{output_root}/{dataset_root.name}/framewise_guidance.parquet
```

## 2. 数据格式

这个脚本输出的 schema 仍然兼容当前仓库的 `pi05` / `pi05_full` canonical 输入约定。

### 2.1 observation.state: 22D

```text
left 10D + right 10D + left_valid + right_valid
```

每只手 10D：

```text
[x, y, z, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]
```

完整下标：

```text
0:2    left xyz
3:8    left rot6d
9      left gripper
10:12  right xyz
13:18  right rot6d
19     right gripper
20     left_valid
21     right_valid
```

### 2.2 action: 20D

```text
left 10D + right 10D
```

这里的 `action` 不是原始数据里的 `action` 全量，而是由原始 `observation.state` 和原始 `action` 混合构造得到。

因此它表达的是：

```text
当前帧 state 对应的 canonical 双臂 pose + 原始 action 里的 gripper
```

### 2.3 其他辅助字段

sidecar 还会写出：

- `action_valid_mask`
- `state_mask`
- `arm_present_mask`
- `state_present`
- `prev_action`
- `prev_action_valid_mask`
- `task_vlm_caption`
- `subtask_text_sidecar`
- `next_subtask_text_sidecar`

这些字段的格式和已有 `build_pi05_canonical_sidecar.py` 基本一致，训练侧可以复用现有读取逻辑。

其中如果原始数据集存在 `meta/subtasks.parquet`，脚本会同步写出：

- `subtask_text_sidecar`
- `next_subtask_text_sidecar`

这里会按 `subtasks.parquet` 里的 `frames=[start, end]` 作为左右全包区间处理，
也就是 `start` 和 `end` 两帧都会被打上对应的 subtask 文本，不再按右开区间处理。

## 3. 适用数据集

这个脚本只接受真机数据。

当前代码里会检查：

```text
meta/info.json -> robot_type == spark3_pose_gripper
```

如果不是这个 `robot_type`，脚本会直接报错退出。

## 4. 真机数据处理逻辑

和普通 real sidecar 一样，这个脚本会：

- 保留真机原始坐标系
- 不做 UMI 那种每个 episode anchor 坐标系转换
- 把四元数转成 `rot6d`
- 拼成统一的双臂 canonical 表示

区别在于：

- `state` 用原始 `observation.state`
- `action` 的末端位姿用原始 `observation.state`
- `action` 的 gripper 用原始 `action`

可以把它理解成：

```text
sidecar_action = concat(state_pose_rot6d, raw_action_gripper)
```

只是实际实现时会分别根据左右臂 pose/gripper 的有效性去构造 mask。

## 5. Mask 某个臂

脚本支持：

```bash
--real-mask-arm none
--real-mask-arm left
--real-mask-arm right
```

含义是：

- `none`
  - 不屏蔽任何手臂
- `left`
  - 左臂的 `state_mask` 和 `action_valid_mask` 全部置 0
  - `left_present` 也会被置成 0
- `right`
  - 右臂的 `state_mask` 和 `action_valid_mask` 全部置 0
  - `right_present` 也会被置成 0

注意这里的 mask 是训练监督层面的 mask，不是把对应维度从向量结构里删掉。

也就是说：

- state 仍然是 22D
- action 仍然是 20D
- 只是被 mask 的那只手不会参与 loss

## 6. 常用命令示例

### 6.1 处理单个真机数据集

```bash
python3 scripts/build_pi05_real_state_as_action_sidecar.py \
  --dataset-root /mnt/sparky_data/your_real_dataset \
  --output-root /mnt/cpfs/repo/pi05_real_state_as_action_sidecars \
  --overwrite
```

### 6.2 处理单个真机数据集并 mask 掉右臂

```bash
python3 scripts/build_pi05_real_state_as_action_sidecar.py \
  --dataset-root /mnt/sparky_data/your_real_dataset \
  --output-root /mnt/cpfs/repo/pi05_real_state_as_action_sidecars \
  --real-mask-arm right \
  --overwrite
```

### 6.3 用 txt 批量处理多个真机数据集

```bash
python3 scripts/build_pi05_real_state_as_action_state_gripper_sidecar.py \
  --dataset-roots-file /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/sparky_fruit_packing_R7_norotation_roots.txt \
  --output-root /mnt/cpfs/datasets/pi05_canonical_sidecars_fruit_packing_sparky_stateAsAction_R7norotation60 \
  --real-mask-arm none \
  --overwrite


python3 scripts/build_pi05_real_state_as_action_sidecar.py \
  --dataset-roots-file /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/sparky_pegSorting_roots.txt \
  --output-root /mnt/cpfs/datasets/pi05_canonical_sidecars_pegSorting_sparky_onlyRightArm_stateAsAction \
  --real-mask-arm left \
  --overwrite


python3 scripts/build_pi05_real_state_as_action_sidecar.py \
  --dataset-roots-file /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/sparky_fruit_packing_roots.txt \
  --output-root /mnt/cpfs/datasets/pi05_canonical_sidecars_fruit_packing_sparky_stateAsAction_withSubtasks \
  --real-mask-arm none \
  --overwrite


python3 scripts/build_pi05_real_state_as_action_state_gripper_sidecar.py \
  --dataset-roots-file /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/sparky_fruit_packing_R7_undist_roots.txt \
  --output-root /mnt/cpfs/datasets/pi05_canonical_sidecars_fruit_packing_sparky_stateAsAction_R7_stateGripperUndist \
  --real-mask-arm none \
  --overwrite
```

### 6.4 同时生成 relative action stats

```bash
python3 scripts/build_pi05_real_state_as_action_sidecar.py \
  --dataset-root /mnt/sparky_data/your_real_dataset \
  --output-root /mnt/cpfs/repo/pi05_real_state_as_action_sidecars \
  --relative-stats-chunk-size 50 \
  --relative-stats-action-gap-frames 0 \
  --overwrite
```

## 7. 关键参数说明

- `--dataset-root`
  - 单个数据集 root，可重复写多次
- `--dataset-roots-file`
  - txt 文件，每行一个数据集 root
- `--output-root`
  - sidecar 输出根目录
- `--real-mask-arm`
  - 是否屏蔽左臂或右臂
- `--missing-value`
  - 原始 state 里的缺失值哨兵，默认是 `9930.0`
- `--relative-stats-chunk-size`
  - 如果大于 0，会额外输出 relative action stats
- `--relative-stats-action-gap-frames`
  - 生成 relative stats 时使用的 future gap
- `--base-sidecar`
  - 单数据集模式下可选，用来和已有 sidecar 合并
- `--base-sidecar-map`
  - 多数据集模式下可选，给每个数据集指定已有 sidecar
- `--overwrite`
  - 覆盖已有输出

## 8. 训练时怎么用

如果你用这个脚本生成了 sidecar，那么训练时建议和普通 canonical sidecar 一样：

```bash
--policy.canonicalize_bimanual_rot6d=false
--policy.max_state_dim=22
--policy.max_action_dim=20
```

因为 sidecar 里已经是 canonical 22D/20D 了，不需要在训练时再做一次在线 canonicalization。

如果按 sidecar root 自动查找，训练参数一般是：

```bash
--dataset.roots_file=/path/to/roots.txt \
--dataset.sidecar_root=/path/to/pi05_real_state_as_action_sidecars \
--policy.max_state_dim=22 \
--policy.max_action_dim=20 \
--policy.canonicalize_bimanual_rot6d=false
```

## 9. 和普通 real sidecar 的区别

普通的：

```text
sidecar state <- raw state
sidecar action <- raw action
```

这个脚本的：

```text
sidecar state <- raw state
sidecar action <- raw state pose + raw action gripper
```

所以它更像是在做：

- state pose reconstruction 风格监督
- current-state-pose-as-target + raw-gripper 风格监督

而不是直接学习原始控制 action 的完整 20D 向量。

## 10. 结果检查建议

跑完后建议优先检查：

- `summary.json`
  - 看 `real_mask_arm` 是否符合预期
  - 看 `state_source` 是否是 `raw_observation_state`
  - 看 `action_source` 是否是 `raw_observation_state_pose_plus_raw_action_gripper`
- `framewise_guidance.parquet`
  - 抽几帧确认 `action` 的 pose 是否来自同一帧 `state`，gripper 是否来自同一帧原始 `action`
- `action_valid_mask`
  - 确认被 mask 的那一侧是否已经变成全 0

如果后面你想把这套 sidecar 接进现有 pegSorting 训练脚本，建议再额外核对一次：

- `dataset.sidecar_root`
- `policy.max_state_dim`
- `policy.max_action_dim`
- `policy.canonicalize_bimanual_rot6d`
