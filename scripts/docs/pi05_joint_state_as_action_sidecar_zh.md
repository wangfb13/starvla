# 真机关节角 State/Action Sidecar 使用说明

这份文档说明如何使用：

```text
scripts/build_pi05_joint_state_as_action_sidecar.py
```

这个脚本面向 train-ready 真机数据，专门生成“关节角 state + 关节角 action”的 sidecar。

核心逻辑是：

- sidecar 里的 `observation.state` 不再使用末端位姿 22D 表示
- sidecar 里的 `observation.state` 改成双臂关节角 + 左右 gripper + 左右臂有效性，共 18D
- sidecar 里的 `action` 也改成同样的 16D
- 因为原始数据 `action` 里没有关节角，所以直接复用原始 `observation.state` 里的关节角和 gripper 作为 action
- 额外写出左右臂有效性、逐维 mask、`prev_action` 等训练辅助字段

可以把它理解成：

```text
raw observation.state (joint + gripper) -> sidecar observation.state
raw observation.state (joint + gripper) -> sidecar action
```

## 1. 适用数据集

这个脚本只接受真机数据，当前会检查：

```text
meta/info.json -> robot_type == spark3_pose_gripper
```

如果不是这个 `robot_type`，脚本会直接报错。

## 2. 关节角来源

脚本会从原始数据集的 `meta/info.json` 里读取 `observation.state` 的维度定义，并按照下面这些字段抽取双臂关节角：

```text
arm_left_1
arm_left_2
arm_left_3
arm_left_4
arm_left_5
arm_left_6
arm_left_7
arm_right_1
arm_right_2
arm_right_3
arm_right_4
arm_right_5
arm_right_6
arm_right_7
gripper_left
gripper_right
```

对于 gripper，脚本也兼容常见别名：

```text
left_gripper / main_gripper_left / left_Position_Rad
right_gripper / main_gripper_right / right_Position_Rad
```

## 3. 输出内容

对每个输入数据集 root，脚本会输出：

- `framewise_guidance.parquet`
- `stats.json`
- `summary.json`
- `build_manifest.json`
  - 写在 `output_root` 根目录下

输出目录结构：

```text
{output_root}/{dataset_root.name}/framewise_guidance.parquet
```

## 4. sidecar 数据格式

### 4.1 observation.state: 18D

顺序是：

```text
main_arm_left_1
main_arm_left_2
main_arm_left_3
main_arm_left_4
main_arm_left_5
main_arm_left_6
main_arm_left_7
main_gripper_left
main_arm_right_1
main_arm_right_2
main_arm_right_3
main_arm_right_4
main_arm_right_5
main_arm_right_6
main_arm_right_7
main_gripper_right
left_arm_present
right_arm_present
```

也就是：

```text
左臂 7 关节 + 左 gripper + 右臂 7 关节 + 右 gripper + 左右臂有效性标志
```

### 4.2 action: 16D

`action` 和上面的 `observation.state` 前 16 维完全一致，也就是：

```text
main_arm_left_1 ... main_arm_left_7 main_gripper_left
main_arm_right_1 ... main_arm_right_7 main_gripper_right
```

区别只是语义上：

- `observation.state` 是当前帧的关节状态输入
- `action` 是当前帧直接复用得到的关节监督目标

因为原始 train-ready 数据的 `action` 里没有关节角，所以这里不会使用原始 `action` 去构造 joint action。

### 4.3 有效性字段

脚本还会额外输出：

- `state_mask`
  - 18D，逐维有效性
- `action_valid_mask`
  - 16D，逐维有效性
- `arm_present_mask`
  - 2D，顺序为 `[left_arm_present, right_arm_present]`
- `state_present`
  - 标量，只要左右臂任意一侧有效就为 1
- `left_arm_present`
- `right_arm_present`
- `prev_action`
- `prev_action_valid_mask`

其中：

- `left_arm_present=1`
  - 表示左臂 7 个关节和左 gripper 这 8 维都有效
- `right_arm_present=1`
  - 表示右臂 7 个关节和右 gripper 这 8 维都有效

注意：

- `state_mask` 的最后 2 维对应 `left_arm_present` / `right_arm_present`
- 这两维本身总是有效，因此这两维的 mask 固定为 1
- 但它们的数值会根据左右臂当前是否完整有效而变成 0 或 1

## 5. 和关节角部署的对应关系

脚本输出的 16 维顺序已经按 Spark3 joint 下发推荐顺序排好。

`summary.json` 里会额外写出：

```json
"recommended_robot_config": {
  "command_mode": "joint",
  "action_joint_names": [
    "main_arm_left_1",
    "...",
    "main_gripper_right"
  ]
}
```

如果你后面要把训练出来的策略直接用于关节角推理，下发时建议把机器人配置成：

```text
robot.command_mode=joint
robot.action_joint_names=[summary.json 里的 action_joint_names]
```

注意：

- sidecar 只会覆盖数据的 shape 和数值
- 原始数据集 `meta/info.json` 里的 action names 不会被自动改成 joint names
- 所以后续部署时，joint action 的顺序请以这个脚本文档和 `summary.json` 为准

## 6. Mask 某个臂

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
  - 左臂 8 维 `state_mask` / `action_valid_mask` 全部置 0
  - `left_arm_present` 置 0
- `right`
  - 右臂 8 维 `state_mask` / `action_valid_mask` 全部置 0
  - `right_arm_present` 置 0

注意这里不会删除向量维度，只是把监督 mask 清零。

## 7. 常用命令示例

### 7.1 处理单个 train-ready 数据集

```bash
python3 scripts/build_pi05_joint_state_as_action_sidecar.py \
  --dataset-root /mnt/tacumi_data/lerobot_datasetV30_train_ready/sparky_ugripper/sparky_ugripper_fruit_packing_v2gripper_R8_vertical_6F_144_v30_tcp_undist_train_ready \
  --output-root /mnt/cpfs/datasets/pi05_joint_state_sidecars \
  --overwrite
```

### 7.2 批量处理多个 train-ready 数据集

```bash
python3 scripts/build_pi05_joint_state_as_action_sidecar.py \
  --dataset-roots-file /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5_starvla/scripts/sparky_fruit_packing_R8_vertical6Fundist_roots.txt \
  --output-root /mnt/cpfs/datasets/R8_vertical6Fundist_joint_state_sidecars \
  --overwrite
```

### 7.3 只训练右臂，mask 左臂

```bash
python3 scripts/build_pi05_joint_state_as_action_sidecar.py \
  --dataset-root /mnt/tacumi_data/lerobot_datasetV30_train_ready/sparky_ugripper/sparky_ugripper_fruit_packing_v2gripper_R8_vertical_6F_144_v30_tcp_undist_train_ready \
  --output-root /mnt/cpfs/datasets/pi05_joint_state_sidecars_right_only \
  --real-mask-arm left \
  --overwrite
```

## 8. 关键参数说明

- `--dataset-root`
  - 单个数据集 root，可重复多次
- `--dataset-roots-file`
  - 文本文件，一行一个数据集 root
- `--output-root`
  - sidecar 输出根目录
- `--schema-name`
  - 写入 sidecar 和 summary 的 schema 名称
  - 默认值是 `joint_state_as_action_bimanual_v1`
- `--missing-value`
  - 缺失值哨兵，默认 `9930.0`
- `--real-mask-arm`
  - 训练时屏蔽某个手臂
- `--base-sidecar`
  - 单数据集场景下，可把已有 sidecar 合并进来
- `--base-sidecar-map`
  - 多数据集场景下，用 JSON 指定每个数据集对应的 base sidecar
- `--overwrite`
  - 覆盖已存在输出

## 9. 推荐检查项

生成后建议至少检查：

- `summary.json` 里的 `state_dim == 18`
- `summary.json` 里的 `action_dim == 16`
- `summary.json` 里的 `action_feature_names` 顺序是否符合你的 joint 下发顺序
- `left_arm_present_ratio` / `right_arm_present_ratio` 是否正常
- `framewise_guidance.parquet` 的 `state_mask` / `action_valid_mask` 是否不是全 0

如果这些都正常，就可以把这个 sidecar 用到 joint action 训练里了。
