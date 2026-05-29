# PI05 / PI05-Full Sidecar 训练使用说明

这份文档说明当前工程里如何使用 canonical sidecar 数据训练 `pi05` 和 `pi05_full`。建议先理解统一输入格式和 relative action 口径，再按后面的步骤生成 sidecar 和启动训练。

## 1. 当前训练数据流

使用 sidecar 训练时，原始 LeRobot 数据目录保持不变。训练时仍从原始目录读取图像、视频、episode 信息和任务文本，但 `observation.state` / `action` / mask 等训练向量会被 sidecar 覆盖。

sidecar 的默认查找方式是：

```text
{dataset.sidecar_root}/{dataset_root.name}/framewise_guidance.parquet
```

因此 `roots_file` 里每一行是一个原始 LeRobot dataset root，而 sidecar root 下需要有同名子目录。

## 2. 统一输入数据格式

当前 canonical sidecar 给 `pi05` / `pi05_full` 提供统一的双臂 22D state 和 20D action。

### 2.1 Action: 20D

```text
action = left 10D + right 10D
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
```

### 2.2 Observation State: 22D

```text
observation.state = action-like 20D + left_valid + right_valid
```

前 20 维和 action 对齐：

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

### 2.3 Mask 和文本字段

sidecar 还会提供这些常用字段：

- `action_valid_mask`: 20D，控制 action loss 哪些维度有效，单臂数据会屏蔽缺失手臂。
- `state_mask`: 20D，表示 state 前 20 维 pose/gripper 是否有效。
- `arm_present_mask`: 2D，表示左右手是否存在。
- `state_present`: 标量，表示当前帧 state 是否整体可用。
- `prev_action` / `prev_action_valid_mask`: 给需要 previous action conditioning 的实验使用。
- `task_vlm_caption` / `subtask_text_sidecar` / `next_subtask_text_sidecar` / `user_prompt_sidecar`: 给 `pi05_full` 的文本增强或阶段训练使用；`pi05` 可忽略额外字段。

## 3. Rot6D 表示

旋转统一使用 `rot6d`，不是欧拉角或四元数。`rot6d` 是旋转矩阵前两列拼成的 6 个数：

```text
rot6d = [R[:, 0], R[:, 1]]
```

模型训练时回归连续 6D 向量；需要做几何运算时临时转成旋转矩阵，算完再转回 `rot6d`。这样避免欧拉角跳变和万向节锁，也避免四元数 `q` / `-q` 双表示的问题。

## 4. 开启 Relative Action 时的训练处理

当前推荐训练脚本默认开启：

```bash
--policy.use_relative_actions=true
--policy.relative_action_keyframes='["0:3","10:13"]'
--policy.relative_exclude_joints='[]'
--policy.action_gap_frames=6
```

这表示所有 action 维度都会参与 chunk relative，但不同维度的处理方式不同。

### 4.1 xyz + rot6d 使用 SE(3) relative

`relative_action_keyframes='["0:3","10:13"]'` 的含义是：

```text
left:  xyz 从 0 开始，rot6d 从 3 开始
right: xyz 从 10 开始，rot6d 从 13 开始
```

对每个 chunk 目标 action，训练时使用当前 sample 的 `observation.state` 作为参考位姿：

```text
rel_pos = R_state^T @ (target_pos - state_pos)
rel_rot = R_state^T @ R_target
rel_rot6d = matrix_to_rot6d(rel_rot)
```

所以 `rot6d` 不会做逐维相减，`xyz` 也不是简单世界系差值，而是当前 TCP/参考姿态局部坐标系下的位移。

### 4.2 Gripper 默认也做 chunk relative

当前脚本默认：

```bash
--policy.relative_exclude_joints='[]'
```

因此 `gripper` 不再保持 absolute，而是作为连续量做：

```text
rel_gripper = target_gripper - state_gripper
```

如果要做夹爪 absolute 的对照实验，可以启动时覆盖：

```bash
RELATIVE_EXCLUDE_JOINTS='["gripper"]' bash shfile/dsw_training101_pi05full_umi_groupA_18.5h.sh
```

### 4.3 Chunk 和 Episode 末尾

训练会按 `chunk_size` 取未来 action chunk。目标帧下标大致是：

```text
current_index + action_gap_frames + chunk_offset
```

如果到了 episode 末尾不够一个 chunk，会 clamp 到该 episode 的最后一帧，不会跨 episode 取 action。

### 4.4 Relative Stats

开启 `use_relative_actions=true` 后，action normalization 必须使用 relative action 口径的 stats，不能直接用 framewise absolute action 的 `stats.json`。

当前 keyframes 非空时，训练加载器会基于 `framewise_guidance.parquet` 动态计算 relative action stats，确保 `xyz + rot6d` 的 SE(3) relative 和 gripper relative 口径一致。`observation.state` stats 使用 sidecar 的 canonical state stats。

## 5. UMI 数据和真机数据的区别

sidecar 生成脚本会读取每个数据集的 `meta/info.json` 里的 `robot_type` 自动判断类型。

### 5.1 UMI 数据

以下 `robot_type` 会按 UMI 数据处理：

```text
ugripper_left
ugripper_right
dual_ugripper
```

UMI 原始 pose 是 TCP 在 SLAM/world 坐标系下的位姿。sidecar 预处理会对每个 episode 使用第一帧有效 state pose 作为 anchor，把 state 和 action pose 转成相对该 anchor 的位姿：

```text
T_anchor_relative = inv(T_first_valid_state) @ T_t
```

展开为：

```text
p_rel = R_anchor^T @ (p_t - p_anchor)
R_rel = R_anchor^T @ R_t
```

然后把相对旋转转成 `rot6d` 保存。

UMI 的原始 state/action gripper 会先裁剪到 `[-0.436, 0.0]`，再统一除以 `-0.436` 归一化到 `[0, 1]`：

```text
gripper = clip(raw_gripper, -0.436, 0.0) / -0.436
```

因此 `0 -> 0`，`-0.218 -> 0.5`，`-0.436` 以及更小值都会变成 `1`。归一化后的 gripper 在 sidecar 中保存为 framewise 连续值，训练时再按 chunk relative 转换。

UMI 训练默认使用全局：
这个可以根据自身系统的实际情况进行修改，粗略算的话就是系统模型推理时间按照30帧算得到的帧数

```bash
--policy.action_gap_frames=6
```

### 5.2 真机数据

以下 `robot_type` 会按真机数据处理：

```text
spark3_pose_gripper
```

真机数据默认认为记录时的 `state/action` 已经考虑了控制延时，因此 sidecar 预处理不再做 episode anchor 相对化，pose 保持原始数据坐标系，只做：

```text
quat -> rot6d
pack into 22D/20D canonical schema
```

训练和动态 stats 里，真机数据会自动使用 `gap=0`。如果你显式设置了 dataset-level gap override，则显式设置优先。

## 6. 生成 Sidecar 数据

### 6.1 准备 roots txt

创建一个 txt，每行一个原始 LeRobot dataset root。可以有空行和 `#` 注释。

示例：

```text
/mnt/cpfs/datasets/pi05_groupA_train_roots_symlink/000__dag911262r00025a_lerobot_0410_233405_train_ready
/mnt/cpfs/datasets/pi05_groupA_train_roots_symlink/001__dag911262r001146_lerobot_0410_233405_train_ready
/mnt/sparky_data/ugripper_peg_in_hole_v1gripper_3/sparky_ugripper_peg_in_hole_200_v30_unified
```

注意：这些路径的最后一级目录名需要唯一，因为 sidecar 会按 `{dataset_root.name}` 建子目录。

### 6.2 执行生成脚本

推荐使用项目内脚本：

```bash
cd /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5

/root/miniconda3/envs/lerobot_v0.5.1/bin/python \
  scripts/build_pi05_canonical_sidecar.py \
  --dataset-roots-file /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/groupA_roots.txt \
  --output-root /mnt/cpfs/datasets/pi05_canonical_sidecars_umi \
  --dataset-type auto \
  --relative-action-keyframes 0:3,10:13 \
  --relative-exclude-joints "" \
  --overwrite
```

如果只处理单个数据集，可以用：

```bash
/root/miniconda3/envs/lerobot_v0.5.1/bin/python \
  scripts/build_pi05_canonical_sidecar.py \
  --dataset-root /path/to/one_lerobot_dataset \
  --output-root /mnt/cpfs/datasets/pi05_canonical_sidecars_debug \
  --dataset-type auto \
  --overwrite
```

### 6.3 输出文件

每个输入 dataset 会生成一个同名 sidecar 子目录：

```text
/mnt/cpfs/datasets/pi05_canonical_sidecars_umi/{dataset_root.name}/
```

目录里主要有：

```text
framewise_guidance.parquet
stats.json
summary.json
```

如果执行时设置：

```bash
--relative-stats-chunk-size 50
--relative-stats-action-gap-frames 6
```

还会额外写：

```text
stats_relative_action_chunk50_gap6.json
```

不过当前训练脚本使用 keyframes 时会动态计算 relative stats，以避免误用旧口径 stats。这个可选文件更多用于检查、记录或后续扩展。

### 6.4 快速检查

生成后先看顶层 manifest：

```bash
ls /mnt/cpfs/datasets/pi05_canonical_sidecars_umi
cat /mnt/cpfs/datasets/pi05_canonical_sidecars_umi/build_manifest.json | head
```

检查某个数据集 summary：

```bash
cat /mnt/cpfs/datasets/pi05_canonical_sidecars_umi/{dataset_root.name}/summary.json | head -80
```

检查 parquet 行数和字段：

```bash
/root/miniconda3/envs/lerobot_v0.5.1/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

p = Path("/mnt/cpfs/datasets/pi05_canonical_sidecars_umi/{dataset_root.name}/framewise_guidance.parquet")
df = pd.read_parquet(p)
print(df.columns.tolist())
print("rows:", len(df))
print("state dim:", len(df.iloc[0]["observation.state"]))
print("action dim:", len(df.iloc[0]["action"]))
PY
```

期望：

```text
state dim = 22
action dim = 20
```

## 7. 使用 Sidecar 训练 PI05-Full

推荐入口：

```bash
cd /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5
bash shfile/dsw_training101_pi05full_umi_groupA_18.5h.sh
```

脚本默认读取：

```bash
DATASET_ROOTS_FILE="${ROOT_DIR}/scripts/groupA_roots.txt"
CANONICAL_SIDECAR_ROOT="/mnt/cpfs/datasets/pi05_canonical_sidecars_umi"
INFO_PATH="${ROOT_DIR}/shfile/info_groupA_fisheye_only.json"
```

核心训练参数是：

```bash
--dataset.roots_file="${DATASET_ROOTS_FILE}"
--dataset.sidecar_root="${CANONICAL_SIDECAR_ROOT}"
--dataset.info_path="${INFO_PATH}"
--policy.type=pi05_full
--policy.canonicalize_bimanual_rot6d=false
--policy.max_state_dim=22
--policy.max_action_dim=20
--policy.use_relative_actions=true
--policy.relative_action_keyframes='["0:3","10:13"]'
--policy.relative_exclude_joints='[]'
--policy.action_gap_frames=6
--policy.use_state_input=false
```

如果要换 roots 或 sidecar 目录：

```bash
DATASET_ROOTS_FILE=/path/to/roots.txt \
CANONICAL_SIDECAR_ROOT=/path/to/sidecar_root \
bash shfile/dsw_training101_pi05full_umi_groupA_18.5h.sh
```

## 8. 使用 Sidecar 训练 PI05

推荐入口：

```bash
cd /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5
bash shfile/dsw_training101_pi05_umi_groupA_18.5h.sh
```

`pi05` 和 `pi05_full` 使用同一套 sidecar 数据。区别主要是：

```text
pi05      : policy.type=pi05
pi05_full : policy.type=pi05_full
```

`pi05_full` 会更多使用 sidecar 中的文本增强/阶段字段；`pi05` 可以忽略这些额外字段。

## 9. 多机多卡 PI05-Full

多机多卡脚本：

```bash
shfile/dlc_torchrun_train_pi05full.sh
```

在 DLC 多机任务里，平台通常会自动注入 `NPROC_PER_NODE` / `WORLD_SIZE` / `RANK` / `MASTER_ADDR` / `MASTER_PORT`，脚本会直接读取这些变量，不需要手动配置。当前 DLC 脚本不再提供本地 fallback；如果这些变量缺失，会在启动 `torchrun` 前直接报错退出，避免误降级成单机或错误拓扑。
sidecar 相关参数仍然是：

```bash
DATASET_ROOTS_FILE=/path/to/roots.txt
CANONICAL_SIDECAR_ROOT=/path/to/sidecar_root
```

## 10. 常见问题

### 10.1 还需要 `INFO_PATH` 吗？

需要。sidecar 会覆盖 state/action，但 `INFO_PATH` 仍用于指定训练使用的相机模态、图像特征和 dataset feature 信息。
目前是我自己产生一个训练需要的key值，主要为了区分是否有触觉加入训练


```bash
INFO_PATH="${INFO_PATH:-${ROOT_DIR}/shfile/info_groupA_fisheye_only.json}"
```
info_groupA_fisheye_only.json内容如下

```json
{
  "features": {
    "observation.images.cam_left": {
      "dtype": "video",
      "shape": [
        3,
        224,
        224
      ],
      "names": [
        "channels",
        "height",
        "width"
      ],
      "intrinsics": {
        "model": "none",
        "params": []
      }
    },
    "observation.images.cam_right": {
      "dtype": "video",
      "shape": [
        3,
        224,
        224
      ],
      "names": [
        "channels",
        "height",
        "width"
      ],
      "intrinsics": {
        "model": "none",
        "params": []
      }
    }
  }
}

```


### 10.2 为什么 sidecar 训练要关闭 `canonicalize_bimanual_rot6d`？

因为 sidecar 已经提前输出 22D/20D canonical state/action。如果再开在线 canonicalize，会重复转换。

正确设置：

```bash
--policy.canonicalize_bimanual_rot6d=false
```

### 10.3 为什么 action 是 20D，不是 pi05_base 的 32D？

当前 sidecar schema 是 20D 双臂 canonical action。加载 `lerobot/pi05_base` 时，32D action head 的少量 projection 参数会因为维度不同被跳过，这是预期现象；其余兼容权重会正常加载。

不要只通过 `--policy.max_action_dim=32` 硬改。若未来要 32D，需要实现 20D 有效 action + padding + mask 的完整适配。

### 10.4 训练输出目录已存在怎么办？

如果不是继续训练，换一个 `JOB_NAME` 或 `OUT_DIR`。如果要 resume：

```bash
RESUME_IF_POSSIBLE=1 bash shfile/dsw_training101_pi05full_umi_groupA_18.5h.sh
```

### 10.5 找不到 sidecar 怎么办？

检查：

```text
roots_file 里的 dataset root 最后一级目录名
sidecar_root 下的子目录名
```

两者必须一致。

### 10.6 想让 gripper 保持 absolute 怎么办？

默认 gripper 做 chunk relative。如果要保持 absolute：

```bash
RELATIVE_EXCLUDE_JOINTS='["gripper"]' bash shfile/dsw_training101_pi05full_umi_groupA_18.5h.sh
```

### 10.7 真机数据为什么 gap=0？

真机数据 `robot_type=spark3_pose_gripper` 默认认为记录的 `state/action` 已经考虑延时，所以动态 stats 和采样使用 `gap=0`。UMI 数据使用全局 `ACTION_GAP_FRAMES`，默认是 6。
