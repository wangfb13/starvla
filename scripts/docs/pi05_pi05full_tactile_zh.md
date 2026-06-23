# PI05 / PI05-Full Tactile 分支说明

这份文档简要说明当前仓库里 `pi05_full` 新增的 tactile 支持，包括：

- tactile 数据是怎么读进来的
- 当前支持哪两种 tactile 编码方式
- 训练时该怎么配
- 什么时候该选 `flat`，什么时候该选 `parsed_vit`

## 1. 当前结论

当前仓库里，tactile 相关新增已经接入 `pi05_full`。

现在可以通过：

- `policy.enable_tactile=true`
- `policy.tactile_feature_keys=[...]`
- `policy.tactile_encoder_type=flat|parsed_vit`

来决定是否启用 tactile，以及用哪种 tactile 编码器。

注意：

- 当前改动是加在 `pi05_full` 上的
- `pi05` 目前没有同步加这条 tactile prefix 分支

## 2. tactile 数据从哪里来

在这个仓库里，tactile 数据不是普通标量 state，也不是直接从 parquet 里拿一列向量给模型。

它的主路径是：

1. 数据集把 tactile 特征记成视觉类特征
2. `dtype` 通常是 `tactile` 或 `depth`
3. 读取时按视频帧解码
4. 对 Sparky 这类 `depth/deformation/shear` tactile，再做专门的归一化

也就是说，当前 tactile 输入本质上是“逐帧解码出来的 tactile 图像/图谱”。

对 Sparky 常见 key，通常会看到类似：

```text
observation.depth_deformation.tactile_left_left
observation.depth_deformation.tactile_left_right
observation.depth_deformation.tactile_right_left
observation.depth_deformation.tactile_right_right
observation.depth_shear.tactile_left_left
observation.depth_shear.tactile_left_right
observation.depth_shear.tactile_right_left
observation.depth_shear.tactile_right_right
```

补充说明：

- `observation.depth_deformation.tactile_*` 这 4 个名字是当前主推荐的 canonical tactile key
- 对 `observation.depth_deformation.tactile_*` / `observation.depth_shear.tactile_*` 这类 3 通道 `.mov`
  - 第 1 通道按 depth 处理，使用 `0..3000 -> 0..1`
  - 第 2/3 通道按 deformation/shear 处理，使用 `25000..35000 -> 0..1`
  - 归一化在视频解码阶段完成，进入模型前不再依赖 `stats.json` 的 tactile 统计
- `multi_dataset.py` 仍兼容一部分历史单臂命名
- 如果老数据里只有：
  - `observation.depth_deformation.tactile_left_left`
  - `observation.depth_deformation.tactile_right_left`
  - 那么代码会结合相机 key 自动判断它应该解释成左臂一对，还是右臂一对
- 相机侧别兼容也仍然保留：
  - `observation.images.left_cam_left` -> `observation.images.cam_left`
  - `observation.images.right_cam_right` -> `observation.images.cam_right`

## 3. 当前支持的两种 tactile 编码器

### 3.1 `flat`

`flat` 是轻量分支。

大致逻辑是：

1. 读取 tactile tensor
2. 如果是图像，就先做池化降采样
3. 展平成向量
4. 再切成固定数量的 tactile tokens
5. 投影到 `pi05_full` prefix hidden size

适合场景：

- 先快速验证 tactile 是否有帮助
- tactile 输入格式还不稳定
- 后面可能会接别的触觉编码器
- 希望先用更省显存、更省算力的方案

它的优点是简单、稳、容易扩展。

### 3.2 `parsed_vit`

`parsed_vit` 是从 `lerobot_vtla_share` 那边移植过来的主分支。

大致逻辑是：

1. 把 tactile 图 resize 到固定大小
2. 按 patch 切分
3. 用小型 transformer block 编码 patch sequence
4. 用 learned query resampler 压成固定数量 tactile tokens
5. 再投影到 `pi05_full` prefix hidden size

它更适合：

- tactile 已经是比较规范的多通道图
- 你想保留更多局部空间结构
- 希望更接近 `vtla_share` 的 tactile 建模方式

## 4. 这两个名字是不是原仓库就有

不是完全原样。

`/mnt/base/user/zfc/lerobot_vtla_share` 里原本是：

- `enable_tactile`
- `enable_parsed_tactile_vit`

对应关系是：

- `enable_tactile=true` 且 `enable_parsed_tactile_vit=false`
  - 对应现在这里的 `tactile_encoder_type=flat`
- `enable_tactile=true` 且 `enable_parsed_tactile_vit=true`
  - 对应现在这里的 `tactile_encoder_type=parsed_vit`

也就是说：

- `flat` 和 `parsed_vit` 这两个名字，是这次为了以后支持多种 tactile encoder，新整理出来的统一开关
- 但它们背后的两条实现路径，原项目里本来就有原型

## 5. 训练时怎么配

### 5.1 轻量模式

如果想先快速试 tactile：

```text
policy.enable_tactile=true
policy.tactile_encoder_type=flat
policy.tactile_feature_keys=["observation.depth_deformation.tactile_left_left","observation.depth_deformation.tactile_left_right","observation.depth_deformation.tactile_right_left","observation.depth_deformation.tactile_right_right"]
```

### 5.2 VTLA 风格 parsed tactile 模式

如果想走更接近 `vtla_share` 的方式：

```text
policy.enable_tactile=true
policy.tactile_encoder_type=parsed_vit
policy.tactile_feature_keys=["observation.depth_deformation.tactile_left_left","observation.depth_deformation.tactile_left_right","observation.depth_deformation.tactile_right_left","observation.depth_deformation.tactile_right_right"]
policy.tactile_input_channels=3
policy.tactile_target_size=[96,128]
policy.tactile_patch_size=16
policy.tactile_patch_encoder_dim=384
policy.tactile_patch_encoder_depth=12
policy.tactile_resampler_heads=6
policy.tactile_token_budget_per_sensor=8
```

## 6. 该怎么选

可以直接按这个经验来：

- 想先跑通，先看 tactile 有没有收益：选 `flat`
- tactile 图结构比较重要，想更接近 `vtla_share`：选 `parsed_vit`
- 数据格式可能还会变，后面还想接别的 tactile backbone：先用 `flat`
- 以后要继续扩展新的 tactile 编码器：沿着 `tactile_encoder_type` 继续加

## 7. 当前这版实现的边界

这次移植的目标是：

- 保留你当前仓库 `pi05_full` 已有的训练主逻辑
- 把 tactile 作为可选 prefix 分支接进去
- 不打乱你现有的 sidecar / subtask / relative action 路径

所以当前更准确地说是：

- 已完成 tactile 分支接入
- 已做成可选编码器结构
- 先支持 `flat` 和 `parsed_vit`

但还没有继续扩展成更多编码器，比如：

- `cnn`
- `mae`
- `precomputed_tokens`

后面如果要加新的 tactile 编码器，主要沿着：

```text
policy.tactile_encoder_type=...
```

继续扩展即可，不需要重写整条 `pi05_full` 主干。
