# PI05 / PI05-Full 模型输出分析说明

这份文档单独说明当前工程里“针对模型输出本身”的分析能力。

这里的“输出分析”不是训练 loss 曲线，而是更接近下面这些问题：

- 模型到底预测了什么？
- 预测文本和标注差多少？
- rollout 出来的动作最终任务是否成功？
- 某个输入模态或局部区域对当前输出有没有贡献？

## 1. 当前有哪些输出分析能力

当前仓库里和模型输出直接相关的能力主要分四类：

- `pi05_full` 的 next-subtask 文本输出验证
- `pi05` / `pi05_full` 的离线 action 输出对比
- rollout / env 成功率评估
- 基于 share 项目经验的可解释性分析方向

其中前两类已经在当前仓库里有现成脚本；第三类主要依赖 `lerobot-eval` 主流程；第四类目前是方法参考，还没有完整移植成你这个仓库里的独立工具。

## 2. 最直接的输出分析：next-subtask 预测

脚本：

```text
scripts/run_pi05full_subtask_validation.py
```

这个脚本的重点不是看 loss，而是直接加载 `pi05_full` checkpoint，调用模型的：

```text
generate_subtask_tokens(...)
```

然后把模型生成的 `predicted_next_subtask` 和 sidecar 里的标注：

```text
next_subtask_text_sidecar
```

做逐样本对比。

### 2.1 它会输出什么

输出目录里通常会有：

- `summary.json`
- `samples.json`
- `samples.csv`
- `report.md`
- `report.html`
- `images/`

核心指标包括：

- `exact_match_rate`
- `mean_similarity`

同时每个样本还会保留：

- 当前帧的 RGB 图
- 当前子任务文本
- 标注的下一个子任务文本
- 模型预测的下一个子任务文本

所以这个工具适合回答：

- 这个 checkpoint 是否真的学会了 next-subtask 预测？
- 错误主要是完全错，还是语义接近但措辞不同？
- 哪类任务或哪类 robot type 更容易预测错？

### 2.2 适用范围

这个脚本：

- 适用于 `pi05_full`
- 不适用于当前仓库里的 `pi05`

原因很直接：当前 `pi05` 没有和 `pi05_full` 对应的 `generate_subtask_tokens` 路径。

## 3. 新增的 action 输出对比

脚本：

```text
scripts/report_pi05_action_predictions.py
```

这个脚本支持：

- `pi05`
- `pi05_full`

它会离线读取 dataset 样本，加载 checkpoint，然后直接跑：

```text
predict_action_chunk(...)
```

再把模型预测的 action chunk 和数据里的 ground-truth action chunk 做对比。

### 3.1 它会输出什么

输出目录里会有：

- `summary.json`
- `samples.csv`
- `sample_step_metrics.csv`
- `summary_plots.png`
- `images/`
- `action_curves/`
- `report.md`

核心信息包括：

- 全样本平均 `MAE`
- 全样本平均 `RMSE`
- 每个 action 维度的平均误差
- 每个 chunk step 的平均误差
- 每个样本的预测轨迹图

它适合回答：

- 模型动作输出和标注动作差多大？
- 误差主要集中在哪些维度？
- 误差是越往后越大，还是整个 chunk 都差不多？
- 某些任务样本是不是明显更难预测？

### 3.2 一个最小例子

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_v0.5.1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/report_pi05_action_predictions.py \
  --checkpoint-dir /path/to/checkpoint_dir \
  --dataset-root /path/to/dataset_root \
  --sidecar-path /path/to/framewise_guidance.parquet \
  --output-dir /path/to/action_eval \
  --num-samples 16 \
  --device cuda
```

如果数据集本身不依赖 sidecar，也可以不传 `--sidecar-path`。

## 4. 行为结果分析：rollout / eval

如果你关心的不是“文本预测对不对”，而是“模型动作输出后任务做没做成”，那应该看 rollout 级评估。

这类分析主要依赖：

```text
src/lerobot/scripts/lerobot_eval.py
```

它对应的问题是：

- policy 输出 action 后，环境 reward 怎么样？
- success rate 高不高？
- episode 多久结束？
- 不同 checkpoint 谁更稳？

这个方向更偏最终任务表现，而不是中间 token / subtask 输出。

## 5. 当前仓库里还没有完整落地的部分

参考 `/mnt/tacumi_data/tmp/lerobot_vtla_share`，那边还涉及一类更偏“解释为什么模型这样输出”的分析：

- whole-modality masking
- patch occlusion
- deletion curve
- deletion AUC
- tactile / RGB 局部贡献比较

这些内容在 share 项目里主要体现在：

- `notebook/pi05_foundationB_interpretability_rookie_report.executed.ipynb`

它的目标不是直接给出任务成功率，而是分析：

- tactile 是否真的帮助了当前预测
- 哪个 RGB / tactile 局部区域对当前输出更关键
- 全模态移除和局部遮挡对 loss / 输出的影响

这部分目前还没有被完整整理成当前仓库里的独立脚本。

也就是说，当前仓库现状是：

- “训练有没有收敛”这类分析：已经有
- “`pi05_full` 具体预测了什么 next-subtask”这类分析：已经有
- “action rollout 最终成功率如何”这类分析：走 `lerobot-eval`
- “输入局部区域如何影响输出”这类解释性分析：方法有参考，但还没正式移植

## 6. 和训练日志分析的区别

现有这些脚本里，容易混淆的是：

- `scripts/report_pi05_cotrain.py`
- `scripts/report_pi05_teaching_run.py`

它们主要分析的是训练过程：

- loss
- per-source loss
- sampling ratio
- action / subtask / tactile 相关曲线

它们能帮助我们判断训练是否健康，但不能直接回答：

- 模型具体生成了什么文本？
- 这个样本为什么预测错了？
- 模型输出动作后任务到底做没做成？

所以如果你的目标是“看模型输出本身”，优先级通常应该是：

1. `pi05_full`：
   先跑 `run_pi05full_subtask_validation.py`
2. `pi05` 或 `pi05_full`：
   再跑 `report_pi05_action_predictions.py`
3. 需要最终任务表现：
   再跑 `lerobot-eval`
4. 需要解释输入贡献：
   再考虑补移植 interpretability 工具

## 7. 现阶段建议怎么用

如果是 `pi05_full` 训练完一个 checkpoint，建议顺序是：

1. 先用 `report_pi05_cotrain.py` 或 `report_pi05_teaching_run.py` 看训练是不是正常
2. 再用 `run_pi05full_subtask_validation.py` 看 next-subtask 输出是否真的提升
3. 再用 `report_pi05_action_predictions.py` 看 action chunk 和 GT 的偏差
4. 如果后面接真实环境或仿真环境，再用 `lerobot-eval` 看最终任务效果

如果是 `pi05`，当前更现实的输出分析方式是：

1. 看训练曲线
2. 跑 `report_pi05_action_predictions.py`
3. 跑 rollout / eval

## 8. 后续可以继续补什么

如果你后面想把“模型输出分析”再做完整，下一步最值得补的是两类工具：

- 可解释性工具
  - tactile / RGB masking
  - patch occlusion
  - 删除曲线和排序稳定性
- 更细粒度的 action 分析
  - 末端位置 / 旋转 / gripper 分项误差
  - 左臂 / 右臂分组误差
  - 相对动作和绝对动作两种口径对比

这样就能把“训练是否收敛”“输出是什么”“为什么这样输出”“最终任务效果如何”四层分析接起来。
