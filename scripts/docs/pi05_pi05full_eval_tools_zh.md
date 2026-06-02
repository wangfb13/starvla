# PI05 / PI05-Full 训练评估工具说明

这份文档说明已经迁移到当前仓库 `scripts/` 下的训练评估工具，以及它们分别适合什么场景。

目标是让我们在 `pi05` 和 `pi05_full` 训练完成后，能比较系统地回答这几个问题：

- 训练是不是正常收敛了？
- 不同数据源是否都在学？
- tactile / conditioning 分支有没有工作？
- `pi05_full` 的 next-subtask 预测有没有提升？

## 1. 工具总览

当前新增的脚本有：

- `scripts/report_pi05_cotrain.py`
- `scripts/report_pi05_teaching_run.py`
- `scripts/build_pi05full_subtask_manifest.py`
- `scripts/run_pi05full_subtask_validation.py`

相关补充文档：

- `scripts/docs/pi05_pi05full_output_analysis_zh.md`
  - 单独说明“模型输出本身”的分析能力
  - 区分训练曲线分析、next-subtask 预测分析、rollout 结果分析

适用范围：

- `report_pi05_cotrain.py`
  - 适用于 `pi05`
  - 适用于 `pi05_full`
  - 只依赖训练日志
- `report_pi05_teaching_run.py`
  - 适用于 `pi05`
  - 适用于 `pi05_full`
  - 侧重画图和新手友好的解释
- `run_pi05full_subtask_validation.py`
  - 只适用于 `pi05_full`
  - 因为它依赖 `pi05_full` 里的 `generate_subtask_tokens`
  - 不适用于当前仓库的 `pi05`
- `build_pi05full_subtask_manifest.py`
  - 只负责生成 `manifest.json`
  - 适用于给 `run_pi05full_subtask_validation.py` 准备输入

## 2. 输入文件约定

最常用的输入一般是：

- 训练主日志：
  - `logs/node0.log`
- 可选的结构化诊断：
  - `diagnostics/train_diag.jsonl`
- `pi05_full` 子任务验证需要：
  - checkpoint 目录
  - manifest JSON
  - panel JSON
  - sidecar parquet

如果是多卡训练，通常优先看：

- `logs/node0.log`

因为 rank0 往往就包含主要训练指标。

### 2.1 manifest.json 和 panel.json 的区别

- `manifest.json`
  - 是候选数据清单
  - 记录哪些 dataset root 参与这次验证，以及它们对应哪个 sidecar parquet
- `panel.json`
  - 是实际抽中的固定样本清单
  - 记录这次真正拿来评估的帧、episode、subtask 文本

通常流程是：

1. 先生成 `manifest.json`
2. 再跑 `run_pi05full_subtask_validation.py`
3. 如果 `panel.json` 不存在，脚本会自动抽样并写出 `panel.json`
4. 后续再评估别的 checkpoint 时，继续复用同一个 `panel.json`

### 2.2 生成 manifest.json

脚本：

```bash
python scripts/build_pi05full_subtask_manifest.py \
  --dataset-roots-file /path/to/groupA_roots.txt \
  --sidecar-root /mnt/cpfs/datasets/pi05_canonical_sidecars_umi \
  --output /path/to/subtask_validation/manifest.json
```

这个脚本会：

- 读取 `dataset-roots-file`
- 对每个 dataset root 查找：
  - `{sidecar_root}/{dataset_root.name}/summary.json`
  - `{sidecar_root}/{dataset_root.name}/framewise_guidance.parquet`
- 从 `summary.json` 里提取 `robot_type`
- 组装成 `run_pi05full_subtask_validation.py` 需要的 `records` 格式

## 3. co-train 日志总结

脚本：

```bash
python scripts/report_pi05_cotrain.py \
  --log /path/to/logs/node0.log \
  --source-name 0:umi \
  --source-name 1:groupA \
  --output-md /path/to/report.md \
  --output-csv /path/to/report.csv \
  --title "pi05_full 18.5h run summary"
```

这个脚本会解析：

- `loss`
- `loss_source/*`
- `loss_source_norm/*`
- `sampling_ratio/source_*`
- 一部分 `conditioning/*`
- tactile 相关常见指标

它适合回答：

- 总 loss 是下降、持平还是发散？
- 哪个 source 最不稳定？
- 哪个 source 的 loss 一直居高不下？
- tactile / conditioning 指标有没有明显变化？

输出：

- 一个 markdown 报告
- 一个逐 step 的 CSV

## 4. 教学版训练分析

脚本：

```bash
python scripts/report_pi05_teaching_run.py \
  --log /path/to/logs/node0.log \
  --output-dir /path/to/teaching_report \
  --output-md /path/to/teaching_report/report.md \
  --output-csv /path/to/teaching_report/parsed.csv \
  --source-name 0:umi \
  --source-name 1:groupA \
  --action-source 0 \
  --action-source 1 \
  --title "pi05_full training teaching report"
```

这个脚本比上一个更偏“看图说话”，会自动画出常见曲线，例如：

- `loss`
- `loss_global/per_sample_mean`
- `action_ce_loss`
- `flow_mse_loss`
- `subtask_ce_loss`
- `loss_tactile/present`
- `loss_mask/action_active_ratio`
- `loss_mask/language_active_ratio`
- `loss_source/*`

它更适合：

- 跑完一版训练后快速人工检查
- 做汇报截图
- 给团队成员解释这次训练是否健康

输出：

- 多张 PNG 曲线图
- 一个 markdown 解释报告
- 一个解析后的 CSV

## 5. PI05-Full 子任务验证

脚本：

```bash
python scripts/run_pi05full_subtask_validation.py \
  --job-name pi05full_eval_demo \
  --checkpoint-dir /path/to/checkpoint_dir \
  --manifest /path/to/manifest.json \
  --panel-json /path/to/panel.json \
  --output-dir /path/to/subtask_eval \
  --num-samples 12 \
  --device cuda \
  --video-backend pyav
```

这个脚本会：

- 从 manifest 里找带 sidecar 的样本
- 选一批带 `subtask_text_sidecar` 和 `next_subtask_text_sidecar` 的帧
- 加载 `pi05_full` checkpoint
- 调用 `generate_subtask_tokens`
- 计算：
  - `exact_match_rate`
  - `mean_similarity`
- 导出：
  - `summary.json`
  - `samples.json`
  - `samples.csv`
  - `report.md`
  - `report.html`
  - 对应样本图像

它适合回答：

- `pi05_full` 有没有学会 next-subtask 预测？
- 某个 checkpoint 的语言 / 子任务能力有没有明显变好？
- 哪类任务上预测更容易错？

注意：

- 当前仓库里的 `pi05` 没有同样的 `generate_subtask_tokens` 路径，所以这个脚本不支持 `pi05`
- 这个工具更偏定性 + 小规模样本验证，不是大规模 benchmark

## 6. 推荐使用顺序

### 6.1 训练刚结束时

先跑：

```bash
python scripts/report_pi05_cotrain.py ...
```

先确认：

- 总 loss 没炸
- per-source loss 没有某一路完全失控
- tactile / conditioning 指标是有意义的

### 6.2 需要更直观看曲线时

再跑：

```bash
python scripts/report_pi05_teaching_run.py ...
```

重点看：

- 总 loss 是否稳步下降
- `action_ce_loss` 和 `flow_mse_loss` 是否同步改善
- `subtask_ce_loss` 是否明显快于 action loss
- 某个 source 是否明显比别的 source 抖动更大

### 6.3 如果是 pi05_full

再补：

```bash
python scripts/run_pi05full_subtask_validation.py ...
```

重点看：

- `exact_match_rate`
- `mean_similarity`
- 错误样本的定性模式

## 7. 什么时候该怀疑训练有问题

常见危险信号：

- `loss` 长时间上升
- 某个 `loss_source/*` 不降反升，而且 `sampling_ratio` 又不低
- `loss_mask/action_active_ratio` 明显不符合你的数据混合预期
- tactile 相关 loss 一直不动
- `pi05_full` 的 next-subtask 预测几乎全错

如果出现这些情况，建议回头检查：

- 数据混合权重
- batch 内 source 暴露比例
- sidecar 字段是否齐全
- tokenizer / prompt / subtask 字段是否对齐
- tactile 输入是否真的接进了模型

## 8. 一个简单落地例子

假设你的训练输出根目录是：

```text
/mnt/cpfs/runs/pi05full_run_A
```

你可以这样做：

```bash
python scripts/report_pi05_cotrain.py \
  --log /mnt/cpfs/runs/pi05full_run_A/logs/node0.log \
  --source-name 0:umi \
  --source-name 1:groupA \
  --output-md /mnt/cpfs/runs/pi05full_run_A/eval/cotrain_report.md \
  --output-csv /mnt/cpfs/runs/pi05full_run_A/eval/cotrain_report.csv
```

再做一份更适合看图的报告：

```bash
python scripts/report_pi05_teaching_run.py \
  --log /mnt/cpfs/runs/pi05full_run_A/logs/node0.log \
  --output-dir /mnt/cpfs/runs/pi05full_run_A/eval/teaching \
  --output-md /mnt/cpfs/runs/pi05full_run_A/eval/teaching/report.md \
  --output-csv /mnt/cpfs/runs/pi05full_run_A/eval/teaching/parsed.csv \
  --source-name 0:umi \
  --source-name 1:groupA \
  --action-source 0 \
  --action-source 1
```

## 9. 建议

如果你的目标是先把训练健康度监控起来，优先级建议是：

1. 先用 `report_pi05_cotrain.py`
2. 再用 `report_pi05_teaching_run.py`
3. 只有在 `pi05_full` 场景下，再加 `run_pi05full_subtask_validation.py`

这样我们能先把最稳定、最通用的评估链路用起来，再逐步做更细的专项验证。
