# Llama MATH-500 的同文本 PRM 打分对照

固定 PR #21 的 500 道题和两组各 5,000 条候选。历史整段行沿用 PR #20 的已保存分数。新回放共用同一份正文和段落，只控制正文里新增的步骤分隔符。 [原始复现](https://github.com/zhaqian21/adaptive-ttc-lab/pull/20) · [分步候选池](https://github.com/zhaqian21/adaptive-ttc-lab/pull/21)

| PRM 评分方式 | E2 | E4 | E2 − E4（百分点） |
|---|---:|---:|---:|
| 历史整段 | 322/500（64.4%） | 321/500（64.2%） | +0.2 |
| 同文本整段 | 327/500（65.4%） | 330/500（66.0%） | -0.6 |
| 同文本逐步，取末步 | 314/500（62.8%） | 308/500（61.6%） | +1.2 |
| 同文本逐步，取均值 | 315/500（63.0%） | 314/500（62.8%） | +0.2 |

固定正文，仅增加步骤分隔符并取末步，E2 下降 2.6 个百分点，E4 下降 4.4 个百分点。同一套逐步分数从末步改取均值，E2 回升 0.2 个百分点，E4 回升 1.2 个百分点。两组的逐步均值均低于同文本整段。

同文本整段时 E4 多答对 3 题，逐步均值时 E2 多答对 1 题。三种同文本评分方式下，E2 比论文表 1 的 44.6% 仍高 18.2–20.8 个百分点。 [论文表 1](https://arxiv.org/html/2602.01070v5#S4.T1)

## 评分口径

从 PR #21 的逐步候选池取出原顺序步骤文本，用相同空行拼接正文。整段版只在末尾放一个 `<extra_0>`，逐步版额外在每个中间步骤末尾放置分隔符。删掉逐步版的中间分隔符，正文逐字符等于整段版。同一候选的两种输入放进同一批前向计算，使用相同的 Qwen2.5-Math-PRM-7B BF16 权重、系统提示和 token 掩码规则。

逐步版使用一份步骤分数分别取末步和算术均值。每题仍从原顺序的 10 条候选中选最高分，最高分并列时选最早的候选，全零时选最后一条。逐题对照沿用各候选的历史判分。

第一轮写入的 3,717 对分数保存在 SQLite 检查点。修正进度计数后从断点继续，最终从归档中提取这些分数并复查 SHA-256。 [恢复记录](results/checkpoint_resume.json)

历史整段还包含公开代码中的计划、工具和计算配置等字段。历史整段和新整段的差值因此包含评分正文变化和批处理数值差异。同文本三组用于比较分隔符和聚合规则。步骤根据存档文本重建，候选生成沿用历史执行。 [论文评分描述](https://arxiv.org/html/2602.01070v5#S2)

## 复跑与审计

归档包含 10,000 个原始分数对、1,000 条逐题选优结果、两卡的评分配置及输出 SHA-256。先用以下命令从现有分数重建结果。

~~~bash
exp=experiments/llama-math500-same-text-prm
python "$exp/compare_control.py" --output /tmp/llama-same-text-check
~~~

重新打分时，使用与 PR #21 相同的 Python 环境、固定上游检出和 PRM 权重。将 `PRM` 指向固定权重目录，`UPSTREAM` 指向作者代码检出，`SCORES` 指向新的评分目录。两张卡按题号奇偶各处理两组的一半候选。

~~~bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$UPSTREAM" python "$exp/score_control.py" \
  --model "$PRM" --output "$SCORES/gpu-0" --lane 0 --batch-size 32 --batch-tokens 8192
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$UPSTREAM" python "$exp/score_control.py" \
  --model "$PRM" --output "$SCORES/gpu-1" --lane 1 --batch-size 32 --batch-tokens 12288
python "$exp/compare_control.py" --scores "$SCORES" --output /tmp/llama-same-text-check
~~~

[E2 逐题](results/e2_paired.jsonl) · [E4 逐题](results/e4_paired.jsonl) · [配对汇总与指纹](results/summary.json)
