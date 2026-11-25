# Llama MATH-500 的逐步 PRM 选优回放

使用公开代码复现的历史 E2（Direct + PRM）和 E4（Dynamic + PRM），每组固定同一批 500 题、每题 10 条候选。逐步版给每条已完成轨迹的各个步骤评分，取算术均值，再从 10 条中选均值最高的一条。两组分别与原整段评分逐题配对。 [原始复现](https://github.com/zhaqian21/adaptive-ttc-lab/pull/20)

| 设置 | 历史整段 | 逐步均值 | 变化（百分点） | 改对 | 改错 | 换选轨迹 |
|---|---:|---:|---:|---:|---:|---:|
| E2 | 322/500 (64.4%) | 312/500 (62.4%) | -2.0 | 8 | 18 | 297 |
| E4 | 321/500 (64.2%) | 317/500 (63.4%) | -0.8 | 16 | 20 | 317 |

原整段版 E2 领先 1 题，逐步均值版 E4 领先 5 题（63.4% 对 62.4%）。E2 与论文表 1 的差距从 +19.8 缩到 +17.8 个百分点。E4 复现比论文低 2.0 个百分点。 [论文表 1](https://arxiv.org/html/2602.01070v5#S4.T1)

## 评分输入

E2 将历史自由文本回答拆为步骤。E4 按原顺序使用历史推理步骤和最终自由文本，剥掉重述、摘要的工具标签，移除只记录工具判定和分数的核验回执。分步规则沿用之前的论文口径实现，已有分隔符优先，其次是空行分段，单段时按非空行切分。PRM 只接收题目和这些文本，沿用原系统提示和固定的 Qwen2.5-Math-PRM-7B BF16 权重。在同一次因果前向计算里从每个 `<extra_0>` 读出正类概率。最高分并列时选最早的候选，全零用最后一条。

这里用归档文本重建步骤，测试论文式外层聚合在固定候选池上的效果。与原整段输入相比，逐步版同时改变了分隔方式和工具元数据的使用。论文在生成过程中定义步骤边界，并将逐步评分用于部分生成策略。 [论文 §2](https://arxiv.org/html/2602.01070v5#S2)

E2 和 E4 分别评分 72,984 和 106,538 个步骤。两张 H20 分别处理 E2 和 E4。14 道历史整段输入的最高分候选全部复现。批量前向的分数有小幅数值差异，整段抽查最大绝对差为 0.015625。

逐步版选中的候选与判分不同的最佳候选分差不超过 0.01 时，逐条重给该题的 10 条候选打分。共核对 E2 33 题、E4 38 题。在主结果里替换这些题的选择后，E2 为 312/500，E4 为 316/500，仍由 E4 领先。 [E2 数值核对](results/e2_near_ties.json) · [E4 数值核对](results/e4_near_ties.json)

## 复跑

两个压缩候选池保存了实际送分的步骤文本、各候选的历史判分和原分数，以及原题目结果的 SHA-256。最终的两份逐题配对结果和原始逐步分数均已归档。

~~~bash
exp=experiments/llama-math500-step-prm-replay
python "$exp/compare_replay.py" --output /tmp/llama-step-prm-check
~~~

重新打分时，使用 PR #20 固定的上游检出和 Python 环境，将 `UPSTREAM` 指向该检出，将 `PRM` 指向固定 PRM 资产目录。按下面的启动参数分别执行两组，打完后用 `compare_replay.py --scores "$SCORES"` 汇总。按 GPU 预算设置 `CUDA_VISIBLE_DEVICES`。

~~~bash
PYTHONPATH="$UPSTREAM" python "$exp/score_steps.py" --model "$PRM" \
  --pool "$exp/inputs/e2_pool.jsonl.gz" --output "$SCORES/e2" --batch-size 32 --batch-tokens 8192
PYTHONPATH="$UPSTREAM" python "$exp/score_steps.py" --model "$PRM" \
  --pool "$exp/inputs/e4_pool.jsonl.gz" --output "$SCORES/e4" --batch-size 32 --batch-tokens 12288
python "$exp/compare_replay.py" --scores "$SCORES" --output /tmp/llama-step-prm-check
~~~

服务器保留原始轨迹目录时，可用 `prepare_pool.py --history "$HISTORY" --baseline "$BASELINE" --output "$INPUTS"` 重建压缩池。把 `HISTORY` 指向包含 `formal/math` 的运行结果目录，`BASELINE` 指向 PR #20 归档中包含 `E2`、`E4` 的 MATH-500 结果目录。评分时增加 `--history "$HISTORY"` 可复跑整段输入抽查。

把 `CASE` 设为 `e2` 或 `e4` 可复跑分数接近的题。

~~~bash
PYTHONPATH="$UPSTREAM" python "$exp/audit_near_ties.py" \
  --pool "$exp/inputs/${CASE}_pool.jsonl.gz" --paired "$exp/results/${CASE}_paired.jsonl" \
  --model "$PRM" --output "/tmp/${CASE}_near_ties.json"
~~~

[E2 逐题对照](results/e2_paired.jsonl) · [E4 逐题对照](results/e4_paired.jsonl) · [汇总和 SHA-256](results/summary.json)
