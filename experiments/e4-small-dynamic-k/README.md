# 三条一致早停，最多五条的 E4 回放

先生成三条完整候选。三条提取答案去掉首尾空白后均非空且完全一致时直接返回，跳过外层 PRM。其他题补到五条，用原始 E4 的单条最高 PRM 分数选优。最高分并列时取最早候选，候选分数全为零时取本轮末条。这个规则适用于两种模型和四个数据集。

所有对照逐题共用候选顺序、原判分和 PRM 分数。固定三、固定五、固定十均对各自候选打分。五条一致早停从五条出发，一致时直接返回，否则继续到十条。表中的成本按归档请求回放，输入与输出 token 都计入。Llama MATH-500 的两次失败尝试无 token 记录，所有方案均按留存请求回算。

## 回放结果

| 数据集 · 模型 | 固定 10 答对 | 固定 5 答对 | 动态 3 或 5 答对 | 动态平均候选数 | 固定 5 → 动态生成 token | 固定 5 → 动态 PRM 请求 |
|---|---:|---:|---:|---:|---:|---:|
| MATH-500 · Llama | 321/500 | 305/500 | 303/500 | 4.47 | 30,555,399 → 28,034,810 | 2,500 → 1,840 |
| MATH-500 · Qwen | 397/500 | 384/500 | 384/500 | 3.81 | 36,919,872 → 30,115,089 | 2,500 → 1,010 |
| AMC23 · Llama | 17/40 | 16/40 | 16/40 | 4.70 | 2,843,842 → 2,696,036 | 200 → 170 |
| AMC23 · Qwen | 23/40 | 24/40 | 24/40 | 4.20 | 4,019,718 → 3,572,652 | 200 → 120 |
| AIME24 · Llama | 4/30 | 5/30 | 5/30 | 5.00 | 2,666,682 → 2,666,682 | 150 → 150 |
| AIME24 · Qwen | 4/30 | 3/30 | 3/30 | 4.93 | 3,853,416 → 3,839,259 | 150 → 145 |
| AMO-Bench · Llama | 1/50 | 1/50 | 1/50 | 5.00 | 8,706,038 → 8,706,038 | 250 → 250 |
| AMO-Bench · Qwen | 3/50 | 3/50 | 3/50 | 4.80 | 11,986,391 → 11,347,217 | 250 → 225 |

相比固定 5 条，MATH-500 的 Llama 少答对 2 题，Qwen 改对 2 题、改错 2 题。两者的 token 分别减少 8.25% 和 18.43%。比固定 10 条分别少答对 18 和 13 题。其余三个数据集的逐题正确集合与固定 5 条相同。

## 复跑

- MATH-500 使用 [Llama 原始候选账本](https://github.com/zhaqian21/adaptive-ttc-lab/pull/25) 和 [Qwen 历史候选账本](https://github.com/zhaqian21/adaptive-ttc-lab/pull/15)。
- AMC23 使用 [PR #27](https://github.com/zhaqian21/adaptive-ttc-lab/pull/27) 两个模型的固定十条候选，按原始单条 PRM 重新选优。
- AIME24 与 AMO-Bench 的成本从原始请求恢复。使用各模型的固定 tokenizer 复建最终两次请求的输入长度，再使用保存的顺序和耗时唯一划分十条轨迹。保留前 3、前 5 和完整 10 条成本。 [AIME24 候选来源](https://github.com/zhaqian21/adaptive-ttc-lab/pull/28) · [AMO-Bench 候选来源](https://github.com/zhaqian21/adaptive-ttc-lab/pull/30)

从仓库根目录执行。`QWEN_MATH_POOL` 指向 [PR #15 候选账本](https://github.com/zhaqian21/adaptive-ttc-lab/pull/15) 里的 `results/candidate_pool.jsonl`。脚本核对该文件及其他候选账本的 SHA-256。

~~~bash
exp=experiments/e4-small-dynamic-k
python "$exp/dynamic_k.py" --qwen-pool "$QWEN_MATH_POOL" --output /tmp/e4-small-dynamic-k
~~~

从原始运行恢复 AIME24 或 AMO-Bench 输入时，`RAW_DATASET` 指向包含两个 GPU 分片的 `formal/aime24` 或 `formal/amo`。用相应模型的公开 E4 `predictions.jsonl` 作为参考，并传入包含固定上游代码和模型 tokenizer 的 `SOURCE`。

~~~bash
python experiments/e4-amo-bench-replay/prepare_pool.py --model qwen --dataset aime24 --all-prefixes \
  --raw "$RAW_DATASET" --reference "$ORIGINAL_PREDICTIONS" --source "$SOURCE" --output /tmp/qwen_aime24.jsonl
~~~

[逐题结果](results) · [五种规则的全量指标](results/summary.json) · [输入校验](inputs/checksums.json)
