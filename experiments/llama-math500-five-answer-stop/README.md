# Llama MATH-500 E4 的五条一致早停

在历史 Llama E4 的 500 道题上，对照原单条最高分和答案级聚合，并为两个选优方法独立开关五条一致早停。固定原候选顺序、答案、判分和 PRM 分数。 [原复现](https://github.com/zhaqian21/adaptive-ttc-lab/pull/20) · [答案级聚合](https://github.com/zhaqian21/adaptive-ttc-lab/pull/24)

| 方案 | 正确率 | 早停题数 | 生成 token | PRM 请求 |
|---|---:|---:|---:|---:|
| 历史 E4 | 321/500（64.2%） | 0 | 61,250,837 | 5,000 |
| 原选优 + 五条一致早停 | 323/500（64.6%） | 93 | 56,938,125 | 4,070 |
| 答案级 PRM 聚合 | 332/500（66.4%） | 0 | 61,250,837 | 5,000 |
| 答案级聚合 + 五条一致早停 | 332/500（66.4%） | 93 | 56,938,125 | 4,070 |

加上早停后，答案级聚合保住 332/500（66.4%）和相同的答对题集合。93 题触发早停，生成 token 少 4,312,712（7.0%），外层 PRM 请求少 930 次（18.6%）。相对历史 E4，联合方案改对 15 题、改错 4 题。

## 规则与成本

检查前五条提取答案。去掉首尾空白后都非空且完全一致时，采用首条答案，计入前五条完整轨迹的生成成本，外层 PRM 请求计为 0。其余题沿用原来的十条预算和对应选优规则。

生成 token 包含成功留存的生成模型调用的输入和输出。根据原上游提示和 Llama tokenizer 重建每条轨迹末尾的两个请求长度，再用规划调用与用时约束复核唯一边界。500 题的分段均可以唯一确定。账本合计 85,184 次生成请求、9,305,961 个输出 token，和原 E4 归档一致。历史记录中同一题的两次失败尝试缺少逐请求 token，回放按两组的成功留存轨迹核算成本。 [分段方法](https://github.com/zhaqian21/adaptive-ttc-lab/pull/14)

## 复跑

~~~bash
exp=experiments/llama-math500-five-answer-stop
python "$exp/replay_early_stop.py" --output /tmp/llama-five-stop
~~~

可用原始双卡记录重建费用账本。将 `HISTORY` 指向包含 `formal/math/gpu-0/E4` 和 `formal/math/gpu-1/E4` 的结果目录，`SOURCE` 指向包含固定上游检出和 Llama tokenizer 的资产工作区。使用原评估 Python 环境执行，下列命令在 CPU 上恢复提示长度。

~~~bash
python "$exp/prepare_costs.py" --history "$HISTORY" --source "$SOURCE" \
  --pool experiments/llama-math500-step-prm-replay/inputs/e4_pool.jsonl.gz \
  --baseline experiments/llama-table1-24-2026-11-25/results/math/E4/predictions.jsonl \
  --output /tmp/e4_cost_ledger.jsonl
~~~

[逐轨迹费用](results/e4_cost_ledger.jsonl) · [逐题消融](results/paired.jsonl) · [汇总与 SHA-256](results/summary.json)
