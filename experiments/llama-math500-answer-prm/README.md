# Llama MATH-500 答案级 PRM 选优

使用历史 Llama E4 的 500 题和生成后端对照 E2 的同一批 100 题。每题固定已有的 10 条候选、候选顺序、逐条 PRM 分数及原始判分。 [E4 候选池](https://github.com/zhaqian21/adaptive-ttc-lab/pull/21) · [E2 候选池](https://github.com/zhaqian21/adaptive-ttc-lab/pull/23)

| 候选池 | 样本数 | 原单条最高分 | 过滤空答案 | 答案级 PRM 聚合 |
|---|---:|---:|---:|---:|
| E2 Transformers | 100 | 71（71.0%） | 72（72.0%） | 72（72.0%） |
| E2 vLLM | 100 | 62（62.0%） | 63（63.0%） | 69（69.0%） |
| 历史 E4 | 500 | 321（64.2%） | 328（65.6%） | 332（66.4%） |

历史 Llama E4 上，过滤空答案后改对 7 题，原本答对的题都保留。答案级聚合相对原选优改对 15 题、改错 4 题，净增 11 题。E2 vLLM 相对原选优改对 7 题，原本答对的题都保留。

## 选优规则

- 原选优取 PRM 最高分的候选，分数并列选较早的候选，全零回退到最后一条。
- 过滤空答案后，在能提取非空答案的候选中取最高分，分数并列取较早的一条。所有答案为空时沿用原选优。
- 答案级聚合对提取答案去掉首尾空白后按字符串分组，累加组内 PRM 分数。组间平分时先比较最高单条分数，再比较代表候选的顺序。以组内最高分的候选作为代表。没有可用答案组或所有组总分为零时沿用原选优。 [原聚合规则](https://github.com/zhaqian21/adaptive-ttc-lab/pull/8)

三个方案使用相同的历史生成和评分请求。在相同候选池上复算原选优及两个替换规则，并按准确率的逐题变化保存改对和改错列表。

## 复跑

~~~bash
python experiments/llama-math500-answer-prm/replay_selection.py --output /tmp/llama-answer-prm
~~~

输出包括每题的答案组和三个选择。归档记录输入与逐题结果的 SHA-256。 [汇总](results/summary.json) · [E4 逐题](results/e4_paired.jsonl) · [E2 Transformers 逐题](results/e2_transformers_paired.jsonl) · [E2 vLLM 逐题](results/e2_vllm_paired.jsonl)
