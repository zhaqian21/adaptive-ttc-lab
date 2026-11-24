# MATH-500 分歧题答案级 PRM 聚合

离线回放使用 [PR #7 的正式逐题记录](https://github.com/zhaqian21/adaptive-ttc-lab/tree/135d4acc8379ad5f3f1e2a4cab47aec7bdbd982f/experiments/math500-adaptive-k/results/full)。逐题复用自适应 K 的停止决定、候选前缀和已记录的 PRM 分数。308 道提前停止的题直接返回原答案。另有 191 道题进入 PRM 选优，1 道题在预算内没有完成候选。

## 选优规则

- 自适应 K 基线对候选使用原 PRM mean_reward 单条最高分。所有分数为零时沿用作者代码的最后一条回退规则。
- 答案级 PRM 聚合将提取答案去掉首尾空白后按字符串分组，排除空答案。在每组内累加所有候选的 mean_reward，用总分最高的答案组。组间平分时比较各组最高单条分数，再取最先生成的最高分代表。在没有可用组或有效总分时沿用基线选择。
- 多数票用同一分组及平分规则，按支持该答案的候选数排序。PRM 分数只用于平分和选取组内代表。

分组使用字符串等值。每道题的真值标签来自原 MATH-500 审计记录，只用于计算正确率。回放脚本只读取该策略当时可用的候选前缀。

## 回放结果

| 方法 | 全部题答对 | 进入 PRM 的题答对 | 相对自适应 K 改对 | 相对自适应 K 改错 |
| --- | ---: | ---: | ---: | ---: |
| 自适应 K | 383/500（76.6%） | 105/191 | 0 | 0 |
| 答案级 PRM 累加 | 387/500（77.4%） | 109/191 | 6 | 2 |
| 多数票 | 383/500（76.6%） | 105/191 | 9 | 9 |

答案级 PRM 累加净追回 4 题，正确率与 PR #7 的完整重采样持平。三组都沿用自适应 K 的生成输入与输出 token 总量 33,906,655 和 PRM 请求数 1,505，回放没有发出模型请求。相比同轮完整重采样，生成逻辑 token 少 45.2%，PRM 请求少 72.2%。进入 PRM 的 191 题中，有 125 题至少含一个正确候选，答案级聚合选对其中 109 题。以上结果来自 PR #7 的同一批候选。

## 复跑与审计

从仓库根目录运行，使用 Python 3.8 或以上。默认读取仓库中 PR #7 的结果，也可通过 --source 指向包含同一份 predictions.jsonl 和 summary.json 的目录。脚本用固定 SHA-256 校验两个输入，复核全部 500 个题号与基线答案、正确率和成本。

~~~bash
python -m unittest discover -s experiments/math500-prm-answer-aggregation -v
python experiments/math500-prm-answer-aggregation/replay.py \
  --output experiments/math500-prm-answer-aggregation/runs/replay
~~~

逐题导出包含原 PRM 候选前缀长度、共享成本、答案分组权重、每种方法选择的候选和正确性。摘要包含逐题导出的 SHA-256，可对照归档文件。

[逐题结果](results/predictions.jsonl) · [回放摘要](results/summary.json)
