# AMO-Bench E4 的答案聚合和五条一致早停

使用 Llama-3.1-8B-Instruct 与 Qwen2.5-7B-Instruct 的原始 AMO-Bench E4 十条候选，各 50 题。锁定题序、每条候选的提取答案、原判分和 Qwen2.5-Math-PRM-7B 分数。用原始单条最高 PRM 选择逐题复核公开归档。 [Llama 原始结果](https://github.com/zhaqian21/adaptive-ttc-lab/pull/20) · [Qwen 原始结果](https://github.com/zhaqian21/adaptive-ttc-lab/pull/3)

## 回放结果

| 方案 | Llama 答对 | Qwen 答对 |
|---|---:|---:|
| 原始 E4，固定十条 + 单条最高 PRM | 1/50（2%） | 3/50（6%） |
| 单条最高 PRM + 五条一致早停 | 1/50（2%） | 3/50（6%） |
| 固定十条 + 答案级全组求和 | 1/50（2%） | 2/50（4%） |
| **答案级全组求和 + 五条一致早停** | **1/50（2%）** | **2/50（4%）** |
| Top-2 求和 + 五条一致早停 | 1/50（2%） | 2/50（4%） |

Qwen 少答对的 1 题来自聚合，数据集 ID 为 20。正确答案的单条 PRM 分数约为 0.410，同答案只有一条。聚合选中由六条候选支持的错误答案，总分约为 0.781。Qwen 触发早停的另一题在两种方案下都答错。

| 模型 | 五条一致的题数 | 原始 E4 → 联合方案生成 token | 原始 E4 → 联合方案 PRM 请求 |
|---|---:|---:|---:|
| Llama | 0/50 | 17,787,856 → 17,787,856 | 500 → 500 |
| Qwen | 1/50 | 24,032,870 → 23,675,471（−1.49%） | 500 → 490（−2.0%） |

成本按历史生成请求中保存的输入和输出 token 回算。原始请求账本中的模型调用数、输出 token 和最终答案均与两份公开归档逐题吻合。对唯一的早停题，使用固定 tokenizer 和原提示复建末尾请求的输入长度，结合保存的用时唯一划分十条候选，取前五条的请求成本。两套输入共 100 题，无缺失用量的重试记录。数字沿用原 E4 的判分。 [逐题对照与校验](results)

## 复跑

从仓库根目录执行。归档 [输入](inputs) 只保存候选答案、分数、判分、成本和指纹。重新运行会对照 PR #3、#20 归档的原始预测，并复核两个模型对应的题号。

~~~bash
exp=experiments/e4-amo-bench-replay
python "$exp/replay_amo.py" --output /tmp/e4-amo-bench-replay
~~~

从服务器原始运行恢复输入时，`RAW_AMO` 指向包含两个 GPU 分片的 `formal/amo`，`SOURCE` 指向包含固定上游代码和相应模型 tokenizer 的资产目录。Llama 和 Qwen 分别使用各自公开归档中的 E4 `predictions.jsonl`。

~~~bash
python "$exp/prepare_pool.py" --model qwen --raw "$RAW_AMO" \
  --reference "$ORIGINAL_PREDICTIONS" --source "$SOURCE" --output /tmp/qwen.jsonl
~~~
