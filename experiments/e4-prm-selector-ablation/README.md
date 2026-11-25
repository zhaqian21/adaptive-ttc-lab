# E4 的 PRM 选优消融与 AIME24 验证

复用 [PR #26](https://github.com/zhaqian21/adaptive-ttc-lab/pull/26) 和 [PR #27](https://github.com/zhaqian21/adaptive-ttc-lab/pull/27) 的 Llama、Qwen 在 MATH-500 与 AMC23 上的在线候选。对每个完整 A 候选池回放固定 K=10 和五条一致早停。在已运行的 B 候选池上，保留实际早停记录，并对完成 10 条候选的题重新选优。全部选优在 CPU 上使用已保存的 PRM 均值。

| 选优规则 | 如何选答案 |
|---|---|
| 单条最高 | 取原始 E4 的最高 PRM 候选 |
| 全组求和 | 按非空提取答案分组，对同答案的所有候选分数求和 |
| Top-2 求和 | 沿用分组，每个答案只累加组中最高的两条，有一条就计一条 |

答案分组使用去掉首尾空白后的提取答案。分组比较同分时，依次用组内最高分和该候选的原始顺序处理并列。组内最高分候选代表该答案。全部有效组得分为零时回退到原始 E4。三种选优在五条相同非空答案触发早停时均直接返回该答案。逐题结果区分早停答错、已见候选缺少正确解、候选池有正确解但选优漏选。同池的逐题配对另外标出早停导致原本正确的题被答错。

## 四组回放

以下每格为答对题数，按单条最高、全组求和、Top-2 求和排列。

| 模型 · 数据集 | A 池固定 K=10 | A 池五条早停 | B 池保留实际早停 |
|---|---|---|---|
| Llama · MATH-500（500 题） | 314 / 320 / 323 | 313 / 320 / 322 | 308 / 325 / 325 |
| Qwen · MATH-500（500 题） | 401 / 396 / 400 | 400 / 396 / 399 | 394 / 387 / 390 |
| Llama · AMC23（40 题） | 17 / 17 / 17 | 17 / 17 / 17 | 15 / 17 / 17 |
| Qwen · AMC23（40 题） | 23 / 23 / 24 | 22 / 22 / 23 | 23 / 25 / 24 |

Qwen MATH-500 在线 A 对 B 错的 16 题中，1 题五条一致后答错，5 题的 B 候选池缺少正确解，10 题在 B 候选池有正确解的情况下选错。四组逐题分类和计数见 [汇总](results/summary.json)。

## 锁定规则与 AIME24 验证

以两个 MATH-500 A 池的五条早停回放作为校准集。两种候选规则与全组求和逐模型比较，要求两组增益都非负，再比较较小的增益。Top-2 在两组分别多答对 2 题和 3 题，选为 AIME24 的固定规则。锁定参数和校准值见 [选择记录](selection_lock.json)。

AIME24 使用 [Llama 原 E4 复现](https://github.com/zhaqian21/adaptive-ttc-lab/pull/20) 和 [Qwen 原 E4 复现](https://github.com/zhaqian21/adaptive-ttc-lab/pull/3) 的固定候选。两套各 30 题，数据 ID 顺序一致。提取时核对 PRM 的逐条记录、各题最终结果、种子、生成请求数和输出 token，包含原始记录指纹。回放用原始单条最高分重建公开归档的逐题结果。

| 模型 | 五条早停 + 单条最高 | 五条早停 + 全组求和 | 五条早停 + 锁定 Top-2 | 早停题数 | 回放 PRM 请求 |
|---|---:|---:|---:|---:|---:|
| Llama | 4/30 | 5/30 | 5/30 | 0 | 300 |
| Qwen | 4/30 | 4/30 | 4/30 | 1 | 290 |

Top-2 在 AIME24 两套候选上与全组求和的正确数相同。早停只在 Qwen 的 1 题触发，Qwen 的外层 PRM 回放请求从 300 降到 290。AIME24 完整候选池中，Llama 有 23 题、Qwen 有 24 题缺少正确解。评测只复用历史生成和打分记录，报告回放的候选数与外层 PRM 请求数。

## 复跑

从仓库根目录执行。若 PR #26 仍在独立检出目录，用 `--llama-math500` 传入其中包含 `fixed` 和 `early` 的在线结果目录。六个逐题 JSONL 和汇总都可从归档重建，输入版本见 [来源与 SHA-256](sources.json)。

~~~bash
exp=experiments/e4-prm-selector-ablation
python "$exp/replay.py" --llama-math500 "$LLAMA_MATH500_RESULTS" --output /tmp/e4-prm-selector-ablation
~~~

若要从原始运行重建 AIME24 输入，`RAW_AIME24` 指向包含 `gpu-0/E4`、`gpu-1/E4` 的评测目录。分别用各模型对应的公开 E4 `predictions.jsonl` 作为参考。

~~~bash
python "$exp/prepare_aime24.py" --model llama --raw "$RAW_AIME24" \
  --reference "$E4_ARCHIVED_PREDICTIONS" --output /tmp/llama_aime24.jsonl
~~~

[归档输入](inputs) · [逐题消融与留出验证](results) · [结果汇总](results/summary.json)
