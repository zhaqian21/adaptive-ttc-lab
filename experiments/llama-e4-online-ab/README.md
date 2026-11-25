# Llama E4 五条一致早停在线对照

在 MATH-500 的同一批 500 题上分别运行两套 Llama E4。沿用上游 E4 的 Dynamic 搜索、候选生成和外层 PRM 评分。

- **A 固定 K=10 + 答案级 PRM 聚合**。依次生成 10 条候选，每条按公开代码的评分方式调用一次外层 PRM。去掉答案首尾空白后按非空答案分组，以组内 PRM 均值之和选答案，沿用 [PR #24](https://github.com/zhaqian21/adaptive-ttc-lab/pull/24) 的回退和并列规则。
- **B 五条一致早停 + 答案级 PRM 聚合**。前五条都抽取到相同的非空答案时立即返回第一条的答案，跳过后五条和全部外层 PRM 请求。其他题按 A 的流程完成 10 条并聚合。早停条件沿用 [PR #25](https://github.com/zhaqian21/adaptive-ttc-lab/pull/25)。

## 配置

使用 24 服务器的 GPU 0、1。每个 H20 各加载一份 Llama-3.1-8B-Instruct BF16 和 Qwen2.5-Math-PRM-7B BF16。每卡并发评测 24 题，vLLM 0.33.0 开启连续批处理、分块预填充和前缀缓存，上下文 32768。各轮重启两套模型服务，正式 A、B 从空前缀缓存开始。运行时采样参数和分片题号见各轮的 GPU 配置文件。

两轮按相同的题号奇偶分片和提交顺序运行。题目按历史 E4 的生成输出 token 降序排列，同长度按题号升序，以减少长题拖尾。种子按题号、重试序号和当前题的生成请求序号确定。相同种子下，连续批处理的请求组合仍可能影响生成，所以对齐前五条候选的文本指纹和提取答案，并在 A 轮候选上回放 B 的早停规则。

20 题试跑从历史上触发和未触发早停的两组题各选 10 题。在线试跑答对 15 题。6 道早停题各生成 5 条候选，外层 PRM 请求为 0。其余 14 道题各生成 10 条并请求评分，共 140 次外层 PRM 调用。正式指标来自两轮各自完整的 500 题。

## 复跑与审计

按 [Llama 双卡复现说明](../llama-table1-825/README.md#24-号双卡接续) 准备模型、数据集和 Python 环境。用独立运行目录保存结果。24 号启动脚本会检查已有 Qwen 服务、切换模型，并在退出时恢复服务。

~~~bash
experiment=experiments/llama-e4-online-ab
bash experiments/llama-table1-825/run_24.sh "$OUTPUT" "$VLLM_PYTHON" "$experiment/online_ab.py"
~~~

每个阶段输出 `results/<phase>/predictions.jsonl`、`summary.json` 和两张卡各自的 `gpu-<lane>/config.json`。归档时两个 worker 的配置文件分别保存为 `gpu-0_config.json` 和 `gpu-1_config.json`。审计脚本逐题核对采样轮数、早停条件、候选得分、选优结果、种子、服务请求数和累加 token，并重建配对记录与汇总表。

~~~bash
original_runner="$(mktemp)"
git show a5dcb0c38efe34e410eb8ff9c80b5bba5b7696cf:experiments/llama-e4-online-ab/online_ab.py > "$original_runner"
python experiments/llama-e4-online-ab/compare_online.py --source "$original_runner" --output /tmp/llama-e4-online-ab
~~~

生成 token 使用 vLLM 响应中的输入与输出用量。请求计数包含失败和重试。有请求缺少用量时单独计数。外层 PRM 请求按实际发出的 HTTP 调用统计。双卡墙钟从启动评测 worker 到两张卡都完成，模型服务加载时间另计。逐题 P95 从题目进入 worker 到写出候选和答案前，包含题内重试，线程池排队时间另计。

## 数据集与模型迁移

在 Llama 的 MATH-500 两轮结束后，24 服务器按以下顺序接续。三项沿用相同的 A、B 规则、外层 PRM 和双卡配置。

| 顺序 | 生成模型 | 数据集 | 试跑 | 正式 |
|---|---|---|---:|---:|
| 1 | Llama-3.1-8B-Instruct | AMC23 | 8 题 | 每组 40 题 |
| 2 | Qwen2.5-7B-Instruct | MATH-500 | 20 题 | 每组 500 题 |
| 3 | Qwen2.5-7B-Instruct | AMC23 | 8 题 | 每组 40 题 |

AMC23 沿用 [PR #6](https://github.com/zhaqian21/adaptive-ttc-lab/pull/6) 的 Parquet 版本、题号和 MATH 判分。Llama 和 Qwen 共用一份 AMC23 题序和试跑子集，正式题目按历史重采样的输出 token 降序提交。两种模型也共用 MATH-500 的分片和正式提交顺序。Qwen 的 MATH-500 试跑按 [PR #15](https://github.com/zhaqian21/adaptive-ttc-lab/pull/15) 的历史记录选择可早停、有分歧的题各十道。各项均保存冻结题序和数据指纹。

准备一个新的队列目录，包含 `llama-amc23` 与 `qwen-math500`。两者的 `source` 可指向同一份准备好的资产目录。资产目录沿用上文 Llama 双卡环境，另在 `source/datasets/amc23.parquet` 放置固定 AMC23 Parquet，在 `source/assets/model` 链接固定 Qwen 快照。两项的 `assets.exit` 各记录资产预检的返回码。

~~~bash
bash experiments/llama-e4-online-ab/run_followups_24.sh "$PREVIOUS_RUN" "$QUEUE" "$VLLM_PYTHON"
~~~

入口等待上一轮通过双卡审计并恢复常驻服务。每个接续实验写入自己的评测日志和逐题结果，结束后生成 `results/comparison` 并检查常驻服务。队列进度见 `queue-status`。任一步骤出错时，`queue.exit` 记录失败返回码。

Qwen AMC23 使用独立队列目录，准备 `qwen-amc23/source` 和 `qwen-amc23/assets.exit` 后，将前一队列目录作为首个参数。入口会在 Qwen MATH-500 完成、服务恢复和配对审计全部通过后启动。

~~~bash
bash experiments/llama-e4-online-ab/run_followups_24.sh "$QUEUE" "$QWEN_AMC23_QUEUE" "$VLLM_PYTHON" qwen-amc23
~~~

### Llama AMC23 正式结果

| 方案 | 答对 | 早停题数 | 生成 token | 外层 PRM 请求 | 双卡墙钟 |
|---|---:|---:|---:|---:|---:|
| A 固定 K=10 + 聚合 | 17/40 | 0 | 5,616,410 | 400 | 466.3 秒 |
| B 五条一致早停 + 聚合 | 17/40 | 4 | 5,469,642 | 360 | 476.9 秒 |

在线配对中 B 改对 4 题、改错 4 题。生成 token 减少 2.6%，外层 PRM 请求减少 10.0%，双卡墙钟增加 10.6 秒。同一份 A 组候选的回放中，3 道题达到早停条件，答对题的集合一致。 [正式配置与逐题记录](results/llama-amc23) · [完整对照](results/llama-amc23/comparison.json)

### Qwen MATH-500 正式结果

| 方案 | 答对 | 早停题数 | 生成 token | 外层 PRM 请求 | 双卡墙钟 |
|---|---:|---:|---:|---:|---:|
| A 固定 K=10 + 聚合 | 396/500 | 0 | 72,865,037 | 5,000 | 54.0 分钟 |
| B 五条一致早停 + 聚合 | 387/500 | 273 | 58,527,401 | 2,270 | 49.0 分钟 |

在线配对中 B 改对 7 题、改错 16 题。生成 token 减少 19.7%，外层 PRM 请求减少 54.6%，双卡墙钟减少 9.4%。同一份 A 组候选回放有 273 题达到早停条件，答对题的集合一致。原始单条最高分选优在该候选池答对 401 题，答案级聚合答对 396 题。两轮前五条提取答案序列有 252 题相同，前五条完整文本序列有 2 题相同。 [正式配置与逐题记录](results/qwen-math500) · [完整对照](results/qwen-math500/comparison.json)
