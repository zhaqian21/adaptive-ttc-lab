# MATH-500 自适应候选数对照

在 [MATH-500 修正与重采样对照](https://github.com/zhaqian21/adaptive-ttc-lab/pull/4) 的同一采样入口上比较四种策略。生成器为 Qwen2.5-7B-Instruct，沿用作者公开 E4 的规划、工具和推理配置，选优使用 Qwen2.5-Math-PRM-7B 的 mean_reward。版本固定在分片配置中。

生成使用 [作者主配置](https://github.com/AhsanBilal7/adaptive-test-time-compute/blob/d9ca5f71f5c7d155153e8f7decd8b1c524968775/config.yaml#L172) 的温度 0.7、top_p 0.9 和单次输出上限 1,024 token。作者默认配置为 Llama、100 题、10 轮。本对照沿用项目此前的 Qwen E4 复现环境，包括完整 MATH-500、vLLM 的 32,768 上下文长度和相同的模型权重。四组共用解码设置，停止规则和配对种子由本实验指定。

| 策略 | 停止和选优 |
| --- | --- |
| k5 | 生成 5 条完整候选，由 PRM 选优 |
| k10 | 生成 10 条完整候选，由 PRM 选优 |
| full | 用完每题 125,000 个逻辑 token 的预算，由 PRM 选优 |
| adaptive | 前 3 条的提取答案非空且去掉首尾空白后完全相同，就返回首条答案。其余题沿用 full 的预算和 PRM 选优 |

固定 K 组以候选条数为停止条件，记录各自实际消耗。预算组的逻辑 token 同时包含输入与输出。若下一次模型请求的输入加最大允许输出将超出预算，该请求由预算组跳过。已经发出的请求照常计入开销，未完成的候选不进入选优。空提取答案不触发提前停止。正确性标签由作者代码使用的 MATH 评估器在策略选定答案后计算。

## 配对运行

每题用同一条独立采样序列。种子为题号加 10 亿乘以从零开始的尝试序号。四组从该序列取各自实际会生成的前缀，PRM 原始分数也在同一题内复用。在共享运行中，固定 K 需要的候选可能超出预算组停止位置。预算组只计入原停止位置之前的调用。每组报告其独立执行会发出的逻辑调用和 token，双卡墙钟时间记录整个共享运行。提前停止的独立入口用于验证停止后确实没有生成或 PRM 请求。

## 复跑

准备 [MATH-500 复现使用的服务和 Python 环境](../math500-adaptive-revision/README.md#复跑)。两张 H20 各运行一个 vLLM 0.33.0 实例，端口 18420 和 18421，GPU 0 上的 PRM 使用端口 18422。SOURCE 指向包含 upstream、scripts、assets 和 .venv 的基线运行目录。以下命令在 Linux 仓库根目录执行。

~~~bash
exp=experiments/math500-adaptive-k
baseline=experiments/math500-published-code-2026-11-24/scripts
PYTHONPATH=experiments/math500-adaptive-revision python -m unittest discover -v -s "$exp"
bash "$exp/run_dual.sh" "$SOURCE" "$baseline" "$exp/runs/pilot-paired" paired 16
bash "$exp/run_dual.sh" "$SOURCE" "$baseline" "$exp/runs/pilot-adaptive" adaptive 16
bash "$exp/run_dual.sh" "$SOURCE" "$baseline" "$exp/runs/full" paired
~~~

相同配置可复用逐题检查点。结果按零基行号奇偶分为两片，正式轮各 250 题、每卡并发 24。合并时复核数据身份、每组停止点、预算边界、PRM 选优、正确性和样本覆盖。每轮导出分片配置、逐题 JSONL、汇总和预测文件的 SHA-256。逐题记录中的 calls 按请求顺序保存输入 token、实际输出 token、输出上限。各策略记录其用到的前缀、选择、答案和正确性。

正式轮中 GPU 1 先完成分片时，可在另一终端启动以下命令。它等 GPU 1 的原分片成功结束，从 GPU 0 队尾选仍无检查点的题，用同一套代码、模型、种子和预算继续生成。每条检查点写完就合入原分片，已有文件优先保留。

~~~bash
PYTHONPATH="$exp:experiments/math500-adaptive-revision:$baseline:$SOURCE/scripts" \
  "$SOURCE/.venv/bin/python" "$exp/assist.py" "$exp/runs/full"
~~~

## 配对试跑

首次启动时，新入口遗漏了传给生成代理的上下文长度。该次试跑没有完成题目，修复后使用固定的复现服务配置重启。第二次的 16 题分成 4 和 12 题，全部通过合并审计。

| 方法 | 答对 | 生成模型输入及输出 token | PRM 请求 |
| --- | ---: | ---: | ---: |
| 固定 K=5 | 13/16 | 1,254,540 | 80 |
| 固定 K=10 | 13/16 | 2,553,190 | 160 |
| 完整重采样 | 12/16 | 1,979,043 | 192 |
| 自适应 K | 12/16 | 1,099,426 | 57 |

9 题触发早停。正式轮启动时，试跑只剩 GPU 0 的一道长题。该题继续执行直至通过审计。

[配对试跑逐题结果](results/pilot-paired/predictions.jsonl) · [配置和审计摘要](results/pilot-paired/summary.json)
