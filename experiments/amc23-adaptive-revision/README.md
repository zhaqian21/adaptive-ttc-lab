# AMC23 的修正与重采样对照

本实验将 [MATH-500 的三组对照](https://github.com/zhaqian21/adaptive-ttc-lab/pull/4) 扩展到 AMC23。三组使用 Qwen2.5-7B-Instruct 生成候选，用 Qwen2.5-Math-PRM-7B 的 `mean_reward` 选优，并沿用 E4 的单条轨迹搜索逻辑。策略共用同一题最先完成的至多三条初始轨迹。

| 策略 | 初始候选生成后如何使用剩余预算 |
| --- | --- |
| 只重采样（`resample`） | 继续独立采样 |
| 固定修正一次（`revise-once`） | 修正每条轨迹一次，然后分配给新样本 |
| 固定点调度 D（`fixed-point`） | 修正有变化就继续修正，逐 token 相同时重采样 |

数据使用 [math-ai/amc23 的固定 test 分片](https://huggingface.co/datasets/math-ai/amc23/blob/80815d37005feb82cd7f8fbc6901d5d3eff43057/test-00000-of-00001.parquet)，共 40 题。逐题记录保留原始数据 ID，分片使用文件中的零基行号。三组分别按每题最多 125,000 个生成模型逻辑 token 计费，含输入和输出。种子基值为 0，独立采样的种子另加题号和采样序号。PRM 请求单独计数。标准答案用于评分，调度和选优均不读取标准答案。

双卡各运行一份 vLLM 0.33.0 生成服务，以行号奇偶分配题目。两卡共用 GPU 0 的 PRM。试跑每卡按固定哈希挑两题，正式轮每卡跑 20 题，并发 16。三个策略在同一个题目的工作线程中轮换执行顺序。

## 复跑

准备 [MATH-500 对照使用的模型、作者代码、Python 环境和服务](../math500-adaptive-revision/README.md#复跑)。两卡生成端口分别为 18420 和 18421，PRM 端口为 18422。在 Linux 中从仓库根目录执行，`SOURCE` 指向包含 upstream、scripts、assets 和 .venv 的基线运行目录。

~~~bash
exp=experiments/amc23-adaptive-revision
baseline=experiments/math500-published-code-2026-11-24
revision=80815d37005feb82cd7f8fbc6901d5d3eff43057
mkdir -p "$exp/runs"
parquet="$exp/runs/amc23.parquet"
curl --fail --location \
  "https://huggingface.co/datasets/math-ai/amc23/resolve/$revision/test-00000-of-00001.parquet" \
  --output "$parquet"
bash "$exp/run_dual.sh" "$SOURCE" "$baseline/scripts" "$parquet" "$exp/runs/pilot" 2
bash "$exp/run_dual.sh" "$SOURCE" "$baseline/scripts" "$parquet" "$exp/runs/full"
~~~

运行前校验 Parquet 的 SHA-256。同一轮用相同参数重试时复用完整检查点。两条命令分别在输出目录下生成 `export/predictions.jsonl` 和 `export/summary.json`。合并审计核对两卡的版本、配置、题号、实际 token、评分和 PRM 选优，并按题号排序。摘要记录预测文件的 SHA-256。

`candidate_pool_correct` 计最终提交给 PRM 的候选池中至少有一个正确解的题数。`correct` 计 PRM 最终选对的题数。`prm_missed_correct` 计候选池有正确解但 PRM 选错的题数。三项均以当轮全部题目为统计范围。

## 双卡试跑

4 题均通过合并审计。三个策略的候选池覆盖全部 4 题，PRM 在只重采样组漏选 1 题。两卡中耗时较长的分片用时 274 秒。

[试跑逐题结果](results/pilot/predictions.jsonl) · [试跑配置和审计摘要](results/pilot/summary.json)
