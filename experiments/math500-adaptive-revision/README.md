# MATH-500 的修正与重采样对照

本实验在 [作者公开代码复现的 E4](https://github.com/zhaqian21/adaptive-ttc-lab/pull/2) 上比较三种外层候选策略。E4 历史参考为 397/500，使用固定 10 次独立采样。三个新策略共用同一题的初始候选、E4 的规划和搜索逻辑，以及作者发布代码的 PRM `mean_reward` 选优。

| 策略 | 初始候选生成后如何使用剩余预算 |
| --- | --- |
| `resample` | 继续独立采样 |
| `revise-once` | 每条轨迹执行一次贪心修正，再把该位置分配给新样本 |
| `fixed-point` | 修正有变化时把新版本放回队列，逐 token 相同时把该位置分配给新样本 |

修正借鉴 [Adaptive Replay Refinement](https://arxiv.org/html/2610.28294v1#S3)。修正使用同一个模型和固定提示，截断或提取答案失败时保留最后一个完整版本。PRM 对每条轨迹的最后一个完整版本打分。修正组使用普通 vLLM 贪心解码。逐题标准答案只用于记录正确性，调度器和选优器接收生成结果。

## 实验口径

- 固定作者代码提交 d9ca5f71f5c7d155153e8f7decd8b1c524968775，模型、数据和采样参数沿用基线。两张 H20 各运行一个 vLLM 0.33.0 实例，按题号奇偶分配完整题目。两个分片使用 GPU 0 上的常驻 Qwen2.5-Math-PRM-7B。模型版本、运行环境和解码参数由分片配置和基线记录给出。
- 每题先尝试生成 3 条共享初始轨迹，预算不足时使用已完成的轨迹。每个策略计入这部分开销。独立根的种子为基值 0 加题号加 10 亿乘以从零开始的采样序号。三个分支按题号轮换执行顺序。历史 E4 参考使用自己的种子调度和固定轮数。
- 每个策略的生成预算为 125,000 个逻辑 token，含全部提示和输出。此上限接近历史 E4 在该数据集上的逐题生成 token 中位数 123,882。预算在发请求前预留该请求的完整解码上限，实际计入已处理的提示和生成 token，未能完成的轨迹已发出的请求也计费。即使启用前缀缓存，提示仍计入逻辑 token。PRM 请求数和服务往返时间单独报告。
- 同一题的三个策略在同一工作线程里先后执行以共享初始候选。分策略的逐题时间含共享阶段，反映双卡混合运行的负载与排队。分片总耗时覆盖该分片的三个策略。

## 复跑

在 Linux 节点准备基线指定的固定模型、数据、upstream 检出和 Python 环境。基线的 [环境和服务配置](https://github.com/zhaqian21/adaptive-ttc-lab/tree/9d51d691be6d58c0eb78fef0082d039a24182d97/experiments/math500-published-code-2026-11-24/config) 记录两张卡的 vLLM 参数。vLLM 服务端口为 18420 和 18421，模型名称为 published-qwen25。GPU 0 的 [常驻 PRM 启动脚本](https://github.com/zhaqian21/adaptive-ttc-lab/blob/0858e7436f2ea84bdaa92c03b9b0781d9817ecbe/experiments/aime24-amo-bench-published-code-2026-11-24/scripts/serve_prm.py) 使用端口 18422。

在仓库根目录执行，SOURCE 指向包含 upstream、scripts、assets 和 .venv 的基线运行目录。

~~~bash
exp=experiments/math500-adaptive-revision
baseline=experiments/math500-published-code-2026-11-24
baseline_predictions="$baseline/results/E4/predictions.jsonl"

(cd "$exp" && python -m unittest -v)
bash "$exp/run_dual.sh" "$SOURCE" "$baseline/scripts" "$baseline_predictions" "$exp/runs/pilot" 0.33.0 8
bash "$exp/run_dual.sh" "$SOURCE" "$baseline/scripts" "$baseline_predictions" "$exp/runs/full" 0.33.0
python "$exp/audit_math500.py" "$exp/runs/full" "$baseline_predictions" "$exp/runs/full/export"
~~~

两条运行命令分别请求 8 题试跑和完整 500 题。重复相同命令会校验运行配置并复用已完成题目。合并审计检查两个分片的题号、版本、数据 ID、候选与选优一致性、逐题预算和三个策略的覆盖情况。导出包含逐题答案、候选分数、动作、token 和摘要，摘要包含逐题记录的 SHA-256。比较三种策略时，应同时看准确率、实际使用的生成预算和 PRM 调用数。

## 双卡试跑

8 题全部通过合并审计。该轮验证预算耗尽、修正继续、固定点、重采样和 PRM 选优路径。试跑按固定哈希挑题并沿用正式轮参数。

| 策略 | 正确数 | 生成输入 token | 生成输出 token | PRM 请求 |
| --- | ---: | ---: | ---: | ---: |
| 只重采样 | 7/8 | 880,091 | 108,318 | 92 |
| 固定修正一次 | 7/8 | 876,884 | 111,810 | 87 |
| 固定点调度 | 8/8 | 848,906 | 140,628 | 80 |

[试跑逐题结果](results/pilot/predictions.jsonl) · [试跑配置和审计摘要](results/pilot/summary.json)
