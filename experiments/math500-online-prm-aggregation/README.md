# MATH-500 在线答案级 PRM 聚合

将 [PR #8 的离线聚合](https://github.com/zhaqian21/adaptive-ttc-lab/pull/8) 接到 [原自适应 K](https://github.com/zhaqian21/adaptive-ttc-lab/pull/7) 的独立推理入口。前三条候选的提取答案非空且去掉首尾空白后完全一致时，直接返回首条答案。其余题沿用每题 125,000 个生成模型输入与输出 token 的预算。在有完整候选时，用 PRM 对预算内的候选评分。

两轮只改变最终选优规则。原自适应 K 取单条 PRM mean_reward 最高的候选。聚合版使用 PR #8 的分组与平分规则，在同一答案内把已有 mean_reward 相加。两条路径共用同一个采样器、早停规则和 PRM 服务。

## 运行配置

两张 H20 各加载一份 Qwen2.5-7B-Instruct 的 vLLM 0.33.0 服务。GPU 0 常驻 Qwen2.5-Math-PRM-7B。正式轮并发每卡 24，按行号奇偶各执行 250 题，采样种子基值为 0。固定数据、模型版本、源码及解码配置保存在每轮的审计摘要中。使用 [PR #7 的环境和引擎参数](../math500-adaptive-k/README.md#复跑)。

两组依次使用同一套常驻服务实际生成，先跑原自适应 K，再跑聚合版。墙钟时间包括两卡 worker 启动、执行和写逐题文件，合并审计另计。逐题推理时间从开始建立采样器计到选定答案，包含生成、选优和服务排队。两组共享服务和运行顺序保存在此处，耗时对应本次运行。

对照脚本也在两次实际生成的候选池上分别重放两种选优规则，核对实际选择，并报告两轮同题的候选答案、分数和请求记录是否一致。相同候选池上的配对结果用于衡量聚合本身的影响。

## 复跑

在 Linux 仓库根目录执行。SOURCE 指向包含 upstream、scripts、assets 和 .venv 的基线运行目录。

~~~bash
exp=experiments/math500-online-prm-aggregation
base=experiments/math500-published-code-2026-11-24/scripts
PYTHONPATH=experiments/math500-adaptive-k:experiments/math500-adaptive-revision:experiments/math500-prm-answer-aggregation \
  python -m unittest discover -s "$exp" -v
bash "$exp/run_comparison.sh" "$SOURCE" "$base" "$exp/runs/pilot" 16
bash "$exp/run_comparison.sh" "$SOURCE" "$base" "$exp/runs/full"
~~~

各组输出位于其 export 目录，包含分片配置、审计摘要和实际逐题结果。comparison 目录包含两组对照汇总与逐题复核。相同配置重试可复用旧检查点，正式耗时对照使用全新输出目录。
