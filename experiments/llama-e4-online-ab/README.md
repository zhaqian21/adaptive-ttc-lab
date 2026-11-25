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
python experiments/llama-e4-online-ab/compare_online.py --output /tmp/llama-e4-online-ab
~~~

生成 token 使用 vLLM 响应中的输入与输出用量。请求计数包含失败和重试。有请求缺少用量时单独计数。外层 PRM 请求按实际发出的 HTTP 调用统计。双卡墙钟从启动评测 worker 到两张卡都完成，模型服务加载时间另计。逐题 P95 从题目进入 worker 到写出候选和答案前，包含题内重试，线程池排队时间另计。
