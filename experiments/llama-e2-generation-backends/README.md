# Llama MATH-500 E2 的生成后端对照

从 MATH-500 的五个难度层按题数比例固定抽取 100 题。难度 1–5 分别抽取 9、18、21、25、27 题。组内按冻结的哈希规则选题，逐题提示 token 和每条候选的种子见归档。两种后端各生成 1,000 条候选。 [冻结样本](results/pilot_manifest.json)

| 指标 | Transformers | vLLM | vLLM − Transformers（百分点） |
|---|---:|---:|---:|
| 首条正确 | 51/100（51.0%） | 50/100（50.0%） | -1.0 |
| 10 条中有正确解 | 77/100（77.0%） | 79/100（79.0%） | +2.0 |
| PRM 选优正确 | 71/100（71.0%） | 62/100（62.0%） | -9.0 |

对照中 Transformers 最终多答对 9 题（Transformers 独自答对 12 题，vLLM 独自答对 3 题）。vLLM 减 Transformers 的配对准确率差为 -9.0 个百分点，分层自助法 95% 区间为 [-16, -2] 个百分点，精确 McNemar 检验 p=0.035。

有正确候选却未选中的题数为 6 题（Transformers）和 17 题（vLLM）。两组分别生成 494 和 495 条正确候选。在历史 vLLM 归档里，同一批 100 题答对 63 题，与本轮 vLLM 的 1,000 条候选中有 261 条自由文本完全相同。 [历史 E2](https://github.com/zhaqian21/adaptive-ttc-lab/pull/20)

扩到 500 题的条件是 vLLM 最终准确率领先至少 5 个百分点且精确配对检验 p<0.05。本轮领先方向为 Transformers，因此保留 100 题范围。论文和公开代码给出了 E2 的 K=10、主要采样参数及入口，尚缺表 1 原跑的候选输出、种子与有效运行配置来定位原结果差异。 [论文](https://arxiv.org/html/2602.01070v5#S4.T1) · [作者复现说明](https://github.com/AhsanBilal7/adaptive-test-time-compute/blob/d9ca5f71f5c7d155153e8f7decd8b1c524968775/README.md#reproducing-the-paper)

## 生成与评分控制

两种后端使用同一份 Llama-3.1-8B-Instruct BF16 权重、自由文本 Direct 提示、原始提示 token、停止 token 128009 和每候选种子。温度为 0.7，top-p 为 0.9，top-k 为 50，最多生成 1,024 token。逐个核对了 100 个提示与固定上游 Direct 入口的 token ID。对应的 1,000 个历史自由文本请求也通过了提示长度、题号和种子排程校验。

本轮生成的是实际送入历史整段 PRM 和判分器的自由文本回答。公开代码中结构化回答由另一次独立调用产生。本轮保留自由文本调用在历史排程中的种子位置。Transformers 使用原生 `generate()`，每行用独立的 PyTorch 多项分布随机数流。vLLM 每个请求传入同一张表中的种子。两个后端各自实现抽样和批处理。 [公开代码 Direct 入口](https://github.com/AhsanBilal7/adaptive-test-time-compute/blob/d9ca5f71f5c7d155153e8f7decd8b1c524968775/src/universal_agent.py#L260)

GPU 1 完成 vLLM 生成后，保留 GPU 0 已完成的 Transformers 候选，余下任务按固定键拆到两张卡并行生成。GPU 0 初始批大小为 16，续跑批大小为 32，GPU 1 续跑批大小为 64。合并时核对首批和两个分片的候选键、内容哈希以及完整覆盖。 [拆分计划](results/pilot/native_plan.json)

两组的所有候选使用同一份 Qwen2.5-Math-PRM-7B 和作者公开代码的整段模板打分。每题按原候选顺序选最高分，用原上游判分器和 MATH-500 参考答案逐一判分。历史整段输入抽查的 60 个 PRM 分数逐值相同，6 题的候选选择全部一致。两组 2,000 条原始回答还用同一判分器独立复核。

Transformers 运行环境为 Transformers 4.48.3 和 PyTorch 2.6.0+cu124。vLLM 服务为 0.33.0，其引擎环境读取的 Transformers 为 5.20.2、PyTorch 为 2.14.0。

## 复跑

从归档重算逐题选择、分层自助法区间和精确配对检验。运行环境需要 Python 与 requests。

~~~bash
exp=experiments/llama-e2-generation-backends
python "$exp/compare_backends.py" --output /tmp/llama-e2-backends-check
~~~

重新生成时，将 `PYTHONPATH` 指向本轮记录的上游检出，使用固定模型、题集以及原环境重建清单。vLLM 入口接受 vLLM 虚拟环境的 Python、模型目录和端口。以下命令用归档清单演示两种后端的完整 100 题运行。断点续跑分配与准确的批次参见归档中的配置和拆分计划。

~~~bash
python "$exp/prepare_sample.py" --dataset "$MATH500" --model "$LLAMA" --output "$MANIFESTS"
~~~

在独立终端启动临时 vLLM 服务。

~~~bash
CUDA_VISIBLE_DEVICES=1 bash "$exp/serve_vllm.sh" "$VLLM_PYTHON" "$LLAMA" 22670
~~~

服务就绪后回到评测终端。

~~~bash
python "$exp/generate_candidates.py" --backend vllm --manifest "$MANIFESTS/pilot_manifest.json" \
  --model "$LLAMA" --output "$OUT/pilot/vllm" --batch 64
CUDA_VISIBLE_DEVICES=0 python "$exp/generate_candidates.py" --backend transformers \
  --manifest "$MANIFESTS/pilot_manifest.json" --model "$LLAMA" --output "$OUT/pilot/transformers" --batch 16
python "$exp/score_candidates.py" --manifest "$MANIFESTS/pilot_manifest.json" --generation-root "$OUT/pilot"
python "$exp/compare_backends.py" --root "$OUT" --manifests "$MANIFESTS" --output /tmp/llama-e2-backends-check
~~~

[E2 逐题对照](results/paired.jsonl) · [汇总及 SHA-256](results/summary.json)
