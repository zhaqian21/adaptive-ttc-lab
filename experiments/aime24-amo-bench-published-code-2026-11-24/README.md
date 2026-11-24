# Qwen2.5-7B-Instruct 的 AIME24 与 AMO-Bench 复现

本轮使用作者公开代码提交 `d9ca5f71f5c7d155153e8f7decd8b1c524968775`，沿用原有的规划、工具选择、搜索、结构化答案生成、带框答案生成与判分流程。两张 NVIDIA H20 分别运行一个 vLLM 0.33.0 引擎，使用 BF16 和多题连续批处理。同一题内候选按作者代码顺序执行。E2、E4 沿用作者的单段轨迹评分和 `mean_reward` 选优，PRM 由同卡常驻评分进程提供。四组共同使用 temperature 0.7、top-p 0.9、top-k 20、重复惩罚 1.05 和最多 1,024 个新 token。生成请求使用基值 0 按题号和题内请求次序派生种子。[作者代码](https://github.com/AhsanBilal7/adaptive-test-time-compute/tree/d9ca5f71f5c7d155153e8f7decd8b1c524968775)

## 数据与判分

| 数据集 | 分片 | 题数 | 固定数据版本 |
|---|---|---:|---|
| AIME24 | train | 30 | `8d88b2876a82a080e2f172cc9b25d0d9d2cb4792` |
| AMO-Bench | test | 50 | `2f422616c25d862984408fbbfaed63a961e8e025` |

两个数据集用作者的加载器读取固定版 Parquet，判分沿用公开代码主评测入口实际使用的 `MATHEvaluator`。将标准答案作为带框输出进行自检，AIME24 通过 30/30，AMO-Bench 通过 41/50。AMO-Bench 的题号 2、24、25、26、27、40、41、42、46 在答案提取后未判等。下表保留公开代码口径。[固定数据清单](config/datasets.json) · [逐题判分自检](results/grading-control.json)

## 双卡压测

每个并发对照使用相同的固定题号和采样设置。E1–E3 每档 16 题，E4 每档 12 题。耗时为单卡端到端处理时间，模型服务已预热。

| 设定 | 比较档位 | AIME24 耗时 | AMO-Bench 耗时 | 两个数据集胜出的并发 |
|---|---:|---:|---:|---:|
| E1 | 8 / 16 | 12.8 / 7.2 秒 | 13.5 / 8.2 秒 | 16 |
| E2 | 8 / 16 | 144.3 / 88.3 秒 | 161.6 / 94.3 秒 | 16 |
| E3 | 8 / 16 | 218.4 / 207.5 秒 | 253.9 / 222.7 秒 | 16 |
| E4 | 6 / 12 | 1,386.5 / 1,286.4 秒 | 1,735.6 / 1,267.8 秒 | 12 |

正式轮按题号奇偶分片。AIME24 每卡 15 题，E1–E3 实际并发为 15，E4 为 12。AMO-Bench 每卡 25 题，E1–E3 并发为 16，E4 为 12。E3、E4 将压测中生成量最高的一批题提前提交。[AIME24 压测记录](performance/aime24-profile.json) · [AMO-Bench 压测记录](performance/amo-profile.json)

## 正式结果

北京时间 18:43:25 启动正式轮，19:07:22 转入 AMO-Bench，19:50:54 完成合并审计。正式轮耗时 1 小时 7 分 29 秒。两套各四组共 320 条逐题记录已校验，正式题目无重试。

| 设定 | AIME24 复现 | AIME24 论文 | AMO-Bench 复现 | AMO-Bench 论文 |
|---|---:|---:|---:|---:|
| E1 Direct | 2/30（6.67%） | 6.67% | 1/50（2.0%） | 2.0% |
| E2 Direct + PRM | 3/30（10.0%） | 6.67% | 3/50（6.0%） | 2.0% |
| E3 Dynamic | 3/30（10.0%） | 10.0% | 1/50（2.0%） | 4.0% |
| E4 Dynamic + PRM | 4/30（13.33%） | 13.3% | 3/50（6.0%） | 4.0% |

AIME24 的 E4 比 E2 多答对 1 题。AMO-Bench 的 E2 与 E4 各答对 3 题。[论文表 1](https://arxiv.org/html/2602.01070v5#S4.T1) · [两套数据集合并审计](results/audit.json)

[AIME24 逐题结果与审计](results/aime24/) · [AMO-Bench 逐题结果与审计](results/amo/)

## 运行配置与复跑

[正式执行记录](config/formal-runs.json)保存每个数据集、设定和 GPU 分片的参数、题号、提交顺序与运行汇总。[16 次压测记录](performance/trials.json)保存同题对照的配置和耗时。[模型服务配置](config/services.json)保存生成引擎和 PRM 的启动设置，权重位置改为相对路径，GPU UUID 改为设备序号。[源码指纹](config/source-fingerprints.json)记录了原始协调脚本、评测入口和评分服务的哈希。

复跑需要 [MATH-500 复现归档](https://github.com/zhaqian21/adaptive-ttc-lab/tree/9d51d691be6d58c0eb78fef0082d039a24182d97/experiments/math500-published-code-2026-11-24)中的固定上游代码、评测环境与模型权重。按该归档的启动步骤开两路 vLLM。根据本目录的 [数据清单](config/datasets.json)下载两个固定版本的 Parquet，核对 SHA-256，依次保存为 `datasets/aime24.parquet` 和 `datasets/amo.parquet`。将清单复制到 `datasets/manifest.json`。

从本目录运行。在每张卡各启动一个 PRM 评分进程，GPU 0、GPU 1 依次使用端口 18422、18423。将 `MATH500_DIR` 设为已完成依赖安装和权重准备的 MATH-500 归档路径。[PRM 服务实现](scripts/serve_prm.py)

```bash
gpu=0
export PYTHONPATH="$PWD/scripts:$MATH500_DIR/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
"$MATH500_DIR/.venv/bin/python" -c \
  'import serve_prm, sys; serve_prm.ADDRESS = ("127.0.0.1", 18422 + int(sys.argv[1])); serve_prm.serve()' \
  "$gpu"
```

以下是 AIME24 E4 的 GPU 0 分片示例。在另一终端设 `gpu=1` 执行另一分片。AMO-Bench 取 `dataset=amo`。将 `case_id`、`workers` 按上表的正式配置设置。

```bash
gpu=0
dataset=aime24
case_id=E4
workers=12
export PYTHONPATH="$PWD/scripts:$MATH500_DIR/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
"$MATH500_DIR/.venv/bin/python" scripts/run_published_extensions.py "$case_id" \
  --dataset "$dataset" --dataset-manifest datasets/manifest.json \
  --backend vllm --vllm-version 0.33.0 --model published-qwen25 \
  --port "$((18420 + gpu))" --prm-port "$((18422 + gpu))" \
  --workers "$workers" --shards 2 --shard "$gpu" \
  --priority-file "performance/$dataset-profile.json" \
  --output "runs/$dataset/gpu-$gpu/$case_id"
```

归档的八份逐题结果与服务器合并审计中的 SHA-256 一致。使用 `sha256sum -c SHA256SUMS` 校验归档文件。
