# MATH-500 作者公开代码流程复现

## 复现范围

在两张 NVIDIA H20 上复现同一批 MATH-500 test 的 E1–E4。原始实现固定为 `d9ca5f71f5c7d155153e8f7decd8b1c524968775`，使用固定版本的 Qwen2.5-7B-Instruct、Qwen2.5-Math-PRM-7B 与 MATH-500 数据。四种方法保留作者代码的控制器、工具、搜索、结果解析和判分。[作者代码](https://github.com/AhsanBilal7/adaptive-test-time-compute/tree/d9ca5f71f5c7d155153e8f7decd8b1c524968775)

- Direct 每个候选分别生成结构化答案和用于判分的带框答案。
- Dynamic 的搜索内部由基础生成模型按原提示打分。E2、E4 保留独立 Qwen Math PRM 的跨候选单段评分和原有选优规则。
- 加速版沿用作者 Transformers 包装器的提示模板、构造的输入 token 和输出解析，将底层生成交给 vLLM 0.33.0 连续批处理。引擎使用 BF16、前缀缓存、32,768 token 上下文、112 条最大活动序列、8,192 token 批处理预算，以及 0.60 的显存使用系数。
- 同一并发试验比较固定题号和采样设置。每题使用固定初始种子，vLLM 的每次请求使用按题内调用次序派生的种子。不同内核、随机数生成方式和批调度可产生不同轨迹。逐题结果写入独立目录并核对原生 CSV、JSON 和选优分数。

## 推理后端试跑

墙钟耗时从题目处理开始计算，模型加载和引擎启动另计。

| 方法 | 题数 | Transformers 串行 | vLLM 连续批处理 | 倍速 | 判定变化 |
|---|---:|---:|---:|---:|---:|
| E1 Direct | 8 | 101.07 秒 | 6.30 秒 | 16.05× | 4/8 |
| E3 Dynamic | 4 | 105.86 秒 | 14.16 秒 | 7.48× | 0/4 |

## 并发压测

每组各并发复用同一份题号列表。表内为单卡墙钟吞吐，按最后一题完成的时间计算。E2 和 E4 的两档在不同的同型号 H20 上执行。E4 的 8 题预检保存了 7 题，随后将设备转给完整的 48 题压测。预检进度未参与并发选择。

| 方法 | 每档题数 | 测得题/分钟 | 正式并发/卡 |
|---|---:|---|---:|
| E1 Direct | 96 | 16 路 184.21，32 路 336.41，64 路 444.34，96 路 366.94 | **64** |
| E2 Direct + PRM Selection | 48 | 24 路 15.96，48 路 21.76 | **48** |
| E3 Dynamic | 96 | 32 路 27.68，64 路 31.98，96 路 33.70 | **96** |
| E4 Dynamic + PRM Selection | 48 | 24 路 2.43，48 路 2.23 | **24** |

E4 的两档均完成全部 48 题。24 路耗时 1,187.44 秒，48 路耗时 1,288.73 秒，24 路整批吞吐高 8.5%。少数题会产生长搜索轨迹，正式轮会在每张卡自己的题目分片里优先提交压测生成 token 数最高的 E3、E4 题。

原始汇总、配对判定差异和正式并发见 [压测记录](performance/selected-profile.json)。

## 正式运行与最终结果

每组评测同一份 MATH-500 test 的 500 题。GPU 0 和 GPU 1 按偶数、奇数题号各执行 250 题，分别从北京时间 15:12:04 和 15:06:46 开始。E4 于 16:24:38 完成最后一题，四组于 16:24:59 前通过最终合并审计。双卡各组均完成分配的题目，整个正式轮无题目重试。

| 方法 | 作者代码流程 + vLLM | 上一轮按论文描述实现 | 论文表 1 |
|---|---:|---:|---:|
| E1 Direct | 344/500（68.8%） | 68.4% | 71.2% |
| E2 Direct + PRM Selection | 395/500（79.0%） | 75.4% | 72.0% |
| E3 Dynamic | 349/500（69.8%） | 67.2% | 74.2% |
| E4 Dynamic + PRM Selection | 397/500（79.4%） | 76.4% | 81.4% |

论文列对应 Qwen2.5-7B-Instruct 的 MATH-500 设置。[论文表 1](https://arxiv.org/html/2602.01070v5#S4.T1)

合并审计核对了每组 500 个唯一题号、固定数据集、原生 JSON 和 CSV、最终选优结果。同步到本地后复核了四组文件的 SHA-256 和逐题答对数。[合并审计](results/audit.json) · [E1 逐题记录](results/E1/predictions.jsonl) · [E2 逐题记录](results/E2/predictions.jsonl) · [E3 逐题记录](results/E3/predictions.jsonl) · [E4 逐题记录](results/E4/predictions.jsonl)

## 复跑步骤

从本目录执行以下命令。评测进程使用 Linux、Python 3.12、uv 和两个可用的 H20。引擎使用独立的 vLLM 0.33.0 环境。两个环境的实测包版本见 [环境记录](config/environments.json)。

```bash
bash scripts/bootstrap.sh
.venv/bin/python scripts/fetch_assets.py
```

在两个终端分别用 GPU 0、GPU 1 启动引擎。将 `VLLM_BIN` 指向独立环境中的 vLLM 可执行文件，并在第二个终端设 `gpu=1`。

```bash
gpu=0
port=$((18420 + gpu))
model="$PWD/assets/model/Qwen--Qwen2.5-7B-Instruct--a09a35458c702b33eeacc393d103063234e8bc28"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)" \
  "$VLLM_BIN" serve "$model" --served-model-name published-qwen25 \
  --host 127.0.0.1 --port "$port" --dtype bfloat16 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.60 --max-model-len 32768 --max-num-seqs 112 \
  --max-num-batched-tokens 8192 --generation-config vllm --enable-prefix-caching
```

在另外两个终端分别跑一份分片。先确认对应服务端口健康，再按压测选出的并发数运行。分片内复用作者代码的逐候选串行流程，日志按题保存。

```bash
set -e
gpu=0
port=$((18420 + gpu))
curl -fsS "http://127.0.0.1:$port/health"
run="$PWD/runs/full"
mkdir -p "$run"
cp scripts/run_published_code.py "$run/run_published_code.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$PWD/scripts"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
for entry in E1:64 E2:48 E3:96 E4:24; do
  IFS=: read -r case_id workers <<< "$entry"
  .venv/bin/python "$run/run_published_code.py" "$case_id" \
    --backend vllm --model published-qwen25 --port "$port" --vllm-version 0.33.0 \
    --workers "$workers" --shards 2 --shard "$gpu" \
    --priority-file performance/selected-profile.json --output "$run/gpu-$gpu/$case_id"
done
```

两卡全部完成后合并审计。

```bash
dataset="$PWD/assets/dataset/HuggingFaceH4--MATH-500--6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl"
.venv/bin/python scripts/audit_published_code.py runs/full "$dataset"
```

压测的固定样本数、题号顺序和逐档汇总见 [15 次压测试验](performance/trials.json)。压测执行脚本按当时运行的字节保留，复跑时把 `PYTHONPATH` 指向 `scripts/`。原始运行的脚本哈希和上游提交见 [源码指纹](config/source-fingerprints.json)。

公开配置将权重位置写成相对路径，将 GPU UUID 转为序号。原始启动参数、正式试验配置和分片汇总分别见 [引擎记录](config/engine.json) 和 [正式执行记录](config/formal-runs.json)。归档中的四份逐题文件与合并审计哈希一致。执行 `sha256sum -c SHA256SUMS` 可校验归档。
