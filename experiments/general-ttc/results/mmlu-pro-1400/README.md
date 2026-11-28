# MMLU-Pro 1,400 题的 E1–E4 对照

2026 年 11 月 28 日，单张 AutoDL RTX 4090（约 48 GiB 显存）实测。每个组合使用同一份冻结题单，共 14 个领域、每领域 100 题。种子为 0，单次生成输出上限为 1,024 token，生成模型和评分模型均使用 BF16。

每格为 **正确率 / 总 token（万）**。总 token 为生成输入、生成输出与评分输入的实测计数之和。E1 单次直接作答。E2 直接采样十条并用 PRM 选优。E3 动态选策略，外层预算为一条，搜索分支可调用 PRM。E4 逐轮动态选策略，外层预算为十条，再用 PRM 选优。

| 组合 | E1 直接作答 | E2 十条采样 + PRM | E3 动态策略 | E4 动态策略 + 十条采样 + PRM |
|---|---:|---:|---:|---:|
| Qwen + VersaPRM | 51.21% / 67.75 | 61.93% / 1,293.19 | 53.29% / 130.10 | 62.07% / 1,975.35 |
| Llama + VersaPRM | 45.00% / 89.64 | 61.64% / 1,723.44 | 46.50% / 146.58 | 60.50% / 2,284.36 |
| Qwen + dPRM | 51.86% / 67.84 | 63.07% / 1,324.66 | 53.57% / 130.21 | 63.21% / 2,006.56 |
| Llama + dPRM | 45.86% / 89.66 | 63.14% / 1,771.07 | 43.93% / 146.70 | 62.93% / 2,328.33 |

同一组合从 E1 到 E2 均提高 **10.71–17.29 个百分点**。从 E2 到 E4，两组 Qwen 各多答对 2 题，两组 Llama 分别少答对 16 题和 3 题，总 token 增加 31.5%–52.8%。Llama + dPRM 的 E2 答对 884 题，E4 答对 881 题，后者多用 557 万 token。

## 实验记录

- [全量汇总](summary.csv) 和 [分领域汇总](by-category.csv) 保存精确题数、正确率、生成输入和输出 token、PRM 输入 token、请求数及可用的计时指标。
- [逐题归档](runs/) 按组合和 E1–E4 保存 22,400 行 JSONL 原始记录，包含候选、控制器决定、PRM 步分、最终选择和 token。每轮保留配置、服务报告中的评分配置和运行阶段。Qwen + VersaPRM 的 E1 由先跑的 500 题和追加的 900 题合并，计时为两段实测时间之和。
- [冻结题单](mmlu-pro-1400.json.gz) 解压后 SHA-256 为 `9246bdbb6f285f89babb74cbefa6cd3f8e443f79bf798130c867ab2d4b509dd0`。保留了实际运行时的文档字节与顺序。题目来自 [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) 经 [VersaPRM 固定提交](https://github.com/UW-Madison-Lee-Lab/VersaPRM/blob/81a3c993eb80c38ee3900405bee67405dd79f610/synth_cot_generation/data/mmlu_eval_questions.json) 发布的评测题单。上游 VersaPRM 仓库的许可声明随归档保存于 [provenance](provenance/)。
- [原始执行记录](provenance/) 保存模型与权重分片校验值、队列脚本、试跑、逐题进度日志及恢复记录。Qwen + VersaPRM 的 E2、E4 从 48 调到 96 并发，Llama + VersaPRM 的 E4 从 96 调到 112 并发并保留前 158 题。断点续跑的耗时包含中断和恢复所占时间。
- Qwen + dPRM 的 E2 做过短时并发验证，96 并发两次取样分别为 340.4 和 341.9 次评分/分钟，额外加压到约 112 并发时为 348.0 次/分钟。正式轮采用 96 并发，探测请求独立于逐题结果。

## 校验与复现

在本目录执行以下命令。校验脚本只使用 Python 标准库，逐题重算 16 轮结果和 224 条分领域汇总。

```bash
sha256sum -c SHA256SUMS
python verify.py
```

生成端是 vLLM 0.33.0，使用 Qwen2.5-7B-Instruct 或历史版本的 Llama-3.1-8B-Instruct。服务使用单卡、BF16、32,768 最大上下文、112 最大同时序列、8,192 最大批处理 token、0.60 显存比例和前缀缓存。各轮客户端并发记录在对应运行目录中。

VersaPRM 使用 [原评分服务](../../serve_prm.py) 和 [依赖](../../requirements-prm.txt)。dPRM-qwen 将 [发布的 LoRA](https://huggingface.co/dongboklee/dPRM-qwen) 加载到固定的 Qwen3-8B 基座。按 [下载验收清单](provenance/models/dprm.json) 中的目录、版本和 SHA-256 准备权重，并把该清单复制到模型根目录的 `downloads/dprm-ready.json`。使用 [独立评分依赖](../../requirements-dprm.txt) 启动 [dPRM 服务](../../serve_dprm.py) 的 `--verify`，比较步分 logits 和全序列 logits。评分标记和标签取自 [Multi-RM 固定提交](https://github.com/db-Lee/Multi-RM/blob/65b3e8e17a91a8a85807cb85035bddfd256db861/discriminative/reward_model.py)。

运行 [通用框架](../../README.md) 时解压归档题单，给对应的生成与评分端点，沿用各轮配置的模型名、并发及预算。E1 基线每个组合单独运行。评分前的生成结果会随连续批处理变化，表格记录的是实际观察到的单轮结果。
