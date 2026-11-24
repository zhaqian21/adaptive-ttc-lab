# Llama 的 12 组公开代码复现

使用 Llama-3.1-8B-Instruct 和 Qwen2.5-Math-PRM-7B，复用 [MATH-500](https://github.com/zhaqian21/adaptive-ttc-lab/pull/2) 与 [AIME24、AMO-Bench](https://github.com/zhaqian21/adaptive-ttc-lab/pull/3) 的作者公开代码流程。上游固定为 `d9ca5f71f5c7d155153e8f7decd8b1c524968775`。生成端沿用公开代码的对话模板、采样和输出解析，vLLM 处理完全相同的 prompt token ID。两个 H20 各部署一份 BF16 生成模型和 PRM，按题号奇偶分片。

Llama 使用 ModelScope 上的固定 BF16 快照 `359efdbb8af05b788a4ad4185215c6b8caa9052c`。运行前校验模型文件和三套数据集，资产指纹见 [资产清单](assets.json)。每个数据集依次执行 E1–E4，候选轮数分别为 1、10、1、10。采样种子基值为 0。先用 8 题批次验证 MATH-500 的 E1 和三个数据集的 E4，再跑全部 2,320 条方法与题目组合。每组结束后核对候选、评分、答案和两个分片的题号，并写入对照 [论文表 1](https://arxiv.org/html/2602.01070v5#S4.T1) 的逐题结果。

## 24 号双卡接续

使用相同的 Llama 评测入口和固定资产。prepare_24.sh 在独立目录准备模型权重，复用已验证的三套数据、PRM 权重和 Python 环境。这个阶段不加载 GPU 模型。正式运行前确认 GPU 0、1 上的 Qwen 生成服务已空闲，临时让出设备。Llama 运行结束后按原启动命令恢复 Qwen 生成和 PRM 服务。

~~~bash
llama=experiments/llama-table1-825
bash "$llama/prepare_24.sh" "$BASELINE_SCRIPTS" "$QWEN_SOURCE" "$AIME_AMO_DATA_DIR" "$OUTPUT"
bash "$llama/run_24.sh" "$OUTPUT" "$VLLM_PYTHON"
~~~

该入口适用本项目已部署的三个常驻服务。仍使用两份 Llama 和两份 PRM，E1–E4 的参数与 825 版相同。结果与服务恢复状态分开记录。

## 复跑

下载 [MATH-500 公开代码归档的固定版本](https://github.com/zhaqian21/adaptive-ttc-lab/tree/9d51d691be6d58c0eb78fef0082d039a24182d97/experiments/math500-published-code-2026-11-24) 后，在当前目录执行以下命令。`BASELINE` 指向归档下包含 `environment.lock.txt` 和 `scripts` 的目录。

~~~bash
python make_job.py "$BASELINE" local/llama-table1-start.sh
~~~

在 825 上传生成的脚本，选择常规作业并申请 **2 张 H20、24 核 CPU、128 GiB 内存、6 小时时限**。用下面的命令启动，其中 `LAUNCHER` 为上传后的脚本绝对路径，`RUN_DIRECTORY` 是新的专用运行目录。

~~~bash
exec bash "$LAUNCHER" "$RUN_DIRECTORY"
~~~

运行目录保存 `preparation.status`、模型服务和每组的日志。`results/status.json` 给出当前评测组，`results/export/summary.json` 和各组的审计文件给出完整结果。汇总指标注明样本数和逐题输出的 SHA-256。
