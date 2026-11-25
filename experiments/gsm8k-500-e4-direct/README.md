# GSM8K 固定 500 题的直接作答与 E4 对照

保留 [PR #32](https://github.com/zhaqian21/adaptive-ttc-lab/pull/32) 的 200 题和两个模型的候选池。按同一个随机数生成器的后续状态，从余下 1,119 道测试题中无放回抽 300 题。两批各自按数据集原始行号排序，题目清单固定在 [questions.json](questions.json)。模型使用与 PR #32 相同的 Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct 和 Qwen2.5-Math-PRM-7B 快照。

每个模型对全部 500 题单次直接作答。提示词使用上游 Direct 分支中用于正式判分的自由文本提示词，每题只请求一次成功生成，沿用 E4 的采样参数和数学等价判分。直接作答没有规划、工具调用、候选重排和 PRM。新增 300 题每题补齐原始 E4 的 10 条候选。固定 3、5、10 条、五条一致早停、动态 3/5 共用每题同一批 E4 候选，保留原 PRM 选优。

GPU 0 运行 Qwen。GPU 1 和 GPU 7 分别运行 Llama 的偶数和奇数题号，分别提供 250 道直接作答和 150 道新增 E4。生成采用 vLLM 0.33.0、BF16 和 32,768 token 上下文上限。各卡单次作答并发 48，E4 并发 24。每卡先跑直接作答，再补 E4。GPU 7 使用已部署的 Llama 和 PRM，GPU 0 和 1 在实验结束后恢复原来的常驻服务。

[结果表](results/table.md)中的生成 token 是输入和输出之和，E4 开销按同池回放统计。PRM 请求仅计外层选优。最后两列分别统计固定 10 条答对而所列规则答错的题，以及其中在一致早停分支丢掉的题。逐题结果和两批题的分开统计均在 [结果目录](results)。

## 复跑

从仓库根目录执行。运行目录复用 PR #32 的固定源码、Python 环境和模型资产目录，`data` 指向包含同一 GSM8K parquet 的目录。24 服务器上 GPU 7 的 Llama 和 PRM 服务应在 18427、18437 端口就绪。脚本只在现有生成服务队列为空时切换 GPU 0 和 1。

~~~bash
exp=experiments/gsm8k-500-e4-direct
python "$exp/sample.py" "$RUN_DIRECTORY/data" > /tmp/gsm8k-500-questions.json
cmp "$exp/questions.json" /tmp/gsm8k-500-questions.json
bash "$exp/run_24.sh" "$RUN_DIRECTORY" "$VLLM_PYTHON"
python "$exp/report.py" --run "$RUN_DIRECTORY/results" --pools "$RUN_DIRECTORY/export" --output "$RUN_DIRECTORY/export"
~~~

已有归档时，从 PR #32 的旧池和本实验的新增池重建逐题结果和表格。

~~~bash
exp=experiments/gsm8k-500-e4-direct
python "$exp/report.py" --pools "$exp/results" --output /tmp/gsm8k-500-e4-direct-replay
~~~
