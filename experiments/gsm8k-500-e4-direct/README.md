# GSM8K 固定 500 题的直接作答与 E4 对照

保留 [PR #32](https://github.com/zhaqian21/adaptive-ttc-lab/pull/32) 的 200 题和两个模型的候选池。按同一个随机数生成器的后续状态，从余下 1,119 道测试题中无放回抽 300 题。两批各自按数据集原始行号排序，题目清单固定在 [questions.json](questions.json)。模型使用与 PR #32 相同的 Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct 和 Qwen2.5-Math-PRM-7B 快照。

每个模型对全部 500 题单次直接作答。提示词使用上游 Direct 分支中用于正式判分的自由文本提示词，每题只请求一次成功生成，沿用 E4 的采样参数和数学等价判分。直接作答没有规划、工具调用、候选重排和 PRM。新增 300 题每题补齐原始 E4 的 10 条候选。固定 3、5、10 条、五条一致早停、动态 3/5 共用每题同一批 E4 候选，保留原 PRM 选优。

GPU 0 运行 Qwen。GPU 1 和 GPU 7 分别运行 Llama 的偶数和奇数题号，分别提供 250 道直接作答和 150 道新增 E4。生成采用 vLLM 0.33.0、BF16 和 32,768 token 上下文上限。各卡单次作答并发 48，E4 并发 24。每卡先跑直接作答，再补 E4。GPU 7 使用已部署的 Llama 和 PRM，GPU 0 和 1 在实验结束后恢复原来的常驻服务。

[结果表](results/table.md)中的生成 token 是输入和输出之和，E4 开销按同池回放统计。PRM 请求仅计外层选优。最后两列分别统计固定 10 条答对而所列规则答错的题，以及其中在一致早停分支丢掉的题。逐题结果和两批题的分开统计均在 [结果目录](results)。

单次作答的 Qwen 为 468/500，Llama 为 434/500。五条一致早停时，Qwen 从 481 到 480，token 少 38.5%。Llama 与固定 10 条同为 474 题，token 少 20.9%，逐题有两题从对变错、两题从错变对。动态 3/5 的两种模型答对数都与固定 5 条持平，Qwen 和 Llama 分别节省 32.4% 和 21.2% 的生成 token。

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

## 后续五个数据集

同一直接作答口径接续 MATH-500、AMC23、AIME24、AMO-Bench 和 AIME 2026。每个模型共 650 题，题目 ID 已和 [PR #31](https://github.com/zhaqian21/adaptive-ttc-lab/pull/31)、[PR #32](https://github.com/zhaqian21/adaptive-ttc-lab/pull/32) 的两种模型结果核对一致。数据资产与题目顺序见 [后续清单](followup_questions.json)。三卡分工沿用 GSM8K，直接作答流程无需加载临时 PRM。GPU 7 复用现有生成服务。

[六个数据集的单次作答正确率](followup-results/table.md)和[逐题记录](followup-results)采用同一判分口径。

后续运行目录同样包含 `source` 和 `data`。MATH-500 及 AMC23 从已准备的 source 加载。data 包含固定 AIME24/AMO-Bench manifest 及其两个 parquet，并复用 PR #32 的 AIME 2026、GSM8K parquet。队列等待 GSM8K E4 正常结束且常驻服务恢复，然后自动接续。

~~~bash
exp=experiments/gsm8k-500-e4-direct
bash "$exp/queue_24.sh" "$RUN_DIRECTORY" "$FOLLOWUP_DIRECTORY" "$VLLM_PYTHON"
python "$exp/followup_report.py" --run "$FOLLOWUP_DIRECTORY/results" \
  --gsm8k-summary "$RUN_DIRECTORY/export/summary.json" --archive "$FOLLOWUP_DIRECTORY/export" \
  --output "$FOLLOWUP_DIRECTORY/export"
~~~

离线汇总时使用归档结果和 GSM8K 汇总。

~~~bash
exp=experiments/gsm8k-500-e4-direct
python "$exp/followup_report.py" --gsm8k-summary "$exp/results/summary.json" \
  --archive "$exp/followup-results" --output /tmp/direct-previous-benchmarks-replay
~~~
