# MATH-500 按需续写和单次作答多采样

两种模型用 [PR #33](https://github.com/zhaqian21/adaptive-ttc-lab/pull/33) 的固定 MATH-500 与 GSM8K 题目、Direct 自由文本提示、原采样参数和数学等价判分。GPU 0 运行 Qwen，GPU 1、7 按题号奇偶分跑 Llama。单卡并发 48，vLLM 0.33.0。三卡先完成按需续写，再沿用同一批服务接续多采样。两项均使用纯生成模型。

## 按需续写

MATH-500 每模型每题生成一份最多 2,048 token 的输出。保存逐 token ID 和结束原因，并对同一输出的两个前缀使用原判分。

- 固定 1,024 使用前 1,024 个 token。
- 固定 2,048 使用完整输出。
- 按需续写只在 1,024 个 token 用完、模型仍可继续且提取不到答案时使用后续输出。短前缀能提取答案时优先保留短前缀。

每个回放规则的生成 token 为该前缀的输出长度加一次输入。实际运行生成到 2,048 的上限，表中报告同一输出流下各规则的回放消耗。输出归档保留短、长各自的判分和文本哈希。汇总同时记录各规则相对固定 1,024 的逐题增减，以及同种子短前缀和 PR #33 旧作答的文本一致数。

## 便宜的多采样基线

GSM8K 和 MATH-500 各 500 题。每题生成 5 条独立的 Direct 候选，每条上限 1,024 token。种子为原题号加上候选从零计数的序号乘 1,000,000。单次作答选择第 1 条，3 条和 5 条方案各用相应前缀投票。按去掉首尾空白后的非空答案计票，同票选首次出现的答案，全空则选第一条。记录每条候选的生成文本和实际 token。

[按需续写结果](results/continuation/table.md) · [多采样结果](results/voting/table.md)

## 复跑

运行目录使用上轮 24 服务器的固定 `source` 模型、上游和 Python 环境。data 包含先前 GSM8K 的固定测试 parquet。GPU 7 的 18427 和 18437 服务已运行时，其配置与 PR #33 一致。服务器托管入口按顺序完成两项；GPU 0 和 1 运行结束后恢复原常驻服务。

~~~bash
exp=experiments/math500-direct-continuation
bash experiments/gsm8k-500-e4-direct/run_24.sh "$RUN_DIRECTORY" "$VLLM_PYTHON" "$exp/run.py"
for study in continuation voting; do
  python "$exp/report.py" --study "$study" --run "$RUN_DIRECTORY/results" \
    --archive "$RUN_DIRECTORY/export/$study" --output "$RUN_DIRECTORY/export/$study"
done
~~~

复算发布结果时使用已有归档。

~~~bash
exp=experiments/math500-direct-continuation
for study in continuation voting; do
  python "$exp/report.py" --study "$study" --archive "$exp/results/$study" --output "/tmp/direct-$study-replay"
done
~~~
