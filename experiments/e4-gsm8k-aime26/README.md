# GSM8K 和 AIME 2026 的动态 K 对照

GSM8K 使用固定版本的测试集，以种子 42 对 1,319 题无放回抽取 200 题，按原始行号排序。AIME 2026 使用固定版本的全部 30 题，按竞赛题号排序。两种模型共用同一份 [题目清单](questions.json)。清单保存每题原始 ID、题面 SHA-256 和标准答案。[GSM8K](https://huggingface.co/datasets/openai/gsm8k) · [AIME 2026](https://huggingface.co/datasets/MathArena/aime_2026)

24 服务器的两张 H20 各部署一份生成模型和 Qwen2.5-Math-PRM-7B。卡 0 运行 Llama-3.1-8B-Instruct，卡 1 运行 Qwen2.5-7B-Instruct。沿用作者公开 E4 的动态推理、原始 PRM 均值选优、AIME24 使用的数学答案等价判分，以及 [PR #27](https://github.com/zhaqian21/adaptive-ttc-lab/pull/27) 的 vLLM 0.33.0 设置。每卡先跑每数据集两题，通过请求与判分检查后接续剩余题。这四题的候选池直接用于正式结果。

每模型每题只生成一份有序的 10 条候选池。五种规则复用这份候选池的原始答案、得分和每轮实际输入与输出 token。固定 3、5、10 条均用单条最高 PRM 分数选优。五条一致早停在前五条非空答案完全一致时直接返回，否则按固定 10 条选优。动态 3/5 先看前三条，一致时直接返回，否则按固定 5 条选优。一致性比较沿用 [PR #31](https://github.com/zhaqian21/adaptive-ttc-lab/pull/31) 的去首尾空白后字符串匹配。

## 指标

- [对照表](results/table.md) 与 [机器可读汇总](results/summary.json) 都从逐题结果生成。平均候选数按各规则实际使用的前缀长度计算。
- 生成 token 为保留候选各轮成功尝试的输入与输出之和。每次尝试的请求记录在候选池，汇总另列重试和缺少 usage 的请求。
- PRM 请求指外层选优请求。一致早停免除外层评分，其余规则按保留候选数计费。
- “固定 10 对、该规则错”在同一题上比较。最后一列仅统计确实因答案一致触发早停的这类题，动态 3/5 未触发早停时丢掉的题仍计入前一列。JSON 另给出动态 3/5 相对固定 5 条因一致早停丢掉的题数。

## 复跑

从仓库根目录执行。`ROOT` 是独立运行目录，`SOURCE` 是由 [现有 24 服务器 E4 流程](../llama-table1-825/README.md) 准备的源码和模型目录，须包含两个固定生成模型、PRM、上游及 Python 环境。两个具体模型和 PRM 快照见 [Llama 配置](results/llama_config.json) 和 [Qwen 配置](results/qwen_config.json)。

~~~bash
exp=experiments/e4-gsm8k-aime26
mkdir -p "$ROOT/data"
curl -fL 'https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet' -o "$ROOT/data/gsm8k.parquet"
curl -fL 'https://huggingface.co/datasets/MathArena/aime_2026/resolve/d61f486805306e6e6b5913504c07763a99e1b112/data/train-00000-of-00001.parquet' -o "$ROOT/data/aime26.parquet"
ln -s "$SOURCE" "$ROOT/source"
printf '0\n' > "$ROOT/assets.exit"
bash experiments/llama-table1-825/run_24.sh "$ROOT" "$VLLM_PYTHON" "$exp/run.py"
python "$exp/compare.py" --run-root "$ROOT/results" --pools "$ROOT/export" --output "$ROOT/export"
~~~

已有归档时，只需 Python 即可重新生成表格和逐题结果。

~~~bash
exp=experiments/e4-gsm8k-aime26
python "$exp/compare.py" --pools "$exp/results" --output /tmp/e4-gsm8k-aime26-replay
~~~
