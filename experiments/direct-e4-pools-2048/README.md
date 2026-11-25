# 六数据集的 2,048 token Direct 和 E4 候选池

两个模型使用已有的 GSM8K 500、MATH-500 500、AMC23 40、AIME24 30、AIME 2026 30 和 AMO-Bench 50，共每模型 1,150 题。题目 ID、题面哈希、参考答案及两类种子见 [题目清单](question_plan.json)。数据范围与 [六数据集 1,024 token 对照](https://github.com/zhaqian21/adaptive-ttc-lab/pull/35) 相同。GSM8K 沿用固定的 500 题。

GPU 0 跑 Qwen，GPU 1、7 分跑 Llama。沿用 vLLM 0.33.0、BF16、32,768 上下文、各模型原快照与采样设置，每次生成调用的输出上限改为 2,048 token。Direct 的单卡并发为 48，E4 为 24。GPU 7 复用原生成和 PRM 服务，另两张卡临时使用实验服务并在退出时恢复原部署。

- **Direct 每题 5 条。** 沿用上一轮的自由文本提示、逐题种子和多数投票。保留逐 token ID、原答案判分、每次生成的实际用量以及 1,024 token 前缀，回放 1、3、5 条和单次按需续写。总共 11,500 条候选，Direct 不调用 PRM。
- **E4 每题 10 条。** 沿用公开代码的动态规划、选路和原 PRM 选优。每题生成 10 条完整轨迹并记录原始 PRM 分数。每个已完成候选保存计划、推理步骤、最终作答、工具与计算配置、PRM 打分文本和分数，供重新评分。总共 23,000 条候选。固定 3、5、10 条、五条一致早停和动态 3/5 都从这份池回放。

Direct 的短、长版本共用同一次生成前缀。E4 的 2,048 版本实际重新生成，因为先前阶段的输出可能改变后续规划与选路。按题固定种子基于原来记录的题号。旧运行与本轮各自保留实际用量、逐题判分和版本指纹。

## 运行和复算

运行目录由服务器保存检查点。source 指向已固定上游、模型和 Python 环境的源码目录。data 包含 GSM8K、AIME 2026、AIME24 与 AMO-Bench 的 parquet 和此前的 manifest。GPU 7 已部署此前的 Llama 生成和 PRM 服务。两个阶段依次自动运行，单个 worker 失败时重试一次，成功题目从检查点复用。

~~~bash
exp=experiments/direct-e4-pools-2048
python "$exp/freeze.py"
bash experiments/gsm8k-500-e4-direct/run_24.sh "$RUN_DIRECTORY" "$VLLM_PYTHON" "$exp/run_pools.py"
for phase in direct5 e4; do
  python "$exp/archive_pools.py" --phase "$phase" --run "$RUN_DIRECTORY/results" \
    --archive "$RUN_DIRECTORY/export/$phase" --output "$RUN_DIRECTORY/export/$phase"
done
~~~

每个模型和数据集的候选按 25 题拆分为压缩文件，汇总记录每个文件的 SHA-256。生成 token 计输入和输出，E4 回放按最后一次成功尝试的已使用前缀计算，原始候选池另保留历史重试的请求记录。使用归档复算时传入相位目录作为 `--archive`，省略 `--run`。
