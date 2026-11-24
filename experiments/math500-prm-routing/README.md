# MATH-500 的 PRM 验证与预算路由消融

在 [原自适应 K 与答案级聚合的在线对照](https://github.com/zhaqian21/adaptive-ttc-lab/pull/9) 后接续运行。统一使用种子 0、MATH-500 和每题 125,000 个生成模型输入与输出 token。两卡按题号奇偶分片，正式轮每卡并发 24。复用同一套 E4 采样、模型、引擎和 GPU 0 的 PRM 服务。

| 阶段 | 比较 |
| --- | --- |
| factorial | 同一条预算内完整候选序列，比较单候选 PRM、答案级累加，以及这两种选优分别加前三条一致早停的结果。 |
| g | 早停与答案级累加基础上，按答案组候选数从多到少评分。已评分的第一名总分严格高过剩余组总分的上界时跳过后续 PRM。等分时继续评分，沿用原代表候选的平分规则。 |
| f | 保留前三条一致早停。第 5、7 条候选后用已有 PRM 分数检查是否停止。领先答案至少支持 2/3 的全部候选，且总分领先第二答案至少 1.0 时停止。其余继续生成到预算上限。 |
| fg | 使用 F 的停采规则，并在最终选优时使用 G。F 阶段取得的分数继续使用，每条候选至多请求一次 PRM。 |
| matched | 在 factorial 的同一候选池上重放 G、F、F+G。给 G 按逐题相同的 PRM 请求数做生成顺序对照，给 F 用接近其总逻辑 token 的固定预算做对照。 |

PRM 单候选分数落在 [0, 1]。G 以组大小为尚未评分组的总分上界。所有可用答案组总分为零时沿用原 PRM 回退。

## 复跑

准备 [PR #9 使用的服务和 Python 环境](../math500-online-prm-aggregation/README.md#运行配置)。两张卡的生成服务端口为 18420 和 18421，PRM 端口为 18422。SOURCE 指向包含 upstream、scripts、assets 和 .venv 的同一运行目录。

~~~bash
exp=experiments/math500-prm-routing
baseline=experiments/math500-published-code-2026-11-24/scripts
export PYTHONPATH="experiments/math500-adaptive-k:experiments/math500-adaptive-revision:experiments/math500-prm-answer-aggregation"
python -m unittest discover -s "$exp" -v
for method in factorial g f fg; do
    bash "$exp/run_dual.sh" "$SOURCE" "$baseline" "$exp/runs/smoke/$method" "$method" 8
    bash "$exp/run_dual.sh" "$SOURCE" "$baseline" "$exp/runs/full/$method" "$method"
done
python "$exp/matched.py" "$exp/runs/full"
~~~

逐题文件包含实际请求记录、预算边界、曾送评的候选编号和具体停止决定。审计在已保存的调用记录上复核每一步，并重新判分。factorial 共享候选生成和评分，各策略的开销从各自的停止点计算。g、f、fg 的墙钟时间分别由两卡真实在线运行取得。matched 不发出模型请求。

run_queue.sh 可串行执行上述顺序。它等待在线对照的成功审计，随后逐阶段试跑、执行 500 题并验证。全部模块结果写出后按带时区的时间参数判断是否进入另行提供的 Llama 任务入口。
