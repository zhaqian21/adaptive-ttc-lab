# 2,048 token E2 接续任务

接续 [E4 十条池](https://github.com/zhaqian21/adaptive-ttc-lab/pull/36)，沿用作者公开代码的 E2 Direct + PRM。每题生成 10 条轨迹，按原 PRM 选优。每次生成调用的输出上限为 2,048 token，沿用固定题目、模型及数学判分。GPU 0 跑 Qwen，GPU 1、7 分跑 Llama，单卡并发 24。归档保留原始轨迹、PRM 打分文本与分数以及逐题用量。

E4 正常完成并恢复常驻服务后，先执行 MATH-500 的两种模型各 500 题。以 **2026 年 11 月 26 日北京时间 10:30** 为接续目标。MATH-500 完成时至少剩 20 分钟才加入 AIME24 两种模型各 30 题。AIME24 完成后至少剩 10 分钟才加入 AMO-Bench 两种模型各 50 题。已经开始的队列阶段按题目清单跑完。服务器记录每次进入或跳过的时间决定。

## 服务器托管

使用与 E4 一致的 source、data、模型和固定上游环境。GPU 7 复用已有 Llama 生成及 PRM。队列通过服务器 tmux 启动，独立于本机 SSH 连接。发生进程级失败时重启一次相应 worker，从已经保存的题目检查点续跑。服务恢复后，程序自动审计并归档已完成的组。

~~~bash
exp=experiments/e2-pool-2048-followup
bash "$exp/queue_24.sh" "$E4_RUN_DIRECTORY" "$E2_RUN_DIRECTORY" "$VLLM_PYTHON"
~~~

服务器根目录的 `queue.status` 和 `queue.exit` 给出接续状态及退出码。每个数据集的工作目录保存逐题检查点。审计完成后写入 `export/` 和 `e2-review.tar`，表格只汇总实际完成的组。重放已生成的归档时，从同一提交执行。

~~~bash
exp=experiments/e2-pool-2048-followup
python "$exp/archive_e2.py" --archive "$E2_RUN_DIRECTORY/export" --output /tmp/e2-2048-replay
~~~
