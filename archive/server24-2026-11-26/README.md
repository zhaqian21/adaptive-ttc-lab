# 24 服务器增量归档

归档日期为北京时间 2026 年 11 月 26 日。归档收录旧实验的新增候选文本与逐步评分、早期压测明细、断点恢复记录及实际运行脚本。[历史 PR 索引](prior-prs.json) 按 PR 记录分支、提交 SHA 和依赖关系。各 PR 的完整数据继续通过对应分支或提交取回。

## 已发布的实验

- [PR #1](https://github.com/zhaqian21/adaptive-ttc-lab/pull/1) 保存论文口径的 Qwen MATH-500 E1–E4 配置、预测和审计。
- [PR #2](https://github.com/zhaqian21/adaptive-ttc-lab/pull/2)、[PR #3](https://github.com/zhaqian21/adaptive-ttc-lab/pull/3) 和 [PR #20](https://github.com/zhaqian21/adaptive-ttc-lab/pull/20) 保存两模型按作者公开代码复现的正式预测和审计。
- PR #4–#31 的优化、早停、后端和 PRM 实验入口见 [历史 PR 索引](prior-prs.json)。Llama MATH-500 E2、E4 的候选步骤在 [PR #21](https://github.com/zhaqian21/adaptive-ttc-lab/pull/21)。Qwen MATH-500 历史候选的答案、排序及成本在 [PR #14](https://github.com/zhaqian21/adaptive-ttc-lab/pull/14) 和 [PR #19](https://github.com/zhaqian21/adaptive-ttc-lab/pull/19)。
- PR #32–#41 归档了其余候选池、低成本 Direct 和两模型 2048 token 在线对照。最近的正式在线结果见 [Qwen MATH-500](https://github.com/zhaqian21/adaptive-ttc-lab/pull/39)、[Qwen AMC23 和 GSM8K](https://github.com/zhaqian21/adaptive-ttc-lab/pull/40) 及 [Llama 三数据集](https://github.com/zhaqian21/adaptive-ttc-lab/pull/41)。

## 新增实验材料

[候选轨迹清单](trace_manifest.json) 关联每批记录的原结果 PR。这里保存论文口径与公开代码口径 Qwen MATH-500 E2、E4，以及两模型公开代码口径 AIME24、AMO-Bench E2、E4 的新增候选信息。每行保留题号、数据标识、原始结果指纹和有序候选。正式预测、原有简化候选池和既有逐题评分通过上述 PR 访问。

[早期压测明细](records/early-throughput-trials.json) 包含逐试次配置、预测或耗时；松散的服务器脚本和恢复记录保存在 [server-files](server-files)。[来源清单](records/source-provenance.json) 记录原相对路径与 SHA-256。历史脚本中的服务器路径规范为通用目录，需要按新环境设置目录和 GPU UUID。[常驻服务启动脚本](launch_resident.sh) 根据实际生成与评分参数整理，适用于端口空闲且模型已下载的服务器。

## 恢复环境

上游公开代码是 [adaptive-test-time-compute 固定提交](https://github.com/AhsanBilal7/adaptive-test-time-compute/tree/d9ca5f71f5c7d155153e8f7decd8b1c524968775)。后期在线组使用 H20 和 vLLM 0.33.0。环境约束、数据 ID 与校验值见各自 PR，六数据集固定题目清单见 [PR #36](https://github.com/zhaqian21/adaptive-ttc-lab/pull/36)。

| 资产 | Hugging Face 仓库 | Revision |
|---|---|---|
| Qwen | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 |
| Llama | meta-llama/Llama-3.1-8B-Instruct | 359efdbb8af05b788a4ad4185215c6b8caa9052c |
| Math PRM | Qwen/Qwen2.5-Math-PRM-7B | 0610740060112df12585d00a1c5f4624d2f59051 |

本目录使用模型仓库与 revision 描述权重。数据覆盖和版本按试验 PR 的清单恢复。825 产物不在本次服务器 24 归档范围内。

在下载的归档目录执行校验：

~~~bash
python verify_archive.py
~~~
