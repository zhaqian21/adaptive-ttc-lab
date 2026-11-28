# 跨学科文本推理的 E1–E4

沿用数学版的动态策略选择与轨迹评估。Qwen2.5-7B-Instruct 生成推理步骤，VersaPRM 在搜索中给分支打分，在搜索结束后按轨迹各步平均分选优。

| 组别 | 外层轮数 | 推理与选优 |
| --- | ---: | --- |
| E1 | 1 | 单次直接生成 |
| E2 | 10 | 独立直接生成，按完整轨迹的 PRM 分数选优 |
| E3 | 1 | 根据题目选直接生成、Best-of-N、Beam Search 或 Lookahead |
| E4 | 10 | 逐轮动态选策略，再按完整轨迹的 PRM 分数选优 |

通用版用一次结构化调用完成规划、工具和策略选择。领域提示包含拆解、比较选项和核对约束。答案抽取使用明确的选项字母。E3 和 E4 选择搜索时调用 VersaPRM，E4 复用搜索阶段已打分的完整轨迹。默认外层轮数与论文一致，`--k 2` 可用于试跑。首轮数据使用 VersaPRM 作者发布的 MMLU-Pro 评估题单。

## 准备

Python 3.9 或更新版本运行数据准备和客户端。评分服务使用支持 BF16 的 Linux CUDA 环境，安装 [评分依赖](requirements-prm.txt)。已有 vLLM 服务可直接作为生成端，默认地址为 `http://127.0.0.1:18420/v1`。在 [assets.json](assets.json) 中固定了模型和数据版本。

```bash
python prepare.py mmlu --per-category 100 --output /data/general-ttc/questions.json
python prepare.py models --directory /data/general-ttc/models
python serve_prm.py --models /data/general-ttc/models --port 18423
```

## 试跑

准备 `--per-category 1` 的题单，在同一题单上运行四组。每组使用独立输出目录。

```bash
python prepare.py mmlu --per-category 1 --output /data/general-ttc/smoke-questions.json
for method in E1 E2 E3 E4; do
  python run.py --manifest /data/general-ttc/smoke-questions.json \
    --category math --category law --category history \
    --method "$method" --k 2 --search-steps 1 --code-revision "$(git rev-parse HEAD)" \
    --output "/data/general-ttc/smoke/$method"
done
```

输出包括配置、每题选中的答案和候选、控制器决定、PRM 步分、生成和评分 token、调用次数及实测耗时。搜索对照可以用 `--fixed-strategy beam` 或 `--fixed-strategy lookahead`。生成组可以用 `--workers` 并行提交题目。用同一配置和 `--resume` 继续运行时，已落盘的题目会被复用，允许调整并发。正式实验使用完整题单和默认轮数，并按学科汇总。

[数学版方法与实验配置](https://arxiv.org/html/2602.01070v5#S2) · [VersaPRM 的参考评分实现](https://github.com/UW-Madison-Lee-Lab/VersaPRM/blob/81a3c993eb80c38ee3900405bee67405dd79f610/evaluation/prm_models/prm_llama.py)