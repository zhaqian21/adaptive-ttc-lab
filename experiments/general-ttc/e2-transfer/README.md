# E2 跨模型与评分器的在线对照

继承 PR #46 的实际在线运行器。新增 Llama + dPRM、Qwen + VersaPRM、Llama + VersaPRM 各四种组合，依次运行固定 10 条＋平均步分、固定 10 条＋最低步分、五条一致早停＋平均步分、五条一致早停＋最低步分。

每轮使用同一份冻结题单、种子 0、输出上限 1,024 token、搜索步数 2、BF16 和 96 并发。统一重启生成端，在正式计时前以相同策略和 8 并发预热同样的 8 题。早停后跳过 dPRM 或 VersaPRM，其余题补齐十条再评分。各轮独立在线运行，vLLM 请求调度可能带来同种子下的输出变化。运行重试时保留已完成题目。

## 运行与验收

AutoDL 的运行目录放在 general-ttc 数据目录下，和 `extension-mmlu-pro`、`e2-online-ablation` 同级。使用已有的 `extension-mmlu-pro/run_matrix.py` 管理单卡上的生成与评分服务。运行目录需要当前生成端和评分端的 PID 文件，以及原 dPRM 依赖目录和评分脚本入口。仓库工作树、模型文件、题单和 GitHub CLI 授权均保存在服务器。

```bash
bash host_transfer.sh /data/general-ttc/e2-transfer
python -B archive_transfer.py
```

后台启动队列后，`stage`、`queue.log` 和 `publication.log` 记录运行阶段、完成统计和上传验收结果。全量完成后自动归档 16,800 条逐题记录，重算正确率和 token，核对 GitHub 的文件对象后提交 PR。Qwen + dPRM 行引用 PR #46 已验收的实测记录。
