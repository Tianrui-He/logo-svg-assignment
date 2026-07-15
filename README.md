# Part B：Gemma 3 270M 生成 SVG 徽标

本仓库已完成数据整理、可解释 reward、负例校准、自动测试、LoRA 训练、基座/适配器公平评测、案例导出和提交检查。最终 LoRA 来自 `checkpoint-224`：公开验证集平均代理 reward 从 1.8824 提升到 22.8096，有效 SVG 比例从 0% 提升到 23.53%。模型仍存在明显的重复、截断和 Goodhart 效应，详见 `report.md`。

## 当前文件

- `train.jsonl`：219 条官方训练样本。
- `valid.jsonl`：17 条公开验证样本。
- `reward.py`：提交入口；完整实现位于 `student_kit/reward.py`。
- `reward_calibration.json`：236 条参考 SVG 和四类负例的校准结果。
- `train_config.yaml`：ms-swift 4.x 的首选训练配置。
- `student_kit/eval_self.py`：固定解码参数比较基座与 LoRA。
- `results.json`：固定 17 条验证集上的基座/LoRA 逐样本结果。
- `report.md`：方法、真实实验数字、案例和失败分析。
- `EXPERIMENTS.md`：已完成实验、checkpoint 趋势和未完成实验的诚实记录。
- `tests/test_reward.py`：reward 回归测试。

模型选用 ModelScope 的 `google/gemma-3-270m`。它是 270M 的指令微调版本，适合本数据的 system/user/assistant chat 格式；提交时必须用同一 checkpoint 同时做基座评测和 LoRA 加载。如果老师另行指定非 IT checkpoint，则应在任何训练开始前把所有命令统一改为 `google/gemma-3-270m`，不能混用。

## 一、在本地先检查 reward

```bash
python -m pytest -q
python student_kit/audit_reward.py --output reward_calibration.json
```

预期为 7 个测试通过。当前校准结果：参考 SVG 平均 98.12/100、最低 90.16；破损 XML 2.00、空 SVG 42.15、纯背景 59.82、脚本注入被封顶到 20.00。

## 二、在百度 AI Studio 准备环境

新建一个 GPU 项目，把整个仓库上传或从自己的 GitHub 仓库克隆。进入项目目录后运行：

```bash
bash scripts/setup_ai_studio.sh
```

脚本会安装依赖、从 ModelScope 下载约 575 MB 的模型并重新运行测试。不要无条件重装 `torch`，AI Studio 自带的 CUDA 版本通常最合适。手工等价命令是：

```bash
python -m pip install -U -r requirements.txt
modelscope download --model google/gemma-3-270m --local_dir ./gemma3-270m
python -m pytest -q
```

ms-swift 4.x 的官方参数名是 `tuner_type`；3.x 才使用旧的 `train_type`。本仓库按 4.x 编写。官方资料：

- [ms-swift 支持模型列表](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md)
- [ms-swift 命令行参数](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
- [自定义 messages 数据格式](https://swift.readthedocs.io/en/v4.3/Customization/Custom-dataset.html)
- [ModelScope Gemma 3 270M IT](https://modelscope.cn/models/google/gemma-3-270m)

## 三、先生成基座结果

先用 2 条数据冒烟测试，确认显存、模板和生成都正常：

```bash
python student_kit/eval_self.py \
  --model ./gemma3-270m \
  --valid valid.jsonl \
  --output base_smoke.json \
  --limit 2 \
  --max-new-tokens 2048 \
  --seed 42
```

再生成全部 17 条基座结果：

```bash
python student_kit/eval_self.py \
  --model ./gemma3-270m \
  --valid valid.jsonl \
  --output base_results.json \
  --max-new-tokens 2048 \
  --seed 42
```

基座与最终适配器仍会在一次 `results.json` 评测中重新生成，避免环境或代码变化造成不公平比较。

## 四、运行首个 LoRA 实验

```bash
bash scripts/train.sh
```

配置的重要选择：

- rank 16、alpha 32、dropout 0.05；
- 学习率 2e-4、cosine 调度；
- 最大序列 4096；超长样本删除而不是截断，避免学习残缺 `</svg>`；
- 每设备 batch 1、梯度累积 8，有效 batch 8；
- 每 28 个优化步（约一轮）验证和保存；
- 最多 8 轮，验证 loss 连续三个保存周期不改善则早停；
- `loss_scale: last_round`，只对 assistant 的 SVG 计算损失。

检查训练日志中的预处理样本数。如果因为 `max_length` 删除了样本，先确认数量；删除较多时将 `max_length` 提高到 6144，而不是使用右截断。若 GPU 不支持 BF16，把命令覆盖为：

```bash
swift sft train_config.yaml --model ./gemma3-270m --torch_dtype float16
```

## 五、找到最佳 checkpoint 并公平评测

```bash
python student_kit/find_best_checkpoint.py output
```

复制输出的 `best_model_checkpoint` 路径，然后运行：

```bash
bash scripts/evaluate.sh output/gemma3-270m-lora-r16/checkpoint-XXX
```

这会在同一进程设置、同一验证集、同一种贪心解码和同一个 seed 下生成基座与适配器结果，写入 `results.json`。核心结果位于：

```text
summary.base
summary.adapted
summary.delta
```

不能在看到适配器结果后修改 reward 而只重评适配器；reward 发生任何变化都必须重跑两边。

## 六、做最少而有意义的调参

先阅读 E1 的验证 loss 与生成案例，再按 `EXPERIMENTS.md` 一次只改一个变量：

1. rank 16 → rank 8；
2. 学习率 2e-4 → 1e-4；
3. 只有发生大量截断时才调整最大长度；
4. 通过最佳 checkpoint/早停处理过拟合，不要盲目增加 epoch。

每个候选都用完整 17 条验证集评测，将数据填入 `EXPERIMENTS.md`。最终只提交选中 checkpoint 对应的准确 `train_config.yaml`；如果最佳实验使用了覆盖参数，必须把覆盖值改回 YAML。

## 七、导出案例并完成报告

```bash
python student_kit/export_examples.py --results results.json --output-dir examples
```

打开 `examples/comparisons.html`，它自动选择 reward 提升最大、中位和最差的三组前后对比。人工观察它们是否真的更好，并在 `report.md` 中讨论：

- 合法率提升是否只是学会了固定 SVG 模板；
- prompt fidelity 是否真的改善；
- 高 reward 但视觉或语义较差的 Goodhart 案例；
- 验证 loss 最优与 reward 最优是否是同一 checkpoint；
- 生成是否因 2048 token 上限而截断。

检查所有待填内容：

```bash
rg -n "TODO" report.md EXPERIMENTS.md
```

## 八、整理 adapter 并检查提交

```bash
python student_kit/package_submission.py \
  --checkpoint output/gemma3-270m-lora-r16/checkpoint-XXX \
  --results results.json
```

脚本会将 `adapter_config.json` 与 `adapter_model.safetensors` 复制到 `adapter/`，并检查必交文件。最终仓库至少应有：

```text
adapter/adapter_config.json
adapter/adapter_model.safetensors
reward.py
student_kit/reward.py
train_config.yaml
results.json
report.md
```

最后重新运行测试、确认没有 TODO，再初始化并推送 GitHub：

```bash
python -m pytest -q
git init
git add adapter reward.py student_kit train_config.yaml results.json report.md README.md EXPERIMENTS.md reward_calibration.json tests scripts requirements.txt
git commit -m "Complete Gemma 3 270M SVG LoRA assignment"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/YOUR_REPO.git
git push -u origin main
```

不要提交 `gemma3-270m/` 完整模型或 `output/` 中所有 checkpoint；它们体积大且不是作业要求。只提交几 MB 的最佳 LoRA adapter。
