# Hugging Face 文本分类与参数高效微调

自然语言处理课程 HW2，目前完成任务 1–3。项目使用 Hugging Face Datasets 和 Transformers 构建分类流水线，对照全量微调、LoRA 和 Bottleneck Adapter 的效果。

## 项目内容

- **数据集**：加载 Restaurant、Laptop、ACL-ARC 和 AG News，支持少样本抽样及多数据集标签重映射。
- **全量微调**：在 Restaurant、ACL-ARC 和 AG News 上微调 BERT、BART 和 Qwen。每个组合使用 42、43、44 三个种子，共 27 次实验。
- **参数高效微调**：在 AG News 上比较三个模型的 LoRA 和 Bottleneck Adapter，并比较 BERT 的四组 LoRA 秩设置，共 9 次实验。
- **实验记录**：计算 Accuracy、Micro-F1、Macro-F1、Weighted-F1，通过 WandB 离线记录训练过程。

## 实验结果

### 全量微调

数值为三个种子的均值 ± 样本标准差，单位为百分比。完整四项指标见 [任务 2 汇总](results/task2_mean_and_std.csv)。

| 模型 | 数据集 | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| `bert-base-uncased` | `acl_sup` | 64.27 ± 2.31 | 28.15 ± 2.01 |
| `bert-base-uncased` | `agnews_sup` | 89.56 ± 0.15 | 89.60 ± 0.15 |
| `bert-base-uncased` | `restaurant_sup` | 79.58 ± 0.46 | 65.11 ± 1.33 |
| `facebook/bart-large` | `acl_sup` | 71.22 ± 2.88 | 46.66 ± 14.45 |
| `facebook/bart-large` | `agnews_sup` | 91.14 ± 0.38 | 91.15 ± 0.37 |
| `facebook/bart-large` | `restaurant_sup` | 85.60 ± 0.57 | 75.51 ± 1.52 |
| `Qwen/Qwen1.5-0.5B` | `acl_sup` | 76.26 ± 2.59 | 65.18 ± 3.69 |
| `Qwen/Qwen1.5-0.5B` | `agnews_sup` | 89.87 ± 0.60 | 89.94 ± 0.57 |
| `Qwen/Qwen1.5-0.5B` | `restaurant_sup` | 86.07 ± 0.63 | 77.91 ± 0.73 |

### AG News 微调方式对比

下表为种子 42 的 Macro-F1（%）。LoRA 使用 r=16、alpha=32。

| 模型 | 全量微调 | LoRA | Bottleneck Adapter |
| --- | ---: | ---: | ---: |
| `bert-base-uncased` | 89.49 | 87.53 | 88.67 |
| `facebook/bart-large` | 90.95 | 89.95 | 90.22 |
| `Qwen/Qwen1.5-0.5B` | 90.07 | 90.56 | 90.22 |

### BERT LoRA 参数对比

| r | alpha | Macro-F1（%） |
| ---: | ---: | ---: |
| 4 | 8 | 86.13 |
| 8 | 16 | 87.18 |
| 16 | 32 | 87.53 |
| 32 | 64 | 87.42 |

更多指标与每次运行的配置见 [任务 3 对比](results/task3_peft_comparison.csv) 和 [全部 36 组实验](results/all_runs.csv)。

## 目录结构

```text
hw2_huggingface/
  dataHelper.py          数据集加载、少样本与聚合
  train.py               全量微调、LoRA、Adapter 与评估
  test_datahelper.py     数据集测试
  test_trainscript.py    分类训练辅助函数测试
  data/README.md         数据放置说明
scripts/
  run_experiments.sh     短测与正式实验命令
results/                 已完成实验的指标和配置
ASSIGNMENT.md            作业来源与范围
```

## 运行方式

在仓库根目录运行。数据文件放到 [指定目录](hw2_huggingface/data/README.md) 后运行测试。第一次加载模型及评估模块时需要访问 Hugging Face。

```bash
uv sync --locked --python 3.11
uv run pytest -q hw2_huggingface/test_datahelper.py hw2_huggingface/test_trainscript.py
```

正式实验使用单张 RTX 3090 和 BF16。Linux 单卡完整流程可通过下面的命令复现。先确认短测结果，再运行正式批次。

```bash
bash scripts/run_experiments.sh smoke
bash scripts/run_experiments.sh formal
```

训练输出在 `outputs/`，WandB 离线记录在 `wandb/`。原始数据、模型权重和原始实验日志保存在本地。

## 作业来源

作业要求、初始代码框架和测试用例来自 [WangXiangyv/NLPDL-2026Fall](https://github.com/WangXiangyv/NLPDL-2026Fall)。更多内容见 [作业范围](ASSIGNMENT.md)。
