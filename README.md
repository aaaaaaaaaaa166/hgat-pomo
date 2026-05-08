# HGAT-POMO 毕设项目整理说明

本仓库保留论文写作和实验复现需要的核心材料：源码、小型 CVRPLIB 数据、正式实验指标、外部方法脚本与依赖说明。运行缓存、IDE 配置、模型权重、训练日志和 smoke/debug 结果不再放在工作区里。

## 目录结构

- `src/`: 主方法代码，包括环境、图构建、HGAT/POMO 模型、训练与评估入口。
- `datasets/`: CVRPLIB 实例和 `splits/` 划分文件。
- `experiments/`: 主实验、消融实验、调参和正式协议结果。只建议保留配置、指标与汇总表。
- `external_methods/`: 外部方法复现实验脚本、依赖说明与汇总结果。

## 论文写作优先材料

- 主实验表格: `experiments/thesis_protocol_20260420_formal/`
- 消融实验汇总: `experiments/ablation_final_20260308/`
- 道路感知实验: `experiments/road_formal_20260402/`
- 静态/道路环境对比: `experiments/static_vs_road_20260402/`
- 外部方法对比汇总: `external_methods/results/comparison_summary.md`

## 后续保存原则

新增实验只保留 `config.json`、`summary.*`、`runs.csv`、`metrics/*.json` 和论文表格。`*.pt`、`*.log`、`checkpoints/`、`models/`、`__pycache__/`、smoke/debug 目录都可以重新生成，默认不保留。
