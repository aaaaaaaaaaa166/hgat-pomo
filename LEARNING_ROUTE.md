# HGAT-POMO 学习路线（结合本项目）

这份路线按“先跑通、再理解、再改进”的节奏设计，目标是让你能独立修改环境、模型和训练策略，并复现实验结果。

## 阶段 0：先跑通（0.5 天）

目标：确认训练和评估脚本能完整执行，先建立整体感。

1. 安装依赖（建议在虚拟环境）。
2. 运行一次小规模训练：
```bash
python -m src.main_train --epochs 3 --batch-size 4 --N 12 --K 4 --save-path policy_smoke.pt
```
3. 用上一步模型做快速评估：
```bash
python -m src.main_eval --model-path policy_smoke.pt --N 12 --K 4 --n-instances 10
```

完成标志：你能看到训练日志中的 `loss/cost/entropy`，并得到评估 summary。

## 阶段 1：理解环境语义（1-2 天）

目标：彻底搞懂 action、mask、reward 是如何定义的。

阅读顺序：
1. `src/env/instance_gen.py`
2. `src/env/td_env.py`
3. `src/main_eval.py` 中 baseline 相关函数

重点问题：
1. `action=(k, j)` 中卡车和无人机如何并行？
2. `get_masks()` 如何限制可行动作？
3. `step()` 中 `dt=max(truck_time, drone_time)` 为什么合理？
4. SoC（电量）和截止时间惩罚如何进入奖励？

练习建议：
1. 固定一个小实例（N=8），打印每一步 `info`。
2. 分别关闭 `traffic_sigma` 和 `lateness_penalty`，观察 cost 变化。

## 阶段 2：理解图表示与策略网络（2-3 天）

目标：理解状态如何变成异构图，策略如何两阶段决策。

阅读顺序：
1. `src/graph/build_graph_pyg.py`
2. `src/models/hgat_encoder.py`
3. `src/models/decoder_pomo.py`
4. `src/models/policy.py`

重点问题：
1. order/truck/drone 三类节点各自特征是什么？
2. 五类边（t2o/o2t/d2o/o2d/o2o）分别传递什么信息？
3. 为什么先选 `j` 再选 `k`？
4. `act()` 和 `forward_step()` 在训练/推理上的差异是什么？

练习建议：
1. 把 `k_nn_orders` 从 8 改到 4/12，比较评估结果。
2. 只改 decoder 的 `temperature`，观察探索度变化。

## 阶段 3：理解训练机制（1-2 天）

目标：理解 POMO + REINFORCE 的损失构造和方差控制。

阅读顺序：
1. `src/rl/pomo_rollout.py`
2. `src/main_train.py`
3. `src/rl/reinforce.py`（简化版公式参考）

重点问题：
1. 为什么同一实例要 rollout K 条轨迹？
2. POMO baseline（对 K 求均值）如何降方差？
3. 熵正则和温度退火对训练稳定性有什么影响？
4. curriculum（从小 N 到目标 N）何时有帮助？

练习建议：
1. 记录不同 `K`（4/8/16）下的训练波动。
2. 比较 `use_curriculum` 开/关在前 50 epoch 的收敛速度。

## 阶段 4：实验复现与消融（2-3 天）

目标：能系统比较组件贡献，并产出可写论文的表格。

阅读与执行：
1. `src/experiments/README.md`
2. `python -m src.experiments.run_ablation ...`
3. `python -m src.experiments.export_threeline_table ...`

重点问题：
1. `no_traffic/no_time_window/no_soc/no_curriculum` 分别验证了什么？
2. 哪些设置影响 “best(K)” 与 “mean(K)” 的差距？

## 7 天最小可行计划

1. Day 1：跑通 train/eval，记录 baseline vs model。
2. Day 2：读 `td_env.py`，手动画一次 step 状态转移。
3. Day 3：读图构建和 encoder，搞清 feature 维度。
4. Day 4：读 decoder/policy，跟一遍 action 采样路径。
5. Day 5：读 rollout/train，推导 loss 公式。
6. Day 6：做 1 组小消融（例如 no_traffic）。
7. Day 7：整理结论，补注释和实验报告。

## 常见卡点

1. 看不懂结果先别调模型，先核对环境约束和 mask。
2. 训练不稳定先调 `temperature/entropy_coef/K`，再考虑改网络。
3. 先用小规模（N=12）做快速迭代，稳定后再上 N=30。
