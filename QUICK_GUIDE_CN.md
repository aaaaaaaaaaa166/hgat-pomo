# HGAT-POMO 极简导读（先看这个）

这份文档目标只有一个：让你先“能看懂主链路”，再逐步补细节。  
你不需要一次读完所有代码。

## 先记住一句话

每一步决策是：**卡车去 j，无人机可选去 k 并在 j 会合**。  
环境把这一步的并行执行时间记为 `dt=max(卡车时间, 无人机时间)`，奖励是 `-dt-迟到惩罚`。

---

## 30 分钟最短阅读路径

1. 看训练主循环：`src/main_train.py`
2. 看环境一步转移：`src/env/td_env.py` 的 `step()`
3. 看策略怎么出动作：`src/models/policy.py` + `src/models/decoder_pomo.py`
4. 看 rollout 怎么累计回报：`src/rl/pomo_rollout.py`

读这 4 处时，只回答下面 4 个问题：

1. 状态里有什么？（`t, i, served, soc`）
2. 动作是什么？（`(k, j)`）
3. 奖励怎么算？（`-dt-penalty`）
4. 损失怎么算？（`-(adv * logp).mean()`，其中 `adv=R-基线`）

---

## 代码地图（按“作用”分组）

### 1) 数据与环境

- `src/env/instance_gen.py`  
  生成随机实例：坐标、释放时间、需求、截止时间。

- `src/env/td_env.py`  
  环境核心：可行性检查、动作 mask、`step()` 状态转移与奖励。

### 2) 图表示与策略

- `src/graph/build_graph_pyg.py`  
  把当前状态转成异构图（truck/drone/order 三类节点）。

- `src/models/hgat_encoder.py`  
  异构图编码器，输出每类节点向量表示。

- `src/models/decoder_pomo.py`  
  两阶段解码：先选 `j`，再选 `k` 或 `no-drone`。

- `src/models/policy.py`  
  把“建图+编码+解码”串起来，提供训练/推理接口。

### 3) 训练与评估

- `src/rl/pomo_rollout.py`  
  同一实例跑 K 条轨迹（POMO），收集 `returns/logps/entropy`。

- `src/main_train.py`  
  训练入口：采样实例 -> rollout -> 计算损失 -> 更新参数。

- `src/main_eval.py`  
  评估入口：统计 best/mean/worst，并对比 baseline。

---

## 2 小时理解版（更稳）

### 第 1 阶段（30 分钟）：只读环境

重点盯 `src/env/td_env.py` 的这几个函数：

1. `get_masks()`：动作可行域怎么限制  
2. `_drone_feasible()`：无人机可行性（载重/航程/电量）  
3. `step()`：时间推进、服务完成、SoC 更新、奖励计算

你读完应能回答：为什么是 `dt=max(truck_time, drone_time)`。

### 第 2 阶段（40 分钟）：只读策略决策

1. `src/models/policy.py` 的 `forward_step()`  
2. `src/models/decoder_pomo.py` 的 `select_j()` 和 `select_k()`

你读完应能回答：为什么动作分成两步，而不是一次性选 `(k,j)`。

### 第 3 阶段（50 分钟）：只读训练闭环

1. `src/rl/pomo_rollout.py`  
2. `src/main_train.py`（从 `for ep in ...` 开始）

你读完应能回答：

1. 为什么同一个实例要跑 K 条轨迹？  
2. POMO baseline 在哪里进入损失？  
3. entropy/temperature 在做什么？

---

## 你只要记住的 6 个变量

1. `t`：当前时间  
2. `i`：卡车当前位置  
3. `served`：订单是否已完成  
4. `soc`：无人机电量  
5. `j`：卡车下一跳  
6. `k`：无人机服务订单（或 `K_NONE`）

---

## 新手最容易卡住的点

1. 把 `return` 和 `cost` 搞反  
这里 `reward` 是负成本，所以 `cost = -return`。

2. 以为卡车和无人机串行  
它们在一步内并行，步长由慢的一方决定。

3. 没区分训练接口和推理接口  
`forward_step()` 用于训练保留梯度，`act()` 用于推理。

---

## 一条“边看边验证”的命令

先小规模跑一次，结合日志看代码最容易懂：

```bash
python -m src.main_train --epochs 3 --batch-size 4 --N 12 --K 4 --save-path policy_smoke.pt
python -m src.main_eval --model-path policy_smoke.pt --N 12 --K 4 --n-instances 10
```

读代码时，把日志字段和变量一一对应：

- `cost_mean/cost_best` 对应 `-returns` 的统计
- `entropy` 对应 rollout 收集的熵项
- `N/B/K` 对应实例规模、batch、POMO 轨迹数

---

## 下一步（建议）

当你能完整说出一次 `step -> action -> reward -> loss` 的流程后，  
再去看 `src/experiments/run_ablation.py`，否则会被实验工程细节干扰。
