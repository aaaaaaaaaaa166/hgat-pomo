# Thesis Conclusion Paragraph

在原始动态响应窗口和单资源串行调度条件下，80% 接单率与 50% 准时率的联合目标不可达。多轮模型侧优化，包括 reward 调整、V2 repair、joint teacher、safe-deviation teacher 以及 ServicePolicy 准备流程，均未能在原始业务约束下稳定突破 baseline 并同时满足硬约束。业务约束敏感性实验表明，目标达成依赖响应窗口、配送时窗和并行资源的组合放宽，而不是单一模型训练带来的改进。最小稳定观察可行组合为 response_window=5.0、delivery_window_extension=+3.0、resources=2，对应方法为 oracle_best_on_time。在早期 eval=30 验证中，该组合已达到 acc=0.912、on_time=0.533、hard=0。本轮稳定性验证结果为：eval=50 时 acc=0.915、on_time=0.519、hard=0；eval=100 时 acc=0.912、on_time=0.524、hard=0。因此，本研究建议优先调整业务约束和资源配置，而不是继续投入旧模型的长时间训练或继续扩展 teacher/ServicePolicy 流程。
