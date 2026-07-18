# Physical K-State Technical Route

本脚本只分析选定曲线，并在图例与解释中使用中性命名。

## 技术路线

1. 读取选定的 Time/AI/AV 序列，并将 AI 幅值视为可观测的物理随机状态信号。
2. 对 AI 幅值拟合一维 Gaussian mixture 候选模型，并用 BIC 在 1..8 个隐状态之间选择 K。
3. 用带自保持偏置的 Gaussian HMM 和 Viterbi 解码，把带噪声的连续幅值轨迹分割为稳定的 K 个隐状态。
4. 按各状态的 AI 均值重新排序状态，提取每段 dwell time，并由相邻 dwell 段统计 KxK 转移矩阵。
5. 将 K-state 序列映射到 Gray-code p-bit bias，同时从 AI、跳变强度和局部波动中构造经验随机通道。
6. 把状态退出率作为 hazard rate，驱动 18-spin Ising 随机更新，并用多次运行统计能量 gap、命中率和接受率。
7. 沿外部轴做分箱稳定性检查，观察每个隐状态在不同区间中的占据比例与局部均值漂移。

## 关键结果

- Selected K: 6
- State entropy: 2.3962 bits
- Mean energy gap: 2.5500
- Optimum hit rate: 0.4000
