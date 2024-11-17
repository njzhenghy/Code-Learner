# MOCO

### 对比学习常数

**温度常数**（temperature constant, T）是用于控制分布的平滑度的超参数，主要出现在对比损失函数中，例如InfoNCE损失。

在公式中，温度常数被用来调节对比学习的logits：
$$
\text{loss} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{k}^+)/T)}{\exp(\text{sim}(\mathbf{q}, \mathbf{k}^+)/T) + \sum_{i=1}^{K} \exp(\text{sim}(\mathbf{q}, \mathbf{k}_i^-)/T)}
$$
