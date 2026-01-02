## 使用限制说明（From-scratch 要求）

本作业要求你**从零开始（from scratch）**构建相关组件。具体而言：
### 1. 禁止使用的内容
除下述明确允许的内容外，**不得使用**以下模块中的任何现成实现或定义：
- `torch.nn`
- `torch.nn.functional`
- `torch.optim`
### 2. 明确允许使用的内容
以下内容是**唯一被允许**从上述模块中使用的部分：
1. **参数类**
   - `torch.nn.Parameter`
2. **容器类（Container Classes）**
   - 位于 `torch.nn` 中的容器类，例如：
     - `Module`
     - `ModuleList`
     - `Sequential`
     - 等其他容器类
3. **优化器基类**
   - `torch.optim.Optimizer`（仅限基类）
### 3. 其他 PyTorch 功能
- 除上述限制外，**可以使用任何其他 PyTorch 提供的定义和功能**。
- 尽量使用 rearrange 、torch.einsum、torch.einx 等 tensor 操作函数，避免使用 for 循环。
## 3.4 基础模块：线性层与嵌入层
### 3.4.1 参数初始化
要有效训练神经网络，通常需要仔细初始化模型参数——不当的初始化可能导致梯度消失或梯度爆炸等问题。Pre-norm Transformer 对初始化比较鲁棒，但初始化仍会显著影响训练速度和收敛性。下面是一些在大多数情况下表现良好的近似初始化方法：
- **线性层权重**：  
  \[
  W \sim \mathcal{N}\Big(\mu=0, \sigma^2 = \frac{2}{d_{\text{in}} + d_{\text{out}}} \Big), \quad 截断在 [-3\sigma, 3\sigma]
  \]
- **嵌入层**：  
  \[
  E \sim \mathcal{N}(\mu=0, \sigma^2=1), \quad 截断在 [-3, 3]
  \]
- **RMSNorm**：初始化 scale 为 `1`。