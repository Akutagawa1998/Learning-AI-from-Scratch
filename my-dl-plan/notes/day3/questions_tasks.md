# Day 3 - Questions & Tasks (from MNIST.ipynb)

## Tasks
- Task 0: 选择合适的设备（cuda / mps / cpu）。
- Task 1: 进行数据预处理，下载并加载 MNIST 数据集。
- Task 2: 手动检查训练集的形状。
- Task (advanced):
  - 手动实现 train() 训练循环。
  - 手动实现 forward / backward 的逻辑（理解梯度流）。
  - 调整训练数据规模（不使用全部训练集）。

## Questions
- Question 1: train_ds 是一个什么样的结构？
  - 数据集对象，提供 __len__ / __getitem__，返回 (image, label)。
- Question 2: Why normalize the image?
  - 稳定训练、加速收敛、让特征分布更一致。
- Question 3: What is self.fc1 and how it is called?
  - self.fc1 是 nn.Linear 的实例，保存参数；调用时触发 __call__ -> forward。
- Question 4: What is an optimizer?
  - 根据梯度更新参数的算法（如 SGD/Adam），通过 optimizer.step() 执行更新。
