# Day 3 - Minimal Model (Softmax Regression)

## 目标
- 用最小可学习模型（Softmax / Logistic Regression 多分类）跑通 MNIST 训练闭环。
- 训练过程中 loss 可观察到下降（1 epoch 内前后对比明显）。
- 记录指标到 `results/week01/metrics.csv`（至少包含 epoch / train_loss / val_acc）。

## 验收标准
- `python train.py --config configs/linear_mnist.yaml` 能完整跑完 1 epoch。
- 日志中可看到 train_loss 整体下降（允许波动）。
- `results/week01/metrics.csv` 存在且包含列：epoch, train_loss, val_acc。
- val_acc 明显高于 10%。

## 运行建议
- 先在 Notebook 跑通，再落地到 .py。
- 使用 MNIST 标准归一化（mean=0.1307, std=0.3081）。

## 相关文件
- `my-dl-plan/src/models/linear.py`
- `my-dl-plan/configs/linear_mnist.yaml`
- `my-dl-plan/train.py`
- `my-dl-plan/notes/day3/questions_tasks.md`
