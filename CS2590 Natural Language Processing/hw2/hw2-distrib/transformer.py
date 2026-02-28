# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        # 字符 -> 向量：后续所有层都在这个向量空间里做计算
        self.char_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码：Q0（BEFOREAFTER）严格来说不靠位置也能做得不错，但这里先把模块接上，
        # 方便后续做 Q1 的 BEFORE 任务时直接打开。
        self.positional_encoding = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=False)
        self.use_positional_encoding = False  # 在 train_classifier 里按任务开关

        # 叠多层时要用 ModuleList，不能用普通 Python list（否则参数不会被优化器更新）
        self.layers = nn.ModuleList([TransformerLayer(d_model=d_model, d_internal=d_internal) for _ in range(num_layers)])

        # 输出层：对每个位置的 d_model 向量做分类，输出 3 类（0/1/2）
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # indices: [seq_len]，这里 seq_len 固定是 20
        # x: [seq_len, d_model]
        x = self.char_embedding(indices)

        # BEFOREAFTER 任务不依赖顺序；默认先关掉位置编码。需要顺序信息时再开。
        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)

        # logits: [seq_len, num_classes]；log_probs: [seq_len, num_classes]
        logits = self.classifier(x)
        log_probs = torch.log_softmax(logits, dim=1)
        return log_probs, attn_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # Q/K/V 的线性投影。这里做单头 self-attention：
        # - Q: [seq_len, d_internal]
        # - K: [seq_len, d_internal]
        # - V: [seq_len, d_model]  （直接让输出回到 d_model，方便残差相加）
        self.q_proj = nn.Linear(d_model, d_internal, bias=False)
        self.k_proj = nn.Linear(d_model, d_internal, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # 注意力缩放因子 1/sqrt(d_k)，能让训练更稳定
        self.attn_scale = 1.0 / np.sqrt(d_internal)

        # 前馈网络（position-wise MLP）：对每个位置独立地做两层线性变换
        self.ff1 = nn.Linear(d_model, d_internal)
        self.ff2 = nn.Linear(d_internal, d_model)
        self.activation = nn.ReLU()

    def forward(self, input_vecs):
        """
        :param input_vecs: [seq_len, d_model]
        :return: (output_vecs, attn_map)
          - output_vecs: [seq_len, d_model]
          - attn_map: [seq_len, seq_len]，第 i 行表示第 i 个位置对所有位置的注意力分布
        """
        # 1) self-attention
        # q, k: [seq_len, d_internal]；v: [seq_len, d_model]
        q = self.q_proj(input_vecs)
        k = self.k_proj(input_vecs)
        v = self.v_proj(input_vecs)

        # 打分矩阵 scores: [seq_len, seq_len]，scores[i, j] 表示 i 位置 query 与 j 位置 key 的相似度
        # 这里用矩阵乘法 q @ k^T，并按论文做缩放
        scores = torch.matmul(q, k.transpose(0, 1)) * self.attn_scale

        # 注意力权重：对“被关注的位置维度”做 softmax（每一行和为 1）
        attn = torch.softmax(scores, dim=1)

        # 加权求和得到上下文向量：[seq_len, d_model]
        context = torch.matmul(attn, v)

        # 2) 残差连接：把 attention 输出加回输入，保留原始信息并缓解梯度问题
        x = input_vecs + context

        # 3) 前馈网络（两层线性 + 非线性），依旧是每个位置独立处理
        ff = self.ff2(self.activation(self.ff1(x)))

        # 4) 第二个残差连接
        out = x + ff

        return out, attn


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        # 位置索引要跟输入在同一设备上（CPU/MPS/CUDA），不然加法会直接报 device mismatch
        indices_to_embed = torch.arange(0, input_size, device=x.device, dtype=torch.long)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    """
    Part 1 的训练入口。这里不做 batching（样本很短、数据也不大），先保证实现清晰易改。
    """
    # 设备选择：CUDA > MPS(Apple Silicon) > CPU
    # 这个顺序基本符合速度预期，也最省事
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 一些比较“稳”的默认超参：BEFOREAFTER 任务不靠位置也能学到 85%+（通常更高）
    vocab_size = 27
    num_positions = 20
    num_classes = 3
    d_model = 64
    d_internal = 64
    num_layers = 1

    model = Transformer(
        vocab_size=vocab_size,
        num_positions=num_positions,
        d_model=d_model,
        d_internal=d_internal,
        num_classes=num_classes,
        num_layers=num_layers,
    ).to(device)

    # autograder 里 args 可能不带 task 字段，这里做兼容处理。
    # 默认按 BEFORE 行为处理（即启用位置编码）；只有明确 BEFOREAFTER 时才关闭。
    task_name = getattr(args, "task", "BEFORE")
    model.use_positional_encoding = (task_name != "BEFOREAFTER")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        loss_this_epoch = 0.0

        # 每轮打乱数据，避免模型每次都按同样顺序看样本
        random.seed(epoch)
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)

        start_time = time.time()
        for ex_idx in ex_idxs:
            ex = train[ex_idx]

            # 把数据放到同一设备上
            input_tensor = ex.input_tensor.to(device)    # [20]
            output_tensor = ex.output_tensor.to(device)  # [20]

            log_probs, _ = model(input_tensor)  # [20, 3]

            # NLLLoss 期望输入是 [N, C]，target 是 [N]
            loss = loss_fcn(log_probs, output_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss.item()

        elapsed = time.time() - start_time
        avg_loss = loss_this_epoch / max(1, len(train))

        # 训练过程简单打印一下：loss 是否在下降、每轮耗时是否正常
        print(f"epoch {epoch+1}/{num_epochs} | avg_train_loss={avg_loss:.4f} | time={elapsed:.1f}s")

    # 训练可在 GPU/MPS 上做，但 decode() 固定喂的是 CPU 张量；
    # 这里把模型搬回 CPU，避免后续评估阶段出现 device mismatch。
    model.to(torch.device("cpu"))
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
