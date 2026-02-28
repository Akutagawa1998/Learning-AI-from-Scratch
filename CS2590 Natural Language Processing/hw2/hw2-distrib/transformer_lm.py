# transformer_lm.py

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim


class TransformerLanguageModel(nn.Module):
    """
    基于 TransformerEncoder 的字符级语言模型。
    输入一段字符索引序列，输出每个位置“下一个字符”的对数概率分布。
    """
    def __init__(self, vocab_size, num_positions, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        # 字符 embedding 与位置 embedding 都映射到 d_model，二者直接相加
        self.char_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(num_positions, d_model)

        # 这里允许用官方 TransformerEncoder（Part 2 作业明确允许）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 每个位置的隐藏向量 -> 词表大小 logits
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.num_positions = num_positions

    def _causal_mask(self, seq_len, device):
        """
        生成因果 mask：禁止当前位置看到未来位置。
        上三角（不含对角线）为 -inf，其余为 0。
        """
        # 直接用 bool mask（True 表示被屏蔽），比 float mask 更省内存
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, idx):
        """
        :param idx: [batch_size, seq_len] 的字符索引
        :return: [batch_size, seq_len, vocab_size] 的 log_probs
        """
        batch_size, seq_len = idx.shape
        device = idx.device

        # 位置 id: [0, 1, 2, ... seq_len-1]，再复制到 batch 维
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
        x = self.char_emb(idx) + self.pos_emb(pos_ids)

        # 因果 mask 保证自回归训练合法，不会“偷看答案”
        causal_mask = self._causal_mask(seq_len, device)
        h = self.encoder(x, mask=causal_mask)

        logits = self.output_proj(h)
        return torch.log_softmax(logits, dim=-1)


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index, num_positions, device):
        # 保存训练好的神经网络与索引器，供推理接口调用
        self.model = model
        self.vocab_index = vocab_index
        self.num_positions = num_positions
        self.device = device

    def get_next_char_log_probs(self, context):
        """
        输入任意长度 context，返回下一个字符的 log 概率（长度=词表大小）。
        做法：构造输入 " " + context，并取最后一个位置的输出分布。
        """
        self.model.eval()

        # 约定空格作为起始符，和作业描述一致
        model_input = " " + context

        # 长上下文时只保留最近 num_positions 个字符，保证不越过位置 embedding 上限
        if len(model_input) > self.num_positions:
            model_input = model_input[-self.num_positions:]

        input_ids = []
        for c in model_input:
            idx = self.vocab_index.index_of(c)
            if idx < 0:
                raise ValueError("Context contains character not in vocabulary: %s" % repr(c))
            input_ids.append(idx)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            # [1, seq_len, vocab] -> 取最后位置 -> [vocab]
            log_probs = self.model(input_tensor)[0, -1, :]
        return log_probs.detach().cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        """
        按定义逐字符累加：
        log P(next_chars | context) = sum_t log P(next_char_t | context + prefix_t)
        """
        total_log_prob = 0.0
        running_context = context
        for c in next_chars:
            next_char_log_probs = self.get_next_char_log_probs(running_context)
            c_idx = self.vocab_index.index_of(c)
            if c_idx < 0:
                raise ValueError("next_chars contains character not in vocabulary: %s" % repr(c))
            total_log_prob += float(next_char_log_probs[c_idx])
            running_context += c
        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # 设备优先级：CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 这里用一组更“省内存”的超参，避免 autograder 低内存环境 OOM
    # 重点是减小 batch 和前馈层宽度；另外把 dropout 设为 0，减少训练时额外张量分配。
    vocab_size = len(vocab_index)
    num_positions = 20
    d_model = 48
    nhead = 4
    num_layers = 1
    dim_feedforward = 64
    dropout = 0.0
    batch_size = 4
    num_epochs = 16
    learning_rate = 2e-3

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        num_positions=num_positions,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fcn = nn.NLLLoss()

    # 把长文本切成固定长度 chunk。每个 chunk 预测 chunk 内每个位置的下一个字符。
    # 输入构造：x = [prev_char] + chunk[:-1]，y = chunk
    train_ids = [vocab_index.index_of(c) for c in train_text]
    seq_len = num_positions
    example_starts = [i for i in range(0, len(train_ids) - seq_len + 1, seq_len)]

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(example_starts)
        total_loss = 0.0
        num_batches = 0

        for b_start in range(0, len(example_starts), batch_size):
            batch_starts = example_starts[b_start:b_start + batch_size]
            if len(batch_starts) == 0:
                continue

            x_batch = []
            y_batch = []
            for s in batch_starts:
                target_chunk = train_ids[s:s + seq_len]  # y
                prev_char = train_ids[s - 1] if s > 0 else vocab_index.index_of(" ")
                input_chunk = [prev_char] + target_chunk[:-1]  # x
                x_batch.append(input_chunk)
                y_batch.append(target_chunk)

            x_tensor = torch.tensor(x_batch, dtype=torch.long, device=device)          # [B, T]
            y_tensor = torch.tensor(y_batch, dtype=torch.long, device=device)          # [B, T]

            log_probs = model(x_tensor)                                                # [B, T, V]
            loss = loss_fcn(log_probs.reshape(-1, vocab_size), y_tensor.reshape(-1))  # 展平做 token-level loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print("epoch %d/%d | avg_train_loss=%.4f" % (epoch + 1, num_epochs, avg_loss))

    # 推理阶段沿用训练时选出的设备，设备选择逻辑始终保持 CUDA > MPS > CPU
    model.to(device)
    model.eval()
    return NeuralLanguageModel(model, vocab_index, num_positions, device)
