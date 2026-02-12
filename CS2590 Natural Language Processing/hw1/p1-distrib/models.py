# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
import sys
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        # raise Exception("Must be implemented")
        self.indexer = indexer
    
    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        # bias 项：把常数 1 当作一个特征（模型的截距），对应 weights 里的一维参数
        bias_idx = self.indexer.add_and_get_index("BIAS", add_to_indexer)
        if bias_idx != -1:
            feats[bias_idx] += 1
        for word in sentence:
            idx = self.indexer.add_and_get_index(word, add_to_indexer)
            if idx != -1:
                feats[idx] += 1
        return feats



class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        # bias 项：把常数 1 当作一个特征（模型的截距），对应 weights 里的一维参数
        bias_idx = self.indexer.add_and_get_index("BIAS", add_to_indexer)
        if bias_idx != -1:
            feats[bias_idx] += 1

        for i in range(len(sentence) - 1):
            word = sentence[i]
            next_word = sentence[i + 1]
            feat_name = f"Bigram={word} {next_word}"
            idx = self.indexer.add_and_get_index(feat_name, add_to_indexer)
            if idx != -1:
                feats[idx] += 1
        return feats


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "while", "with", "of", "at",
            "by", "for", "to", "in", "on", "from", "up", "down", "out", "over", "under",
            "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
            "have", "has", "had", "this", "that", "these", "those", "it", "its", "as",
            "so", "than", "too", "very"
        }
        self.negations = {"not", "no", "never", "n't"}
        self.negation_window = 3

    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        bias_idx = self.indexer.add_and_get_index("BIAS", add_to_indexer)
        if bias_idx != -1:
            feats[bias_idx] += 1

        n = len(sentence)
        if n <= 4:
            len_feat = "LenBucket=short"
        elif n <= 12:
            len_feat = "LenBucket=medium"
        else:
            len_feat = "LenBucket=long"
        len_idx = self.indexer.add_and_get_index(len_feat, add_to_indexer)
        if len_idx != -1:
            feats[len_idx] += 1

        negate_left = 0
        for word in sentence:
            if word in self.negations:
                negate_left = self.negation_window
                neg_idx = self.indexer.add_and_get_index("HasNegation", add_to_indexer)
                if neg_idx != -1:
                    feats[neg_idx] += 1
                continue

            if word in self.stopwords:
                if negate_left > 0:
                    feat_name = f"NegUnigram={word}"
                else:
                    continue
            else:
                feat_name = f"NegUnigram={word}" if negate_left > 0 else f"Unigram={word}"

            idx = self.indexer.add_and_get_index(feat_name, add_to_indexer)
            if idx != -1:
                feats[idx] += 1

            if negate_left > 0:
                negate_left -= 1

        for idx in list(feats.keys()):
            if feats[idx] > 2:
                feats[idx] = 2
        return feats


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        # raise Exception("Must be implemented")
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        feats = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)
        score = 0.0
        for (idx, value) in feats.items():
            score += self.weights[idx] * value
        prob = 1.0 / (1.0 + np.exp(-score))
        if prob >= 0.5:
            return 1
        else:
            return 0



def train_logistic_regression(args, train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # raise Exception("Must be implemented")
    
    lr_passed = any(arg == "--lr" or arg.startswith("--lr=") for arg in sys.argv[1:])
    lr = args.lr if lr_passed else 0.1
    num_epochs = args.num_epochs

    # 先构建特征空间
    for ex in train_exs:
        feats = feat_extractor.extract_features(ex.words, add_to_indexer=True)
    weights = np.zeros(len(feat_extractor.get_indexer()))

    # ------------------------------------------------------------
    # This is my maunally written code, which is slower than the numpy version.
    # But it works.
    # Please forgive the Chinese comments, because I want to keep it for my own reference in my blog.
    # 我第一次手写的代码：for循环逐项累加/更新，比numpy版本慢。
    # ------------------------------------------------------------
    # # SGD：训练多轮（num_epochs），每轮 shuffle
    # for epoch in range(num_epochs):
    #     random.shuffle(train_exs)
    #
    #     # 训练模型，每一句：
    #     for ex in train_exs:
    #         # 用到了哪些特征(词)
    #         feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
    #         score = 0.0
    #
    #         # 对于每一个词：
    #         for (idx, value) in feats.items():
    #             # 计算分数： 权重 * 词频
    #             score += weights[idx] * value
    #
    #         # 最终sigmoid函数得到概率
    #         prob = 1.0 / (1.0 + np.exp(-score))
    #         y = ex.label
    #
    #         # 计算梯度： (概率 - 真实标签) * 词频
    #         # 对每个词： 更新权重
    #         for idx, value in feats.items():
    #             weights[idx] -= lr * (prob - y) * value

    # ------------------------------------------------------------
    # 提交版本是优化过的代码
    # 优化写法（NumPy 稀疏向量化）：把每个样本的 (idx, value) 转成 numpy 数组
    # - score 用 np.dot 一次算完
    # - 更新用 weights[idxs] 一次性完成（高级索引）
    # - 预先缓存特征，避免每个 epoch 反复 extract_features
    # ------------------------------------------------------------
    cached_feats = []
    for ex in train_exs:
        feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        idxs = np.fromiter(feats.keys(), dtype=np.int64)
        vals = np.fromiter(feats.values(), dtype=np.float64)
        cached_feats.append((idxs, vals, ex.label))

    # SGD：训练多轮（num_epochs），每轮 shuffle
    order = np.arange(len(cached_feats))
    for epoch in range(num_epochs):
        np.random.shuffle(order)

        # 训练模型，每一句：
        for i in order:
            idxs, vals, y = cached_feats[i]

            # 计算分数： 权重 * 词频（向量化）
            score = float(np.dot(weights[idxs], vals))

            # 最终sigmoid函数得到概率
            # 可选数值稳定：score = np.clip(score, -20, 20)
            prob = 1.0 / (1.0 + np.exp(-score))

            # 计算梯度： (概率 - 真实标签) * 词频；对每个词： 更新权重（向量化）
            weights[idxs] -= lr * (prob - y) * vals

    # optional: compute training loss / dev accuracy for debugging
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(args, train_exs, feat_extractor)
    return model

class DANNetwork(nn.Module):
    """
    Deep Averaging Network: 平均词向量 -> MLP -> log-prob over {0,1}
    """
    def __init__(self, embedding_layer: nn.Embedding, emb_dim: int, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = embedding_layer
        self.ff = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, word_indices: torch.LongTensor) -> torch.Tensor:
        """
        word_indices: shape [n_tokens]
        returns: log_probs shape [2]
        """
        if word_indices.numel() == 0:
            # 极端情况：空句子就用 0 向量
            emb_dim = self.embedding.weight.shape[1]
            avg = torch.zeros(emb_dim, dtype=torch.float32, device=self.embedding.weight.device)
        else:
            embs = self.embedding(word_indices)          # [n_tokens, emb_dim]
            avg = embs.mean(dim=0)                       # [emb_dim]
        logits = self.ff(avg)                            # [2]
        return self.log_softmax(logits)                  # [2]


class TextCNNNetwork(nn.Module):
    """
    Simple TextCNN: embeddings -> 1D conv over time -> global max pooling -> MLP -> log-prob over {0,1}
    """
    def __init__(self, embedding_layer: nn.Embedding, emb_dim: int, num_filters: int = 64, dropout: float = 0.3):
        super().__init__()
        self.embedding = embedding_layer
        self.kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=k) for k in self.kernel_sizes]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(self.kernel_sizes), 2)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, word_indices: torch.LongTensor) -> torch.Tensor:
        if word_indices.numel() == 0:
            # Empty input fallback
            logits = self.classifier(torch.zeros(self.convs[0].out_channels * len(self.kernel_sizes), dtype=torch.float32))
            return self.log_softmax(logits)

        embs = self.embedding(word_indices)              # [seq_len, emb_dim]
        x = embs.transpose(0, 1).unsqueeze(0)            # [1, emb_dim, seq_len]

        pooled = []
        seq_len = x.shape[-1]
        for conv, k in zip(self.convs, self.kernel_sizes):
            # If sentence is shorter than kernel size, pad on the right so conv is still valid
            conv_in = x if seq_len >= k else F.pad(x, (0, k - seq_len))
            h = F.relu(conv(conv_in))                    # [1, num_filters, T]
            p = torch.max(h, dim=2).values               # [1, num_filters]
            pooled.append(p)
        features = torch.cat(pooled, dim=1).squeeze(0)  # [num_filters * num_kernels]
        logits = self.classifier(features)               # [2]
        return self.log_softmax(logits)                  # [2]


def _words_to_indices(words: List[str], word_embeddings: WordEmbeddings) -> torch.LongTensor:
    unk_idx = word_embeddings.word_indexer.index_of("UNK")
    idxs = []
    for w in words:
        wi = word_embeddings.word_indexer.index_of(w)
        idxs.append(wi if wi != -1 else unk_idx)
    return torch.tensor(idxs, dtype=torch.long)


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network: nn.Module, word_embeddings: WordEmbeddings):
        self.network = network
        self.word_embeddings = word_embeddings



    def predict(self, ex_words: List[str]) -> int:
        self.network.eval()
        with torch.no_grad():
            idxs = _words_to_indices(ex_words, self.word_embeddings)
            log_probs = self.network(idxs)
            pred = int(torch.argmax(log_probs).item())
            return pred


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # 1) Embedding layer（默认用 pretrained + 冻结；更快、更稳）
    emb_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    emb_dim = word_embeddings.get_embedding_length()

    # 2) Network / Loss / Optimizer
    network = DANNetwork(
        embedding_layer=emb_layer,
        emb_dim=emb_dim,
        hidden_size=args.hidden_size,
        dropout=0.2
    )

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    # 3) 训练循环（batch_size=1）
    for epoch in range(args.num_epochs):
        network.train()
        random.shuffle(train_exs)

        total_loss = 0.0
        for ex in train_exs:
            idxs = _words_to_indices(ex.words, word_embeddings)
            gold = torch.tensor([ex.label], dtype=torch.long)  # shape [1]

            optimizer.zero_grad()
            log_probs = network(idxs).unsqueeze(0)             # shape [1, 2]
            loss = loss_fn(log_probs, gold)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        # 可选：每个 epoch 打印一下 loss（以及 dev acc）
        # 用框架的 evaluate 会调用 predict_all -> predict，比较慢但简单；先跑通为主
        avg_loss = total_loss / max(1, len(train_exs))
        print(f"[DAN] epoch {epoch+1}/{args.num_epochs}  avg_train_loss={avg_loss:.4f}")

    return NeuralSentimentClassifier(network, word_embeddings)


def train_text_cnn(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Additional architecture for Part 2 exploration: simple TextCNN.
    """
    emb_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    emb_dim = word_embeddings.get_embedding_length()

    # Reuse hidden_size as number of filters per kernel for convenience.
    network = TextCNNNetwork(
        embedding_layer=emb_layer,
        emb_dim=emb_dim,
        num_filters=args.hidden_size,
        dropout=0.3
    )

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        network.train()
        random.shuffle(train_exs)
        total_loss = 0.0

        for ex in train_exs:
            idxs = _words_to_indices(ex.words, word_embeddings)
            gold = torch.tensor([ex.label], dtype=torch.long)

            optimizer.zero_grad()
            log_probs = network(idxs).unsqueeze(0)      # [1, 2]
            loss = loss_fn(log_probs, gold)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_exs))
        print(f"[CNN] epoch {epoch+1}/{args.num_epochs}  avg_train_loss={avg_loss:.4f}")

    return NeuralSentimentClassifier(network, word_embeddings)
