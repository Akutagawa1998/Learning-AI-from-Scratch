# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
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
            feat_name = f"{word} {next_word}"
            idx = self.indexer.add_and_get_index(feat_name, add_to_indexer)
            if idx != -1:
                feats[idx] += 1
        return feats


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


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



def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # raise Exception("Must be implemented")
    
    
    lr = 0.1
    num_epochs = 10

    # 先构建特征空间
    for ex in train_exs:
        feats = feat_extractor.extract_features(ex.words, add_to_indexer=True)
    weights = np.zeros(len(feat_extractor.get_indexer()))

    # ------------------------------------------------------------
    # This is my maunally written code, which is slower than the numpy version.
    # But it works.
    # Please forgive the Chinese comments, because I want to keep it for my own reference in my blog.
    # 旧写法（list/for 循环逐项累加/更新）：保留并注释掉，方便对照
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
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        raise NotImplementedError


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    raise NotImplementedError
