#!/usr/bin/env python3
"""Plot LR training objective and dev accuracy for different step-size schedules."""

import argparse
import math
import random
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sentiment_data import read_sentiment_examples, SentimentExample
from models import UnigramFeatureExtractor
from utils import Indexer


def sigmoid(score: float) -> float:
    # Clip to avoid overflow in exp.
    score = float(np.clip(score, -20.0, 20.0))
    return 1.0 / (1.0 + math.exp(-score))


def build_cached_features(
    exs: List[SentimentExample], feat_extractor: UnigramFeatureExtractor
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    cached = []
    for ex in exs:
        feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        idxs = np.fromiter(feats.keys(), dtype=np.int64)
        vals = np.fromiter(feats.values(), dtype=np.float64)
        cached.append((idxs, vals, ex.label))
    return cached


def compute_log_likelihood(weights: np.ndarray, cached_feats) -> float:
    ll = 0.0
    for idxs, vals, y in cached_feats:
        score = float(np.dot(weights[idxs], vals))
        if y == 1:
            ll += -np.logaddexp(0.0, -score)
        else:
            ll += -np.logaddexp(0.0, score)
    return ll


def compute_accuracy(weights: np.ndarray, cached_feats) -> float:
    correct = 0
    for idxs, vals, y in cached_feats:
        score = float(np.dot(weights[idxs], vals))
        pred = 1 if sigmoid(score) >= 0.5 else 0
        if pred == y:
            correct += 1
    return correct / len(cached_feats)


def get_lr(schedule: str, lr0: float, epoch: int, step: int, decay: float, decay_every: int) -> float:
    if schedule == "constant":
        return lr0
    if schedule == "decay":
        return lr0 * (decay ** (epoch // max(1, decay_every)))
    if schedule == "inv":
        # Use epoch as t so inverse decay is not overly aggressive at sample-update granularity.
        return lr0 / (1.0 + epoch)
    raise ValueError(f"Unknown schedule: {schedule}")


def run_schedule(
    schedule: str,
    lr0: float,
    epochs: int,
    decay: float,
    decay_every: int,
    train_cached,
    dev_cached,
    x_axis: str,
):
    weights = np.zeros(len(feat_extractor.get_indexer()))
    order = np.arange(len(train_cached))

    train_ll_hist = []
    train_acc_hist = []
    dev_acc_hist = []
    x_points = []

    step = 0
    for epoch in range(epochs):
        np.random.shuffle(order)
        for i in order:
            idxs, vals, y = train_cached[i]
            lr = get_lr(schedule, lr0, epoch, step, decay, decay_every)
            score = float(np.dot(weights[idxs], vals))
            prob = sigmoid(score)
            weights[idxs] -= lr * (prob - y) * vals
            step += 1

        train_ll = compute_log_likelihood(weights, train_cached)
        train_acc = compute_accuracy(weights, train_cached)
        dev_acc = compute_accuracy(weights, dev_cached)
        train_ll_hist.append(train_ll)
        train_acc_hist.append(train_acc)
        dev_acc_hist.append(dev_acc)
        if x_axis == "step":
            x_points.append((epoch + 1) * len(train_cached))
        else:
            x_points.append(epoch + 1)

    return x_points, train_ll_hist, train_acc_hist, dev_acc_hist


def plot_constant_lrs(
    lrs,
    epochs,
    decay,
    decay_every,
    train_cached,
    dev_cached,
    x_axis,
    plot_train_acc,
    out_path,
):
    fig, (ax_train, ax_dev) = plt.subplots(2, 1, figsize=(11, 9))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    train_step_points = np.arange(0, epochs + 1) * len(train_cached)
    dev_epoch_points = np.arange(1, epochs + 1)

    for i, lr0 in enumerate(lrs):
        x_points, train_ll, train_acc, dev_acc = run_schedule(
            "constant",
            lr0,
            epochs,
            decay,
            decay_every,
            train_cached,
            dev_cached,
            x_axis,
        )
        color = color_cycle[i % len(color_cycle)] if color_cycle else None
        label_base = f"lr={lr0:g}"
        # Prepend epoch-0 baseline for train loss (all-zero weights before any update)
        init_ll = compute_log_likelihood(np.zeros(len(feat_extractor.get_indexer())), train_cached)
        train_loss_with_start = [-init_ll] + [-v for v in train_ll]
        ax_train.plot(train_step_points, train_loss_with_start, color=color, marker="o", linewidth=2, label=label_base)
        ax_dev.plot(dev_epoch_points, dev_acc, color=color, marker="o", linewidth=2, label=label_base)

    # Top: train loss with linear step axis from 0
    ax_train.set_title("Train Loss vs Step")
    ax_train.set_ylabel("Train Loss (-Log-Likelihood)")
    ax_train.set_xlabel("Step")
    ax_train.set_xlim(0, epochs * len(train_cached))
    ax_train.grid(alpha=0.2)
    ax_train.legend(title="Learning Rate")

    # Bottom: dev accuracy by epoch
    ax_dev.set_title("Dev Accuracy vs Epoch")
    ax_dev.set_xlabel("Epoch")
    ax_dev.set_ylabel("Dev Accuracy")
    ax_dev.set_xticks(range(1, epochs + 1))
    ax_dev.grid(alpha=0.2)
    ax_dev.legend(title="Learning Rate")

    fig.suptitle("Constant LR Comparison: 0.01 vs 0.1 vs 0.5 vs 1.0", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.txt")
    parser.add_argument("--dev", default="data/dev.txt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--schedule", nargs="+", default=["constant", "decay", "inv"])
    parser.add_argument("--decay", type=float, default=0.5)
    parser.add_argument("--decay_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="lr_schedule_plot.png")
    parser.add_argument("--x_axis", choices=["epoch", "step"], default="step")
    parser.add_argument("--plot_train_acc", action="store_true")
    parser.add_argument("--plot_constant_lrs", dest="plot_constant_lrs", action="store_true")
    parser.add_argument("--no_plot_constant_lrs", dest="plot_constant_lrs", action="store_false")
    parser.add_argument("--lr_list", default="0.01,0.1,0.5,1.0")
    parser.add_argument("--out_constant", default="lr_constant_comparison.png")
    parser.set_defaults(plot_constant_lrs=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_exs = read_sentiment_examples(args.train)
    dev_exs = read_sentiment_examples(args.dev)

    feat_extractor = UnigramFeatureExtractor(Indexer())
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    train_cached = build_cached_features(train_exs, feat_extractor)
    dev_cached = build_cached_features(dev_exs, feat_extractor)

    if args.plot_constant_lrs:
        lrs = [float(x.strip()) for x in args.lr_list.split(",") if x.strip()]
        plot_constant_lrs(
            lrs,
            args.epochs,
            args.decay,
            args.decay_every,
            train_cached,
            dev_cached,
            args.x_axis,
            args.plot_train_acc,
            args.out_constant,
        )

    plt.figure(figsize=(10, 6))

    results = {}
    for sched in args.schedule:
        results[sched] = run_schedule(
            sched,
            args.lr,
            args.epochs,
            args.decay,
            args.decay_every,
            train_cached,
            dev_cached,
            args.x_axis,
        )

    color_map = {
        "constant": "tab:blue",
        "decay": "tab:orange",
        "inv": "tab:green",
    }

    init_weights = np.zeros(len(feat_extractor.get_indexer()))
    init_loss = -compute_log_likelihood(init_weights, train_cached)

    ax = plt.subplot(1, 1, 1)
    for sched in args.schedule:
        x_points, train_ll, train_acc, dev_acc = results[sched]
        color = color_map.get(sched, None)

        x_plot = [0] + x_points
        loss_plot = [init_loss] + [-v for v in train_ll]
        if sched == "constant":
            label = f"constant (lr={args.lr:g})"
        elif sched == "decay":
            label = f"decay (x{args.decay:g} every {args.decay_every} ep)"
        elif sched == "inv":
            label = f"inv (lr/(1+t))"
        else:
            label = sched
        ax.plot(x_plot, loss_plot, color=color, marker="o", linewidth=2, label=label)

    ax.set_title("Unigram: Train Loss vs Step (LR Schedule Comparison)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss (-Log-Likelihood)")
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25)
    ax.legend()

    if args.x_axis == "step":
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(ax.get_xticks())
        ax_top.set_xticklabels([f"{int(t / len(train_cached))}" for t in ax.get_xticks()])
        ax_top.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")
