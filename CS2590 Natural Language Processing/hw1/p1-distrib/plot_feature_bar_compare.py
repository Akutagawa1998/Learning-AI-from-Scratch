#!/usr/bin/env python3
"""Bar-chart comparison for UNIGRAM/BIGRAM/BETTER on train/dev Acc/F1.

Default hyperparameter settings are paired as:
(lr, epochs) in [(0.01,5), (0.05,10), (0.1,20), (0.5,25)].
"""

import argparse
import csv
import random
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from models import train_linear_model
from sentiment_data import read_sentiment_examples


FEATURES = ["UNIGRAM", "BIGRAM", "BETTER"]


def compute_acc_f1(golds, preds):
    if len(golds) != len(preds):
        raise ValueError("gold/pred length mismatch")

    num_correct = 0
    num_pos_correct = 0
    num_pred_pos = 0
    num_gold_pos = 0

    for g, p in zip(golds, preds):
        if g == p:
            num_correct += 1
        if p == 1:
            num_pred_pos += 1
        if g == 1:
            num_gold_pos += 1
        if g == 1 and p == 1:
            num_pos_correct += 1

    acc = num_correct / len(golds) if golds else 0.0
    prec = num_pos_correct / num_pred_pos if num_pred_pos > 0 else 0.0
    rec = num_pos_correct / num_gold_pos if num_gold_pos > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, f1


def evaluate_model(model, exs):
    golds = [ex.label for ex in exs]
    preds = model.predict_all([ex.words for ex in exs])
    return compute_acc_f1(golds, preds)


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/train.txt")
    parser.add_argument("--dev_path", default="data/dev.txt")
    parser.add_argument("--lrs", default="0.01,0.05,0.1,0.5")
    parser.add_argument("--epochs", default="5,10,20,25")
    parser.add_argument("--pair_mode", choices=["zip", "cross"], default="zip",
                        help="zip: pair lr[i] with epochs[i]; cross: full Cartesian product")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="feature_bar_compare.png")
    parser.add_argument("--table_out", default="feature_bar_compare.csv")
    parser.add_argument("--markdown_out", default="feature_bar_compare.md")
    args = parser.parse_args()

    lrs = parse_float_list(args.lrs)
    epochs = parse_int_list(args.epochs)

    if args.pair_mode == "zip":
        if len(lrs) != len(epochs):
            raise ValueError("For pair_mode=zip, --lrs and --epochs must have same length")
        configs = list(zip(lrs, epochs))
    else:
        configs = [(lr, ep) for lr in lrs for ep in epochs]

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_exs = read_sentiment_examples(args.train_path)
    dev_exs = read_sentiment_examples(args.dev_path)

    # metrics[metric_name][feat_idx, config_idx]
    metrics = {
        "train_acc": np.zeros((len(FEATURES), len(configs))),
        "dev_acc": np.zeros((len(FEATURES), len(configs))),
        "train_f1": np.zeros((len(FEATURES), len(configs))),
        "dev_f1": np.zeros((len(FEATURES), len(configs))),
    }

    for c_idx, (lr, num_epochs) in enumerate(configs):
        print(f"Running config {c_idx + 1}/{len(configs)}: lr={lr}, epochs={num_epochs}")
        for f_idx, feat_name in enumerate(FEATURES):
            run_args = SimpleNamespace(
                model="LR",
                feats=feat_name,
                lr=lr,
                num_epochs=num_epochs,
            )
            model = train_linear_model(run_args, train_exs, dev_exs)
            train_acc, train_f1 = evaluate_model(model, train_exs)
            dev_acc, dev_f1 = evaluate_model(model, dev_exs)

            metrics["train_acc"][f_idx, c_idx] = train_acc
            metrics["dev_acc"][f_idx, c_idx] = dev_acc
            metrics["train_f1"][f_idx, c_idx] = train_f1
            metrics["dev_f1"][f_idx, c_idx] = dev_f1

    # Export tabular results for report usage.
    rows = []
    for c_idx, (lr, ep) in enumerate(configs):
        for f_idx, feat_name in enumerate(FEATURES):
            rows.append({
                "lr": lr,
                "epoch": ep,
                "feature": feat_name,
                "train_acc": float(metrics["train_acc"][f_idx, c_idx]),
                "train_f1": float(metrics["train_f1"][f_idx, c_idx]),
                "dev_acc": float(metrics["dev_acc"][f_idx, c_idx]),
                "dev_f1": float(metrics["dev_f1"][f_idx, c_idx]),
            })

    with open(args.table_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lr", "epoch", "feature", "train_acc", "train_f1", "dev_acc", "dev_f1"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved table to {args.table_out}")

    with open(args.markdown_out, "w") as f:
        f.write("| lr | epoch | feature | train_acc | train_f1 | dev_acc | dev_f1 |\\n")
        f.write("|---:|------:|:--------|----------:|---------:|--------:|-------:|\\n")
        for r in rows:
            f.write(
                f"| {r['lr']} | {r['epoch']} | {r['feature']} | "
                f"{r['train_acc']:.4f} | {r['train_f1']:.4f} | {r['dev_acc']:.4f} | {r['dev_f1']:.4f} |\\n"
            )
    print(f"Saved markdown table to {args.markdown_out}")

    colors = {
        "UNIGRAM": "tab:blue",
        "BIGRAM": "tab:orange",
        "BETTER": "tab:green",
    }

    # For full Cartesian runs, use one subplot per (lr, epoch) configuration.
    if args.pair_mode == "cross":
        nrows = len(epochs)
        ncols = len(lrs)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
        axes = np.array(axes).reshape(nrows, ncols)

        feat_x = np.arange(len(FEATURES))
        w = 0.18
        metric_items = [
            ("train_acc", "tab:blue"),
            ("dev_acc", "tab:orange"),
            ("train_f1", "tab:green"),
            ("dev_f1", "tab:red"),
        ]

        cfg_to_idx = {(lr, ep): i for i, (lr, ep) in enumerate(configs)}
        for r, ep in enumerate(epochs):
            for c, lr in enumerate(lrs):
                ax = axes[r, c]
                idx = cfg_to_idx[(lr, ep)]
                for m_i, (metric_key, metric_color) in enumerate(metric_items):
                    vals = [metrics[metric_key][f_idx, idx] for f_idx in range(len(FEATURES))]
                    ax.bar(feat_x + (m_i - 1.5) * w, vals, width=w, color=metric_color, alpha=0.85)
                ax.set_title(f"lr={lr}, ep={ep}", fontsize=9)
                ax.set_xticks(feat_x)
                ax.set_xticklabels(FEATURES, rotation=20, ha="right", fontsize=8)
                ax.set_ylim(0.0, 1.0)
                ax.grid(axis="y", linestyle="--", alpha=0.25)

        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in metric_items]
        legend_labels = [name for name, _ in metric_items]
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4)
        fig.suptitle("Cross Grid (4x4): each subplot is one (lr, epoch) configuration", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        cfg_labels = [f"lr={lr}\nep={ep}" for lr, ep in configs]
        x = np.arange(len(configs))
        width = 0.24

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        plot_items = [
            ("train_acc", "Train Accuracy", axes[0, 0]),
            ("dev_acc", "Dev Accuracy", axes[0, 1]),
            ("train_f1", "Train F1", axes[1, 0]),
            ("dev_f1", "Dev F1", axes[1, 1]),
        ]

        for key, title, ax in plot_items:
            for f_idx, feat_name in enumerate(FEATURES):
                offset = (f_idx - 1) * width
                ax.bar(x + offset, metrics[key][f_idx], width=width, color=colors[feat_name], label=feat_name)
            ax.set_title(title)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(cfg_labels)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(cfg_labels)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle("Feature Comparison: UNIGRAM vs BIGRAM vs BETTER")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(args.out, dpi=160)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
