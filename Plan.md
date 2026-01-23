# Learning AI from Scratch - Plan

This plan defines a week-by-week path to build a reproducible training stack,
scale to multi-GPU, and progress from MNIST to a small LM pipeline.

## Week 1: Training scaffold v0 (single GPU first, then DDP later)
Goal: build a reproducible, diagnosable, extensible training loop for MNIST/CIFAR scale.
Deliverables: `train.py` + config + `REPRODUCE.md` + week01 results (3 seeds).

Day 1 (Environment and repo skeleton)
- Focus: repo layout and empty entrypoint that runs.
- Tasks: create folder skeleton, add minimal `train.py` or `main.py`, write a 1-line goal in `README.md`.
- Tasks: create conda env `dl`, install torch, torchvision, tensorboard or wandb, pyyaml or omegaconf.
- Outputs: `README.md`, `train.py`, `results/week01/notes.md` with OS/CUDA/driver/GPU.
- Checks: `python train.py` runs and exits cleanly.

Day 2 (Data pipeline: MNIST)
- Focus: controllable data loading with explicit shapes.
- Tasks: implement `src/data/mnist.py` with train/val dataloaders and transforms.
- Tasks: log and record X.shape, y.shape, dtype, min/max, normalization.
- Outputs: `src/data/mnist.py`, notes describing (B,1,28,28) vs (B,784).
- Checks: a batch prints shapes and value range as expected.

Day 3 (Minimal model: Softmax Regression)
- Focus: a learnable baseline that reduces loss.
- Tasks: implement `src/models/linear.py`, use `CrossEntropyLoss`.
- Tasks: run 1 epoch with a small batch size and log loss per step.
- Outputs: `results/week01/metrics.csv` with epoch train loss and val acc.
- Checks: loss decreases over steps; val acc > random.

Day 4 (Training loop: ckpt + eval)
- Focus: complete train/eval loop with best checkpoint.
- Tasks: write `src/train/trainer.py` with train/eval phases and metric logging.
- Tasks: save best checkpoint by val acc or val loss.
- Outputs: `REPRODUCE.md` initial, checkpoint file, metrics logs.
- Checks: best checkpoint reloads and reproduces val metrics.

Day 5 (Reproducibility: seed + config)
- Focus: deterministic-ish runs using config.
- Tasks: fix random seeds for python/numpy/torch/cuda and set deterministic flags.
- Tasks: add config file for lr, batch size, epochs, seed, data path.
- Outputs: `configs/mnist_baseline.yaml`, updated `results/week01/notes.md`.
- Checks: three runs show consistent trends and documented variance.

Day 6 (Minimal control: learning rate)
- Focus: one-variable ablation (lr).
- Tasks: run lr=1e-1, 1e-2, 1e-3 with short epochs.
- Tasks: log metrics to a single CSV and plot curves.
- Outputs: updated `results/week01/metrics.csv`, `plots/` learning curves.
- Checks: identify under/over-shoot lr and summarize.

Day 7 (Week cleanup and tag)
- Focus: make week01 reproducible and readable.
- Tasks: clean repo, unify logging format, add brief comments where needed.
- Tasks: update `README.md` and `REPRODUCE.md` for fresh start.
- Outputs: clean tree, updated docs, tag `v0.1`.
- Checks: new env can follow docs and reach week01 results.

## Week 2: Multi-GPU (DDP) + AMP + profiling (use 4x A100 well)
Goal: same script runs on 1 GPU or 4 GPUs; quantify throughput/memory/comm bottleneck.
Deliverables: DDP run + AMP run + throughput/memory benchmark table.

Day 8 (DDP minimal run)
- Focus: DDP entrypoint and sampler correctness.
- Tasks: add `torchrun` support, init process group, wrap model in DDP.
- Tasks: use `DistributedSampler` and ensure each rank sees unique data.
- Outputs: `configs/mnist_ddp.yaml`, `scripts/slurm/mnist_ddp.sh` if needed.
- Checks: logs show rank/world_size and global batch calculation.

Day 9 (DDP correctness check)
- Focus: verify DDP matches single-GPU behavior.
- Tasks: run single-GPU and 4-GPU with same global batch/lr policy.
- Tasks: compare loss curve shape and final val acc.
- Outputs: `results/week02/metrics.csv`, notes explaining differences.
- Checks: curves are similar and accuracy gap is small.

Day 10 (AMP: BF16/FP16)
- Focus: enable mixed precision safely.
- Tasks: add AMP with `autocast` and `GradScaler` or BF16 path.
- Tasks: measure samples/s and peak memory.
- Outputs: `configs/mnist_ddp_amp.yaml`, throughput table in `results/week02/`.
- Checks: AMP run is stable and faster than FP32.

Day 11 (Gradient accumulation + effective batch)
- Focus: control effective batch size.
- Tasks: implement grad accumulation and log micro/global batch.
- Tasks: compare no-accum vs accum (same global batch).
- Outputs: notes with batch formulas and tuning guidance.
- Checks: metrics match across accumulation settings.

Day 12 (Profiling: step time breakdown)
- Focus: identify data/compute/optimizer bottlenecks.
- Tasks: time dataloader, forward/backward, optimizer step.
- Tasks: optional torch profiler for a few steps.
- Outputs: `results/week02/profile.md` with table and summary.
- Checks: a clear bottleneck is identified.

Day 13 (Minimal system ablation)
- Focus: system-level tradeoffs.
- Tasks: run 4 configs varying batch, accum, precision.
- Tasks: track throughput, memory, and stability.
- Outputs: `results/week02/bench.csv` and short recommendation.
- Checks: pick one "best" config with evidence.

Day 14 (Week summary: training scaffold v1)
- Focus: consolidate DDP/AMP/profiling into mainline.
- Tasks: refactor code paths, remove ad-hoc hacks.
- Tasks: update `REPRODUCE.md` for single vs multi GPU.
- Outputs: tag `v0.2`, final week02 docs.
- Checks: both single-GPU and DDP runs work with same entrypoint.

## Week 3: From MNIST to CIFAR10 (training principles)
Goal: transferable training habits (optimizer/init/norm/regularization).
Deliverables: CIFAR10 baseline + 3 rigorous ablations.

Day 15 (CIFAR10 data + baseline CNN)
- Focus: CIFAR10 data and a simple CNN baseline.
- Tasks: implement `src/data/cifar.py` with transforms and normalization.
- Tasks: implement `src/models/cnn_small.py` and verify output shapes.
- Outputs: working CIFAR10 baseline run.
- Checks: loss decreases and val acc > random.

Day 16 (Optimizer ablation: SGD vs AdamW)
- Focus: optimizer differences with same schedule.
- Tasks: run SGD (with momentum) and AdamW under same settings.
- Tasks: log convergence speed and final val acc.
- Outputs: comparison table and notes.
- Checks: explain which optimizer is better and why.

Day 17 (Initialization ablation: default vs He/Xavier)
- Focus: initialization effects on stability.
- Tasks: add init option to model config.
- Tasks: run default and He/Xavier, log loss stability.
- Outputs: comparison curves and brief conclusion.
- Checks: note any divergence or slow start.

Day 18 (Normalization ablation: BN or LN)
- Focus: train/eval behavior differences.
- Tasks: implement BN or LN variant in CNN.
- Tasks: compare training speed and val accuracy.
- Outputs: curves and failure mode notes.
- Checks: confirm eval uses correct running stats.

Day 19 (Regularization ablation)
- Focus: weight decay, dropout, augmentation effects.
- Tasks: run at least two regularization variants.
- Tasks: collect >=30 error cases and categorize.
- Outputs: error analysis summary and metrics table.
- Checks: link errors to overfitting or underfitting signals.

Day 20 (Week3 mini-report)
- Focus: write a short ablation report.
- Tasks: summarize experiments, tables, and key plots.
- Outputs: `results/week03/report.md`.
- Checks: report is reproducible and references configs.

Day 21 (Cleanup + tag v0.3)
- Focus: clean, stable CIFAR10 baseline.
- Tasks: refactor and remove dead code.
- Outputs: tag `v0.3`.
- Checks: CIFAR10 baseline can run from clean env.

## Week 4: Transformer components (prepare for LLM)
Goal: implement decoder-only block with unit tests.
Deliverables: attention, MHA, block, causal mask, stable softmax, minimal LM data flow.

Day 22 (QKV shape derivation + attention forward)
- Focus: correct attention math and shapes.
- Tasks: implement attention forward with shape assertions.
- Outputs: attention module and a shape test.
- Checks: test passes for multiple batch/seq sizes.

Day 23 (Causal mask + numerical stability)
- Focus: correct causal masking and stable softmax.
- Tasks: add causal mask and safe softmax (subtract max).
- Outputs: mask utilities and test for mask correctness.
- Checks: future positions are masked in output.

Day 24 (Multi-head attention)
- Focus: correct head splitting/merging.
- Tasks: implement MHA with head dimension checks.
- Outputs: MHA module and tests.
- Checks: output shape matches input.

Day 25 (FFN + residual + (pre-)LayerNorm)
- Focus: Transformer block wiring.
- Tasks: implement FFN, residual, and optional pre-LN.
- Outputs: block module and unit test.
- Checks: gradients flow and shapes match.

Day 26 (Char-level LM dataset + next-token loss)
- Focus: minimal LM data pipeline.
- Tasks: build char dataset, tokenize to ids, make input/target pairs.
- Outputs: dataset module and loader test.
- Checks: target is input shifted by 1.

Day 27 (Tiny decoder-only LM run)
- Focus: make loss decrease on tiny dataset.
- Tasks: run short training and sample text periodically.
- Outputs: metrics log and sample outputs.
- Checks: loss trends down and samples improve.

Day 28 (Week summary + unit tests + tag v0.4)
- Focus: stabilize component tests and docs.
- Tasks: clean tests, add README for module usage.
- Outputs: tag `v0.4`.
- Checks: all tests pass on CPU.

## Week 5: Toy LM to reusable pipeline (Tokenizer + pack + ppl)
Goal: minimal pretrain pipeline: tokenize->pack->train->ppl eval->ckpt.
Deliverables: tokenizer script, pack script, ppl eval script, throughput benchmark.

Day 29 (Tokenizer integration)
- Focus: robust encode/decode and special tokens.
- Tasks: use HF tokenizer, define BOS/EOS/PAD.
- Outputs: tokenizer script and quick encode/decode test.
- Checks: decode(encode(x)) matches x for basic samples.

Day 30 (Pack fixed-length blocks)
- Focus: reduce padding and improve efficiency.
- Tasks: implement packer that concatenates and chunks.
- Outputs: pack script and stats on padding ratio.
- Checks: packed sequences have correct length.

Day 31 (PPL eval)
- Focus: stable evaluation with best checkpoint.
- Tasks: add eval dataloader and perplexity computation.
- Outputs: ppl evaluation script and metrics.
- Checks: ppl reported for fixed val set.

Day 32 (Throughput/memory benchmark)
- Focus: measure throughput vs seq_len and batch size.
- Tasks: run short benchmark for different seq_len/micro-batch.
- Outputs: benchmark table.
- Checks: record tokens/s and peak memory.

Day 33 (Warmup/clip/accum ablation)
- Focus: stabilization knobs.
- Tasks: run short comparisons for warmup, grad clip, accumulation.
- Outputs: ablation table and short notes.
- Checks: identify settings that prevent divergence.

Day 34 (Pipeline doc)
- Focus: document the full pretrain pipeline.
- Tasks: write data->tok->train->eval doc with commands.
- Outputs: pipeline doc file.
- Checks: a reader can reproduce the pipeline.

Day 35 (Tag v0.5)
- Focus: freeze a usable toy-LM pipeline.
- Outputs: tag `v0.5`.
- Checks: all scripts run on a clean env.

## Week 6: Use 4x A100 effectively (short experiments)
Goal: budgeted comparisons for data/model/training strategy.
Deliverables: budgeted ablation + "budget decision memo".

Day 36 (Define training budget)
- Focus: choose tokens or wall-clock budget.
- Tasks: define budget and logging for time/tokens.
- Outputs: budget definition notes.
- Checks: budget can be enforced in scripts.

Day 37 (Data versioning manifest)
- Focus: reproducible dataset tracking.
- Tasks: create manifest with dataset hash, size, filters.
- Outputs: data manifest file.
- Checks: manifest describes data source and filtering.

Day 38 (Minimal dedup/quality filter)
- Focus: A/B dataset creation.
- Tasks: implement simple dedup or quality filter and create A/B.
- Outputs: dataset A/B stats and notes.
- Checks: A/B datasets are clearly different.

Day 39 (Budgeted training A vs B)
- Focus: compare under same budget.
- Tasks: train A and B with same budget and track ppl.
- Outputs: metrics table and sample quality notes.
- Checks: differences are statistically meaningful.

Day 40 (Model size ablation)
- Focus: tradeoff between model size and data.
- Tasks: train small model longer vs larger model shorter.
- Outputs: ablation table with ppl and samples.
- Checks: explain which is better under budget.

Day 41 (Write budget memo)
- Focus: record how to decide under budget.
- Tasks: write memo with recommendation and rationale.
- Outputs: budget decision memo.
- Checks: memo references data and runs.

Day 42 (Tag v0.6)
- Focus: freeze week06 results and docs.
- Outputs: tag `v0.6`.
- Checks: results referenced in memo are reproducible.

## Week 7: SFT loop (tag v0.7)
Day 43 (Instruction data + mini eval set)
- Focus: format and evaluation set creation.
- Tasks: define instruction format and labels.
- Outputs: dataset schema and 20-50 eval examples.
- Checks: eval set covers multiple task types.

Day 44 (SFT run)
- Focus: get SFT training to run end-to-end.
- Tasks: implement SFT data loader and training config.
- Outputs: first SFT checkpoint and metrics.
- Checks: loss decreases and eval responds plausibly.

Day 45 (Overfit diagnosis)
- Focus: detect prompt leakage and overfitting.
- Tasks: compare train/val loss, spot memorization.
- Outputs: notes on overfit symptoms.
- Checks: mitigate leakage or data split issues.

Day 46 (Data cleanup + format improvement)
- Focus: improve dataset quality.
- Tasks: remove bad samples and adjust format.
- Outputs: updated dataset version and comparison notes.
- Checks: improvements show in eval.

Day 47 (SFT eval harness)
- Focus: consistent evaluation.
- Tasks: implement eval script for format, correctness, following.
- Outputs: eval harness and baseline results.
- Checks: eval can run on any checkpoint.

Day 48 (Report + REPRODUCE update)
- Focus: record SFT results.
- Tasks: write report and update reproducibility steps.
- Outputs: report and updated `REPRODUCE.md`.
- Checks: report references exact configs.

Day 49 (Tag v0.7)
- Focus: finalize SFT loop.
- Outputs: tag `v0.7`.
- Checks: SFT run reproducible from docs.

## Week 8: LoRA / QLoRA (tag v0.8)
Day 50 (LoRA integration)
- Focus: add PEFT with LoRA modules.
- Tasks: add LoRA config and apply to attention/FFN.
- Outputs: LoRA training run.
- Checks: LoRA trains with reduced memory.

Day 51 (Full FT vs LoRA)
- Focus: compare full fine-tune to LoRA.
- Tasks: run same budget and compare eval metrics.
- Outputs: comparison table and notes.
- Checks: explain quality vs memory tradeoff.

Day 52 (QLoRA integration)
- Focus: 4-bit quantized finetuning if supported.
- Tasks: add QLoRA config and dependencies.
- Outputs: QLoRA run logs.
- Checks: stability and memory usage recorded.

Day 53 (LoRA vs QLoRA)
- Focus: compare quality, memory, speed.
- Tasks: run both under same budget.
- Outputs: summary table.
- Checks: pick a default for later work.

Day 54 (PEFT decision table)
- Focus: write a decision matrix.
- Tasks: summarize when to use FT/LoRA/QLoRA.
- Outputs: decision table doc.
- Checks: table references measurements.

Day 55 (PEFT failure-mode Q&A)
- Focus: document pitfalls.
- Tasks: list common failures and fixes.
- Outputs: Q&A notes.
- Checks: checklist is actionable.

Day 56 (Tag v0.8)
- Focus: freeze PEFT comparison.
- Outputs: tag `v0.8`.
- Checks: PEFT experiments reproducible.

## Week 9: DPO (tag v0.9)
Day 57 (Preference pairs)
- Focus: create chosen/rejected pairs.
- Tasks: build preference dataset and schema.
- Outputs: preference dataset and stats.
- Checks: pairs align with rubric.

Day 58 (DPO run)
- Focus: run DPO training.
- Tasks: implement DPO loss and training script.
- Outputs: DPO checkpoint and logs.
- Checks: training is stable.

Day 59 (Preference alignment eval)
- Focus: measure preference accuracy.
- Tasks: run eval on preference set.
- Outputs: preference accuracy metrics.
- Checks: compare against SFT baseline.

Day 60 (Failure modes)
- Focus: diagnose collapse or overfit.
- Tasks: inspect samples and metrics for collapse.
- Outputs: failure analysis notes.
- Checks: mitigation strategies documented.

Day 61 (SFT-only vs SFT+DPO)
- Focus: end-to-end comparison.
- Tasks: run eval on both models.
- Outputs: comparison table and notes.
- Checks: decision is backed by metrics.

Day 62 (Report)
- Focus: summarize DPO results.
- Tasks: write report with tables and examples.
- Outputs: report file.
- Checks: includes failure analysis.

Day 63 (Tag v0.9)
- Focus: finalize DPO stage.
- Outputs: tag `v0.9`.
- Checks: full DPO pipeline reproducible.

## Week 10: Memory-efficient distributed (FSDP or ZeRO) (tag v1.0)
Day 64 (Choose path + minimal example)
- Focus: decide FSDP vs ZeRO.
- Tasks: run minimal example and measure memory.
- Outputs: decision note and minimal script.
- Checks: memory savings observed.

Day 65 (Port LM/SFT script)
- Focus: integrate chosen method into training.
- Tasks: adapt checkpointing and optimizer states.
- Outputs: working distributed run.
- Checks: loss matches DDP baseline.

Day 66 (DDP vs FSDP/ZeRO)
- Focus: memory/throughput comparison.
- Tasks: run short benchmarks and log metrics.
- Outputs: comparison table.
- Checks: tradeoffs documented.

Day 67 (Pitfalls)
- Focus: common issues checklist.
- Tasks: test ckpt reload, grad accum, bf16, comms.
- Outputs: pitfall notes and fixes.
- Checks: run passes all checks.

Day 68 (Memory structure doc)
- Focus: explain memory components.
- Tasks: write params/grads/opt state/act memory breakdown.
- Outputs: memory structure doc.
- Checks: doc includes formulas and examples.

Day 69 (Report + REPRODUCE update)
- Focus: summarize and document.
- Tasks: write report and update reproducibility steps.
- Outputs: report and updated `REPRODUCE.md`.
- Checks: reproduction verified.

Day 70 (Tag v1.0)
- Focus: finalize distributed memory stage.
- Outputs: tag `v1.0`.
- Checks: distributed runs reproducible.

## Week 11: Inference + efficiency (KV cache, FlashAttention) (tag v1.1)
Day 71 (KV cache)
- Focus: KV cache integration in inference.
- Tasks: add cache usage and verify correctness.
- Outputs: inference script with cache.
- Checks: outputs match no-cache baseline.

Day 72 (Compare with/without KV cache)
- Focus: latency and throughput improvement.
- Tasks: benchmark both modes.
- Outputs: comparison table.
- Checks: report speedup and memory change.

Day 73 (FlashAttention enable)
- Focus: optional performance upgrade.
- Tasks: install and enable if supported.
- Outputs: flash-attn run logs.
- Checks: confirm numerical stability.

Day 74 (Standard attention vs flash-attn)
- Focus: compare performance and memory.
- Tasks: benchmark both paths.
- Outputs: comparison table and notes.
- Checks: decide default attention path.

Day 75 (Inference optimization memo)
- Focus: summarize inference tuning.
- Tasks: write memo of best settings.
- Outputs: optimization memo.
- Checks: memo references measurements.

Day 76 (Organize perf data)
- Focus: consolidate benchmark data.
- Tasks: clean tables and plot key graphs.
- Outputs: tidy results folder.
- Checks: data links to specific runs.

Day 77 (Tag v1.1)
- Focus: freeze inference optimizations.
- Outputs: tag `v1.1`.
- Checks: inference scripts reproducible.

## Week 12: Portfolio packaging (tag v1.2)
Day 78 (Pick flagship project)
- Focus: choose a project for portfolio.
- Tasks: select one pipeline and ensure it is stable.
- Outputs: project selection note.
- Checks: chosen project is reproducible.

Day 79 (Improve REPRODUCE.md)
- Focus: clean reproduction from empty env.
- Tasks: write step-by-step install and run instructions.
- Outputs: updated `REPRODUCE.md`.
- Checks: dry run in a fresh env.

Day 80 (RESULTS.md)
- Focus: present key results.
- Tasks: add tables and plots to `RESULTS.md`.
- Outputs: `RESULTS.md` with summary metrics.
- Checks: each table cites configs.

Day 81 (TALK.md)
- Focus: create a 15-minute narrative.
- Tasks: outline and write a talk track.
- Outputs: `TALK.md`.
- Checks: talk covers setup, results, lessons.

Day 82 (Q&A prompts)
- Focus: prepare for interviews.
- Tasks: write 40 questions with bullet answers.
- Outputs: Q&A doc.
- Checks: answers cite evidence.

Day 83 (Cold-start reproduction drill)
- Focus: simulate a clean reproduction.
- Tasks: run from scratch and record time.
- Outputs: reproduction log.
- Checks: document any issues and fixes.

Day 84 (Release v1.2)
- Focus: final cleanup and release.
- Tasks: polish docs, tag `v1.2`.
- Outputs: final tag and clean repo.
- Checks: all key artifacts are discoverable.
