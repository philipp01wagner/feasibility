# OT Transfer Learning Pipeline (Tabular, Few-Shot Target)

Multi-source transfer learning for tabular classification using Optimal Transport.
Setup: N labeled source tasks + target task (N+1) with <100 labels.

## Install

```bash
pip install pot scikit-learn numpy torch geomloss pandas
```

## Run

```bash
python data.py            # generates synthetic source/target datasets
python baselines.py       # target-only, pooled ERM, pooled+finetune
python ot_strategy_a.py   # per-source OT transport, then pool
python ot_jcpot.py        # multi-source JCPOT
python deep_jdot.py       # end-to-end DeepJDOT (PyTorch)
python evaluate.py        # collects all results into report.md
```

## Plug in your own data

Replace `load_data()` in `data.py` with your loader. It must return:

- `Xs_list`: list of N numpy arrays, shape `(n_i, d)` each — source features
- `ys_list`: list of N numpy arrays, shape `(n_i,)` each — source labels (integer class indices)
- `Xt_train_labeled`, `yt_train_labeled`: target labeled set (small, <100)
- `Xt_train_unlabeled`: target unlabeled set (used for OT alignment)
- `Xt_test`, `yt_test`: held-out target test set

All sources and target must share the same `d` (feature dim) and label space.

## What each script does

- `data.py` — synthetic data generator + `load_data()` you replace.
- `features.py` — L2-normalize features (critical before OT).
- `baselines.py` — target-only, pooled ERM, pretrain+finetune. **If these win, stop.**
- `ot_strategy_a.py` — Step 4 from recipe. Per-source `SinkhornL1l2Transport` to target, weight sources by inverse Wasserstein distance, train classifier on pooled transported features.
- `ot_jcpot.py` — Step 5. POT's `JCPOTTransport` (joint-distribution OT for multi-source under target shift).
- `deep_jdot.py` — Step 6. PyTorch encoder + classifier, trained end-to-end with classification loss + Sinkhorn alignment loss + JDOT label term.
- `evaluate.py` — runs all of the above, builds comparison table.

## Outputs

- `results.json` — accuracy per method
- `report.md` — markdown table ranking methods
