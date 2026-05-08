# OT Transfer Learning Results

| Method | Target Test Accuracy |
|---|---|
| jcpot_mlp | 0.8475 |
| deep_jdot | 0.8475 |
| pretrain_finetune_mlp | 0.8305 |
| strategy_a_mlp | 0.8136 |
| pooled_erm_mlp | 0.7966 |
| target_only_logreg | 0.7288 |
| pooled_erm_logreg | 0.7288 |
| strategy_a_logreg | 0.7288 |
| jcpot_logreg | 0.7288 |

## Per-source solo accuracy (after OT, classifier on that source alone)

- source 0: 0.7288
- source 1: 0.7288
- source 2: 0.7288
- source 3: 0.7288
- source 4: 0.7288
- source 5: 0.7288
- source 6: 0.7288
- source 7: 0.7288
- source 8: 0.7288
- source 9: 0.7288
- source 10: 0.7288
- source 11: 0.7288
- source 12: 0.7288
- source 13: 0.7288
- source 14: 0.7288
- source 15: 0.7288
- source 16: 0.7288
- source 17: 0.7288
- source 18: 0.7288
- source 19: 0.7288
- source 20: 0.7288
- source 21: 0.7288
- source 22: 0.7288

## Source weights (Strategy A, by inverse Wasserstein)

- source 0: weight=0.051  W(s,t)=0.3190
- source 1: weight=0.046  W(s,t)=0.3465
- source 2: weight=0.058  W(s,t)=0.2765
- source 3: weight=0.059  W(s,t)=0.2702
- source 4: weight=0.111  W(s,t)=0.0789
- source 5: weight=0.119  W(s,t)=0.0573
- source 6: weight=0.119  W(s,t)=0.0583
- source 7: weight=0.119  W(s,t)=0.0587
- source 8: weight=0.118  W(s,t)=0.0606
- source 9: weight=0.011  W(s,t)=0.7963
- source 10: weight=0.012  W(s,t)=0.7704
- source 11: weight=0.008  W(s,t)=0.8824
- source 12: weight=0.009  W(s,t)=0.8599
- source 13: weight=0.009  W(s,t)=0.8599
- source 14: weight=0.009  W(s,t)=0.8599
- source 15: weight=0.020  W(s,t)=0.5988
- source 16: weight=0.017  W(s,t)=0.6441
- source 17: weight=0.017  W(s,t)=0.6574
- source 18: weight=0.031  W(s,t)=0.4728
- source 19: weight=0.029  W(s,t)=0.4891
- source 20: weight=0.009  W(s,t)=0.8378
- source 21: weight=0.013  W(s,t)=0.7442
- source 22: weight=0.009  W(s,t)=0.8310
