# Tables 6/7 recomputed on the parent-disjoint model (auto-generated)

Normalized Hamming fraction = mismatches / L (L = 50). train = a_train (generator training parents),
test = held-out test parents, held-out = the 1,000-window test_sample baseline.

## Table 6: real-data baselines (1,000 windows, seed 0)

| set | n | exact unique | nearest generated | pairwise generated | pairwise median |
| --- | --- | --- | --- | --- | --- |
| full_sample | 1000 | 1.0000 | 0.5219 | 0.7377 | 0.7400 |
| a_train_sample | 1000 | 1.0000 | 0.5200 | 0.7361 | 0.7400 |
| a_val_sample | 1000 | 1.0000 | 0.5194 | 0.7385 | 0.7400 |
| test_sample | 1000 | 1.0000 | 0.5219 | 0.7381 | 0.7400 |

Addition (not in old Table 6): distance of the same 1,000 real windows to a_train.

| set | n | nearest a_train (addition) | novel vs a_train (addition) |
| --- | --- | --- | --- |
| a_val_sample | 1000 | 0.4578 | 1.0000 |
| test_sample | 1000 | 0.4568 | 1.0000 |

## Table 7: generated samples

score mean (guide) = A_train guide regressor, from sample_scores.csv; score mean (oracle) = independent
B_train oracle (reverse-complement averaged), from oracle/oracle_per_sample.csv.

| target | set | n | score mean (guide) | score mean (oracle) | exact unique | novel vs train | novel vs test | nearest train | nearest test | nearest generated | pairwise generated | pairwise / held-out | nearest gen / held-out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1.0000 | guided | 1000 | -1.0015 | -0.9637 | 1.0000 | 1.0000 | 1.0000 | 0.4475 | 0.4740 | 0.4963 | 0.7314 | 0.9910 | 0.9508 |
| -1.0000 | unguided | 1000 | -0.1767 | -0.1715 | 1.0000 | 1.0000 | 1.0000 | 0.4583 | 0.4817 | 0.5221 | 0.7374 | 0.9991 | 1.0003 |
| 1.0000 | guided | 1000 | 1.0004 | 0.9202 | 1.0000 | 1.0000 | 1.0000 | 0.4582 | 0.4818 | 0.5095 | 0.7360 | 0.9971 | 0.9761 |
| 1.0000 | unguided | 1000 | -0.1767 | -0.1715 | 1.0000 | 1.0000 | 1.0000 | 0.4583 | 0.4817 | 0.5221 | 0.7374 | 0.9991 | 1.0003 |

## Table 7 additions: size-matched held-out null and extras

| target | set | n | pairwise / held-out (n-matched) | pairwise pct-rank in n-matched test null | nearest gen / held-out (n-matched) | nearest-gen pct-rank in n-matched test null | nearest-train pct-rank in n-matched test null | nearest a_val | nearest full | novel vs full | guide target MAE | oracle target MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1.0000 | guided | 1000 | 0.9904 | 0.0000 | 0.9498 | 0.0000 | 0.0000 | 0.4842 | 0.4334 | 1.0000 | 0.0328 | 0.1171 |
| -1.0000 | unguided | 1000 | 0.9985 | 0.0270 | 0.9992 | 0.3740 | 0.9620 | 0.4939 | 0.4436 | 1.0000 | 0.8265 | 0.8326 |
| 1.0000 | guided | 1000 | 0.9965 | 0.0000 | 0.9751 | 0.0000 | 0.9560 | 0.4930 | 0.4435 | 1.0000 | 0.0584 | 0.1534 |
| 1.0000 | unguided | 1000 | 0.9985 | 0.0270 | 0.9992 | 0.3740 | 0.9620 | 0.4939 | 0.4436 | 1.0000 | 1.1802 | 1.1737 |

Size-matched test null (random n-window subsets of test, 1000 draws, seed 0):

| n | draws | pairwise mean | pairwise 2.5% | pairwise 97.5% | nearest-gen mean | nearest-gen 2.5% | nearest-gen 97.5% | nearest-a_train mean | nearest-a_train 2.5% | nearest-a_train 97.5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1000 | 0.7386 | 0.7374 | 0.7397 | 0.5225 | 0.5199 | 0.5250 | 0.4563 | 0.4540 | 0.4585 |
