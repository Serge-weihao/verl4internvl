# verl4internvl
基于verl支持Internvl纯文本-图文混合训练的repo，同时支持FSDP下序列并行等功能、MoE负载均衡监控以及GRPO、DAPO、GSPO

Supporting  image-text and pure text mixed training for InternVL2.5,  it also supports  functionalities including sequence parallelism under FSDP, MoE load balancing monitoring, as well as GRPO, DAPO, and GSPO.

## Training

recipe/dapo/internvlgspo.sh

## OpenCompass Reasoning Evaluation

| 数据集 | DAPO-GSPO  | BASE Model(Internvl2.5-MPO) |
| ---- | -------- | -------- |
| DynaMath | 11.0 | 9.8 |
| LogicVista | 40.0 | 39.4 |
| MathVerse_MINI_Vision_Only | 33.1 | 26.9 |
| MathVision | 23.5 | 21.8 |
| MathVista_MINI | 65.0 | 64.5 |
| WeMath | 18.5 | 18.1 |
| 平均得分 | 31.8 | 30.1 |

