---
library_name: transformers
license: apache-2.0
base_model: microsoft/swinv2-large-patch4-window12-192-22k
tags:
- image-classification
- vision
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: SWIN_Gaudi_v1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# SWIN_Gaudi_v1

This model is a fine-tuned version of [microsoft/swinv2-large-patch4-window12-192-22k](https://huggingface.co/microsoft/swinv2-large-patch4-window12-192-22k) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 4.8090
- Accuracy: 0.2293
- Memory Allocated (gb): 1.84
- Max Memory Allocated (gb): 58.8
- Total Memory Available (gb): 94.62

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 128
- eval_batch_size: 128
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- total_train_batch_size: 1024
- total_eval_batch_size: 1024
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-06
- lr_scheduler_type: linear
- num_epochs: 30.0

### Training results

| Training Loss | Epoch | Step  | Accuracy | Validation Loss | Memory Allocated (gb) | Allocated (gb) | Memory Available (gb) |
|:-------------:|:-----:|:-----:|:--------:|:---------------:|:---------------------:|:--------------:|:---------------------:|
| 5.3071        | 1.0   | 657   | 0.0801   | 6.2406          | 60.21                 | 3.2            | 94.62                 |
| 3.1366        | 2.0   | 1314  | 0.1047   | 5.8481          | 60.21                 | 3.2            | 94.62                 |
| 2.6048        | 3.0   | 1971  | 0.1238   | 5.5522          | 60.21                 | 3.2            | 94.62                 |
| 1.9918        | 4.0   | 2628  | 0.1301   | 5.5551          | 60.21                 | 3.2            | 94.62                 |
| 1.8353        | 5.0   | 3285  | 0.1415   | 5.4142          | 60.21                 | 3.2            | 94.62                 |
| 1.7262        | 6.0   | 3942  | 0.1495   | 5.4061          | 60.21                 | 3.2            | 94.62                 |
| 1.5135        | 7.0   | 4599  | 0.1468   | 5.4261          | 60.21                 | 3.2            | 94.62                 |
| 1.4225        | 8.0   | 5256  | 0.1573   | 5.3333          | 60.21                 | 3.2            | 94.62                 |
| 1.354         | 9.0   | 5913  | 0.1638   | 5.2205          | 60.21                 | 3.2            | 94.62                 |
| 1.2511        | 10.0  | 6570  | 0.1708   | 5.2129          | 60.21                 | 3.2            | 94.62                 |
| 1.1742        | 11.0  | 7227  | 0.1724   | 5.2002          | 60.21                 | 3.2            | 94.62                 |
| 1.1342        | 12.0  | 7884  | 0.1782   | 5.1635          | 60.21                 | 3.2            | 94.62                 |
| 1.0711        | 13.0  | 8541  | 0.1779   | 5.1436          | 60.21                 | 3.2            | 94.62                 |
| 0.9971        | 14.0  | 9198  | 0.1817   | 5.1076          | 60.21                 | 3.2            | 94.62                 |
| 0.9774        | 15.0  | 9855  | 0.1935   | 4.9076          | 60.21                 | 3.2            | 94.62                 |
| 0.9174        | 16.0  | 10512 | 0.1890   | 5.0318          | 60.21                 | 3.2            | 94.62                 |
| 0.8675        | 17.0  | 11169 | 0.1951   | 5.0392          | 60.21                 | 3.2            | 94.62                 |
| 0.8499        | 18.0  | 11826 | 0.1978   | 5.0243          | 60.21                 | 3.2            | 94.62                 |
| 0.8262        | 19.0  | 12483 | 0.1972   | 5.0843          | 60.21                 | 3.2            | 94.62                 |
| 0.7623        | 20.0  | 13140 | 0.2048   | 5.0004          | 60.21                 | 3.2            | 94.62                 |
| 0.7481        | 21.0  | 13797 | 0.2132   | 4.8428          | 60.24                 | 3.2            | 94.62                 |
| 0.7284        | 22.0  | 14454 | 0.2149   | 4.8461          | 60.24                 | 3.2            | 94.62                 |
| 0.6834        | 23.0  | 15111 | 0.2159   | 4.8741          | 60.24                 | 3.2            | 94.62                 |
| 0.6591        | 24.0  | 15768 | 0.2187   | 4.8993          | 60.24                 | 3.2            | 94.62                 |
| 0.6447        | 25.0  | 16425 | 0.2196   | 4.8415          | 60.24                 | 3.2            | 94.62                 |
| 0.6107        | 26.0  | 17082 | 0.2216   | 4.8600          | 60.24                 | 3.2            | 94.62                 |
| 0.5958        | 27.0  | 17739 | 0.2245   | 4.8391          | 60.24                 | 3.2            | 94.62                 |
| 0.5836        | 28.0  | 18396 | 0.2265   | 4.8561          | 60.24                 | 3.2            | 94.62                 |
| 0.5547        | 29.0  | 19053 | 0.2295   | 4.7933          | 60.24                 | 3.2            | 94.62                 |
| 0.547         | 30.0  | 19710 | 0.2293   | 4.8090          | 60.24                 | 3.2            | 94.62                 |


### Framework versions

- Transformers 4.45.2
- Pytorch 2.6.0+hpu_1.20.0-543.git4952fce
- Datasets 3.5.0
- Tokenizers 0.20.3
