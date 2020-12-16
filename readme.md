# Upper case sentences predictor

## Steps to reproduce

*make utility needed*

**make install**  - install env and dependencies

**make get-model** - downloads my model

**make train** - trains new model(data folder needed)


## Training

I have CUDA env installed but in my case it promt out of memory error

Training process on CPU's log:

```log
TRAIN epoch # 1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [19:28<00:00,  4.67s/it]
Train loss 0.5733927503824234
INFERENCE: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [03:40<00:00,  1.13it/s]
DEV f05-score = 0.0000, precision = 0.0000, recall = 0.0000 | threshold = 0.90
TRAIN epoch # 2: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [19:18<00:00,  4.64s/it]
Train loss 0.3941761130765081
INFERENCE: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [03:51<00:00,  1.08it/s]
DEV f05-score = 0.0000, precision = 0.0000, recall = 0.0000 | threshold = 0.90
TRAIN epoch # 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [19:23<00:00,  4.65s/it]
Train loss 0.3116109628491104
INFERENCE: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [03:44<00:00,  1.11it/s]
DEV f05-score = 0.0000, precision = 0.0000, recall = 0.0000 | threshold = 0.90
TRAIN epoch # 4: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [19:27<00:00,  4.67s/it]
Train loss 0.11852215618081391
INFERENCE: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [03:44<00:00,  1.11it/s]
DEV f05-score = 0.9852, precision = 0.9939, recall = 0.9516 | threshold = 0.90
TRAIN epoch # 5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [19:33<00:00,  4.70s/it]
Train loss 0.02757496785232797
INFERENCE: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [03:43<00:00,  1.12it/s]
DEV f05-score = 0.9945, precision = 0.9980, recall = 0.9806 | threshold = 0.90
INFERENCE: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [03:41<00:00,  1.13it/s]


```

I've also reduced training examples size 2k was enough to get good results

```log
FINALLY, FOR BEST MODEL saved model path:
DEV f05-score = 0.9945, precision = 0.9980, recall = 0.9806 | threshold = 0.90
TEST f05-score = 0.9924, precision = 0.9940, recall = 0.9862 | threshold = 0.90
```