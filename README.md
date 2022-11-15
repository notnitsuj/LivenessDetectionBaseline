# Liveness Detection Baseline
This repository provides some scripts to easily train and test some [timm](https://github.com/rwightman/pytorch-image-models) models for Zalo AI Challenge 2022: Liveness Detection.

<div align="center">
    <img src="img/livenessdetection.png" width="100%"/>
</div>

Approach:
- Simple image classification problem
- The module consists of a pretrained model from timm with an output size of 1 and a sigmoid layer
- Training and testing data are created by randomly sampling 10 frames from each video
- When training, each frame is fed into the model irrespective of which video it comes from
- When testing, the predictions from all 10 frames of each video are averaged to produce a final prediction

## Dependencies

```
pip3 install -r requirements.txt
```

## Data
Download the training and public test data from the competition portal to folder `data` and extract them. The structure of the folder will be:

```
.
├── data
│   ├── train
│   │   ├── videos
│   │   │   └── *.mp4
│   ├── public_test
│   │   ├── public
│   │   │   ├── videos
│   │   │   │   └── *.mp4
│   ├── public_test_2
│   │   ├── videos
│   │   │   └── *.mp4
```

To prepare image data for training and testing, run:

```
python3 utils/prep_data.py
```

## Logging
This repository uses [Weights & Biases](https://wandb.ai/) for experiment tracking. See [this](https://docs.wandb.ai/quickstart) for how to get started and log in with your account. Then, go to [train.py](train.py) and modify the `wandb.init()` line for your own project.

## Training

To train with a specific model, run:

```
python3 train.py --backbone <any timm model>
```

## Testing

To get predictions for the public test set 1, run

```
python3 infer.py --data data/public_test/public/images/
```

To get predictions for the public test set 2, run

```
python3 infer.py --data data/public_test_2/images/
```

> Use the flag `-h` to see other training or inference arguments.