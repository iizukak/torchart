# torchart

![ci](https://github.com/iizukak/torchart/workflows/ci/badge.svg?branch=main)

Experimental Art Projects with PyTorch

## Installation

```
$ pip3 install -r ./requirements.txt
$ export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Filter Visualization

### Sample Outputs

These images are visualized filters for VGG16 filters.
Model is from torchvision's [pretrained model](https://pytorch.org/docs/stable/torchvision/models.html).

You can check sample images in [this](./sample_output/filtervisualization) directory.

<img src="sample_output/filtervisualization/vgg16_3_34.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_3_39.png" width=224px height=224px>  <img src="sample_output/filtervisualization/vgg16_3_52.png" width=224px height=224px>


Layer: `model.features[3]`, Filter: 34(Left) 39(Center), 52(Right)

<img src="sample_output/filtervisualization/vgg16_10_1.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_10_163.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_10_237.png" width=224px height=224px>

Layer: `model.features[10]`, Filter: 1(Left) 163(Center) 237(Right)

<img src="sample_output/filtervisualization/vgg16_29_33.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_29_132.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_29_390.png" width=224px height=224px>

Layer: `model.features[29]`, Filter: 33(Left) 132(Center), 390(Right)

### Usage

You can use `layer` and `filters` options to specify target number of layer and filters.

```
$ python3 torchart/filtervisualization/main.py
$ python3 torchart/filtervisualization/main.py layer=3 filters='[23, 34, 39, 52]'
$ python3 torchart/filtervisualization/main.py layer=10 filters='[1, 161, 163, 237, 241]'
$ python3 torchart/filtervisualization/main.py layer=29 filters='[5, 33, 132, 177, 241, 286, 312]'
```

If you want to change more hyper parametrs, Please check `torchart/filtervisualization/config.yaml`.
We are using [hydra](https://hydra.cc/). And you can change hyper parameters as command line arguments.


## Unit testing and Lint

### pytest

```
$ pytest torchart
```

### flake8

```
$ flake8 torchart
```
