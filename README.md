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

To generate images, just run `torchart/filtervisualization/main.py`.

```
$ python3 torchart/filtervisualization/main.py
```

To change target filters and layers, edit this part in `main.py` as you like.

```
TARGET_LAYER_NUMBER = 3
# TARGET_LAYER_NUMBER = 10
# TARGET_LAYER_NUMBER = 29

TARGET_FILTER_NUMBER = [23, 34, 39, 52]  # This is for the Layer 3
# TARGET_FILTER_NUMBER = [1, 161, 163, 237, 241] # This is for the Layer 10
# TARGET_FILTER_NUMBER = [5, 33, 132, 177, 241, 286, 312, 390] # This is for the Layer 29
```

## Unit testing and Lint

### pytest

```
$ pytest ./torchart
```

### flake8

```
$ flake8 ./torchart
```
