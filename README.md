# torchart

![ci](https://github.com/iizukak/torchart/workflows/ci/badge.svg?branch=main)

Experimental Art Projects with PyTorch

## Installation

```
$ pip3 install -r ./requirements.txt
$ export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Usage

### Filter Visualization

<img src="sample_output/filtervisualization/vgg16_3_34.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_3_39.png" width=224px height=224px>  <img src="sample_output/filtervisualization/vgg16_3_52.png" width=224px height=224px>


Layer: `model.features[3]`, Filter: 34(Left) 39(Center), 52(Right)

<img src="sample_output/filtervisualization/vgg16_10_1.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_10_163.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_10_237.png" width=224px height=224px>

Layer: `model.features[10]`, Filter: 1(Left) 163(Center) 237(Right)

<img src="sample_output/filtervisualization/vgg16_29_33.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_29_132.png" width=224px height=224px> <img src="sample_output/filtervisualization/vgg16_29_390.png" width=224px height=224px>

Layer: `model.features[29]`, Filter: 33(Left) 132(Center), 390(Right)

```
$ python3 torchart/filtervisualization/main.py
```

### Run unit test

```
$ pytest ./torchart
```

### Check code format

```
$ flake8 ./torchart
```
