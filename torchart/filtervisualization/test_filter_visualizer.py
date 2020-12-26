import os

import torch

from torchart.filtervisualization.filter_visualizer import FilterVisualizer


def test_init_filter_visualizer() -> None:
    visualizer = FilterVisualizer(100, 3, 1, 1.5, 0.1, 1, "cpu")
    assert len(visualizer.model.features) == 31


def test_up_scaling() -> None:
    visualizer = FilterVisualizer(100, 3, 1, 1.5, 0.1, 1, "cpu")
    visualizer.train(0)
    assert visualizer.output.shape == torch.Size([1, 3, 150, 150])


def test_save_output_image() -> None:
    test_file_path = "./vgg16_3_0.png"
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
    visualizer = FilterVisualizer(100, 3, 1, 1.5, 0.1, 1, "cpu")
    visualizer.train(0)
    visualizer.save_output_image()
    assert os.path.exists(test_file_path)
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
