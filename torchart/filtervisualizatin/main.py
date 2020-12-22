# CNN Filter Visualizer
#
# This script is based on this blog post and repository
# https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# https://github.com/utkuozbulak/pytorch-cnn-visualizations
#

import torch
import torchvision
import numpy as np
import PIL


INITIAL_IMAGE_SIZE = 56
UPSCALING_FACTOR = 1.2
UPSCALING_STEPS = 12
LEARNING_RATE = 0.1
TARGET_LAYER_NUMBER = 15
TARGET_FILTER_NUMBER = 10
OPT_STEPS = 20


class FilterVisualizer():
    def __init__(self, size=56, upscaling_steps=12, upscaling_factor=1.2):
        self.size = size
        self.upscaling_steps = upscaling_steps
        self.upscaling_factor = upscaling_factor
        # Pretrained models are provided by torchvision
        # https://pytorch.org/docs/stable/torchvision/models.html
        self.model = torchvision.models.vgg16(pretrained=True, progress=True)
        self.model = self.model.cuda().eval()

    def train(self, layer=14, filter=0, lr=0.1, opt_steps=20):
        self.layer = layer
        self.filter = filter
        self.lr = lr
        self.set_fook(layer)
        sz = self.size
        img = torch.tensor(
            np.random.uniform(150, 180, (1, 3, sz, sz)) / 255,
            requires_grad=True,
            dtype=torch.float32,
            device="cuda")

        # Register Forward Hook
        for upscaling_step in range(self.upscaling_steps):
            optimizer = torch.optim.Adam([img], lr=lr, weight_decay=1e-6)
            for opt_step in range(opt_steps):
                optimizer.zero_grad()
                self.model(img)
                loss = -self.forward_hook_tensor[0, filter].mean()
                print("upscaling_step:", upscaling_step, ",\tstep: ", opt_step, ",\tloss: ", loss)
                loss.backward()
                optimizer.step()
            self.output = img
            sz = int(self.upscaling_factor * sz)
            img = torchvision.transforms.functional.resize(img, [sz, sz], PIL.Image.BICUBIC)
            # Trick. Convert img to leaf tensor
            img = img.clone().detach().requires_grad_(True)

        # Save Image

    def set_fook(self, layer):
        def forward_hook(module, input, output):
            self.forward_hook_tensor = output
        self.model.features[layer].register_forward_hook(forward_hook)

    def save_output_image(self):
        # TODO: Improve output image quality for human evaluation
        torchvision.utils.save_image(
                torch.nn.functional.normalize(self.output[0]),
                "./torchart/filtervisualizatin/output/vgg16_{}_{}.png".format(
                    str(self.layer), str(self.filter)))

def main():
    visualizer = FilterVisualizer(
            size=INITIAL_IMAGE_SIZE, 
            upscaling_steps=UPSCALING_STEPS, 
            upscaling_factor=UPSCALING_FACTOR)
    visualizer.train(
            layer=TARGET_LAYER_NUMBER,
            filter=TARGET_FILTER_NUMBER,
            lr=LEARNING_RATE,
            opt_steps=OPT_STEPS)
    visualizer.save_output_image()


if __name__ == "__main__":
    main()
