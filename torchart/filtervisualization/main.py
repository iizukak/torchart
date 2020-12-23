#
# Convolutional Neural Network Filter Visualizer
#
# This script is based on this blog post and repository
# https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# https://github.com/utkuozbulak/pytorch-cnn-visualizations
#

import torch
import torchvision
import numpy as np
import PIL


#
# Hyper Parameters for the training
#
INITIAL_IMAGE_SIZE = 56
UPSCALING_FACTOR = 1.2 # Each steps, image_size = image * UPSCALING_FACTOR
UPSCALING_STEPS = 21   # Increase this number to generate larger image
LEARNING_RATE = 0.1

TARGET_LAYER_NUMBER = 3
# TARGET_LAYER_NUMBER = 10
# TARGET_LAYER_NUMBER = 29

TARGET_FILTER_NUMBER = [23, 34, 39, 52] # This is for the Layer 3
# TARGET_FILTER_NUMBER =  [1, 161, 163, 237, 241] # This is for the Layer 10
# TARGET_FILTER_NUMBER = [5, 33, 132, 177, 241, 286, 312] # This is for the Layer 29
OPT_STEPS = 30


class FilterVisualizer():
    def __init__(self, size, layer, upscaling_steps, upscaling_factor, learning_rate, opt_steps):
        self.filter = layer
        self.size = size
        self.learning_rate = learning_rate
        self.upscaling_steps = upscaling_steps
        self.upscaling_factor = upscaling_factor
        self.opt_steps = opt_steps
        # Pretrained models are provided by torchvision
        # https://pytorch.org/docs/stable/torchvision/models.html
        self.model = torchvision.models.vgg16(pretrained=True, progress=True)
        self.model = self.model.cuda().eval()
        self.set_fook(self.layer)

    def train(self, filter):
        # Training image for the target layer
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
                loss.backward()
                optimizer.step()
            print("upscaling_step:", upscaling_step, ",\tstep: ", opt_step, ",\tloss: ", loss)
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
                torch.nn.functional.normalize(self.output),
                "./torchart/filtervisualizatin/output/vgg16_{}_{}.png".format(
                    str(self.layer), str(self.filter)))

def main():
    visualizer = FilterVisualizer(
                size=INITIAL_IMAGE_SIZE, 
                upscaling_steps=UPSCALING_STEPS, 
                upscaling_factor=UPSCALING_FACTOR)
    # for i in [5, 33, 132, 177, 241, 286, 312]: # This is for Layer 29
    # for i in [1, 161, 163, 237, 241]: # This is for Layer 10
    for filter in TARGET_FILTER_NUMBER: # This is for Layer 3
        print ("Train {} th filter".format(filter))
        visualizer.train(
           layer=TARGET_LAYER_NUMBER,
           filter=filter,
           lr=LEARNING_RATE,
           opt_steps=OPT_STEPS)
        visualizer.save_output_image()


if __name__ == "__main__":
    main()
