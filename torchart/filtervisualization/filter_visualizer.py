#
# Convolutional Neural Network Filter Visualization Class
#
# This script is based on this blog post and repository
# https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# https://github.com/utkuozbulak/pytorch-cnn-visualizations
#


import torch
import torchvision
import numpy as np
import PIL


class FilterVisualizer:
    def __init__(
        self,
        size,
        layer,
        upscaling_steps,
        upscaling_factor,
        learning_rate,
        opt_steps,
        device,
    ):
        self.layer = layer
        self.size = size
        self.upscaling_steps = upscaling_steps
        self.upscaling_factor = upscaling_factor
        self.learning_rate = learning_rate
        self.opt_steps = opt_steps
        self.device = device
        # Pretrained models are provided by torchvision
        # https://pytorch.org/docs/stable/torchvision/models.html
        self.model = torchvision.models.vgg16(pretrained=True, progress=True)
        if self.device == "cuda":
            self.model = self.model.cuda().eval()
        else:
            self.model = self.model.cpu().eval()
        self.set_fook(self.layer)

    def train(self, filter):
        # Training image for the target layer
        print("\nTrain {} th filter".format(filter))
        current_size = self.size
        self.filter = filter
        img = torch.tensor(
            np.random.uniform(150, 180, (1, 3, current_size, current_size)) / 255,
            requires_grad=True,
            dtype=torch.float32,
            device=self.device,
        )

        # Register Forward Hook
        for upscaling_step in range(self.upscaling_steps + 1):
            optimizer = torch.optim.Adam(
                [img], lr=self.learning_rate, weight_decay=1e-6
            )
            for opt_step in range(self.opt_steps):
                optimizer.zero_grad()
                self.model(img)
                loss = -self.forward_hook_tensor[0, self.filter].mean()
                loss.backward()
                optimizer.step()
            print(
                "layer: {}, filter: {} upscaling_step: {}, loss:{}".format(
                    self.layer, self.filter, upscaling_step, loss
                )
            )
            self.output = img
            current_size = int(self.upscaling_factor * current_size)
            img = torchvision.transforms.functional.resize(
                img, [current_size, current_size], PIL.Image.BICUBIC
            )
            # Trick. Convert img to leaf tensor
            img = img.clone().detach().requires_grad_(True)

        # Save Image

    def set_fook(self, layer):
        def forward_hook(module, input, output):
            self.forward_hook_tensor = output

        self.model.features[layer].register_forward_hook(forward_hook)

    def save_output_image(self):
        torchvision.utils.save_image(
            torch.nn.functional.normalize(self.output),
            "vgg16_{}_{}.png".format(
                str(self.layer), str(self.filter)
            ),
        )
