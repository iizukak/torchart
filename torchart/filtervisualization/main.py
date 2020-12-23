from filter_visualizer import FilterVisualizer

#
# Hyper Parameters for the training
#
INITIAL_IMAGE_SIZE = 56
UPSCALING_FACTOR = 1.2  # Each steps, image_size = image * UPSCALING_FACTOR
UPSCALING_STEPS = 12  # Increase this number to generate larger image
LEARNING_RATE = 0.1

TARGET_LAYER_NUMBER = 3
# TARGET_LAYER_NUMBER = 10
# TARGET_LAYER_NUMBER = 29

TARGET_FILTER_NUMBER = [23, 34, 39, 52]  # This is for the Layer 3
# TARGET_FILTER_NUMBER =  [1, 161, 163, 237, 241] # This is for the Layer 10
# TARGET_FILTER_NUMBER = [5, 33, 132, 177, 241, 286, 312] # This is for the Layer 29
OPT_STEPS = 30
DEVICE = "cuda"  # or "cpu"


def main():
    visualizer = FilterVisualizer(
        size=INITIAL_IMAGE_SIZE,
        layer=TARGET_LAYER_NUMBER,
        upscaling_steps=UPSCALING_STEPS,
        upscaling_factor=UPSCALING_FACTOR,
        learning_rate=LEARNING_RATE,
        opt_steps=OPT_STEPS,
        device=DEVICE
    )
    for filter in TARGET_FILTER_NUMBER:
        print("Train {} th filter".format(filter))
        visualizer.train(filter=filter)
        visualizer.save_output_image()


if __name__ == "__main__":
    main()
