import hydra
from omegaconf.dictconfig import DictConfig

from torchart.filtervisualization.filter_visualizer import FilterVisualizer


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    visualizer = FilterVisualizer(
        size=cfg.initial_image_size,
        layer=cfg.layer,
        upscaling_steps=cfg.upscaling_steps,
        upscaling_factor=cfg.upscaling_factor,
        learning_rate=cfg.learning_rate,
        opt_steps=cfg.opt_steps,
        device=cfg.device,
    )
    for filter in cfg.filters:
        visualizer.train(filter=filter)
        visualizer.save_output_image()


if __name__ == "__main__":
    main()
