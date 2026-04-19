import torch.nn as nn

from dlj.head.head.vit.vit import build_vit_config, run_training
from transformers import ViTForImageClassification


def build_complex_model():
    model = ViTForImageClassification(build_vit_config())
    model.classifier = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 2),
    )
    return model


def main():
    run_training(
        model_builder=build_complex_model,
        output_name="vit_complex.ckpt",
        head_name="Complex (FC->ReLU->FC->output)",
        log_name="vit_complex.log",
    )


if __name__ == "__main__":
    main()
