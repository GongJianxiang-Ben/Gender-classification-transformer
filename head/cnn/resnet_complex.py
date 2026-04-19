import torch.nn as nn

from resnet import run_training
from resnet18 import BasicBlock, ResNet


def build_complex_model():
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2)
    model.fc = nn.Sequential(
        nn.Linear(512, 384),
        nn.ReLU(inplace=True),
        nn.Linear(384, 2),
    )
    return model


def main():
    run_training(
        model_builder=build_complex_model,
        output_name="resnet_complex.pth",
        head_name="Complex (FC->ReLU->FC->output)",
        log_name="resnet_complex.log",
    )


if __name__ == "__main__":
    main()
