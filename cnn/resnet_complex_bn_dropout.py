import torch.nn as nn

from dlj.head.cnn.resnet import run_training
from dlj.head.cnn.resnet18 import BasicBlock, ResNet


def build_complex_bn_dropout_model():
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2)
    model.fc = nn.Sequential(
        nn.Linear(512, 384),
        nn.BatchNorm1d(384),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(384, 2),
    )
    return model


def main():
    run_training(
        model_builder=build_complex_bn_dropout_model,
        output_name="resnet_complex_bn_dropout.pth",
        head_name="Complex (FC->BN->ReLU->Dropout->FC->output)",
        log_name="resnet_complex_bn_dropout.log",
    )


if __name__ == "__main__":
    main()
