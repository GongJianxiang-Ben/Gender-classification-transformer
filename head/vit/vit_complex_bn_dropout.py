import torch.nn as nn

from head.vit.vit import build_vit_config, run_training
from transformers import ViTForImageClassification


def build_complex_bn_dropout_model():
    model = ViTForImageClassification(build_vit_config())
    model.classifier = nn.Sequential(
        nn.Linear(768, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, 2),
    )
    return model


def main():
    run_training(
        model_builder=build_complex_bn_dropout_model,
        output_name="vit_complex_bn_dropout.ckpt",
        head_name="Complex (FC->BN->ReLU->Dropout->FC->output)",
        log_name="vit_complex_bn_dropout.log",
    )


if __name__ == "__main__":
    main()
