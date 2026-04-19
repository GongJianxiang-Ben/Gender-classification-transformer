from resnet import build_simple_model, run_training


def main():
    run_training(
        model_builder=build_simple_model,
        output_name="resnet_simple.pth",
        head_name="Simple (FC->output)",
        log_name="resnet_simple.log",
    )


if __name__ == "__main__":
    main()
