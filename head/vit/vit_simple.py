from dlj.head.head.vit.vit import build_simple_model, run_training


def main():
    run_training(
        model_builder=build_simple_model,
        output_name="vit_simple.ckpt",
        head_name="Simple (FC->output)",
        log_name="vit_simple.log",
    )


if __name__ == "__main__":
    main()
