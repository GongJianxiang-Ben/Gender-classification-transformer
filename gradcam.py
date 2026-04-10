import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms

from resnet18 import ResNet, BasicBlock


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0)

        output = self.model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients  = self.gradients[0]   # [C, H, W]
        activations = self.activations[0] # [C, H, W]

        weights = gradients.mean(dim=(1, 2))  # [C]
        cam = (weights[:, None, None] * activations).sum(dim=0)  # [H, W]
        cam = torch.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), class_idx, output.softmax(dim=1)[0].cpu().detach().numpy()


def overlay_cam(img_pil, cam, alpha=0.5):
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    cam_resized = np.array(Image.fromarray(
        (cam * 255).astype(np.uint8)).resize((224, 224))) / 255.0

    # Colormap
    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    overlay = alpha * heatmap + (1 - alpha) * img_np
    overlay = np.clip(overlay, 0, 1)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="gradcam_output")
    parser.add_argument("--num_samples",type=int, default=6)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load model
    model = ResNet(img_channels=3, num_layers=18,
                   block=BasicBlock, num_classes=2)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    # Target last conv layer
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    # Collect images from data_dir
    # Expects organized/ with f/ and m/ subdirs
    class_dirs = {"f": 0, "m": 1}
    class_names = {0: "Female", 1: "Male"}
    all_images = []

    for cls_name, cls_idx in class_dirs.items():
        cls_dir = os.path.join(args.data_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append((os.path.join(cls_dir, fname), cls_idx))

    random.shuffle(all_images)

    # Collect correct and incorrect predictions
    correct_samples   = []
    incorrect_samples = []

    for img_path, true_label in all_images:
        if len(correct_samples) >= 3 and len(incorrect_samples) >= 3:
            break
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = val_tf(img_pil).to(device)
            cam, pred_idx, probs = gradcam.generate(img_tensor)

            if pred_idx == true_label and len(correct_samples) < 3:
                correct_samples.append((img_pil, cam, true_label,
                                        pred_idx, probs))
            elif pred_idx != true_label and len(incorrect_samples) < 3:
                incorrect_samples.append((img_pil, cam, true_label,
                                          pred_idx, probs))
        except Exception as e:
            continue

    all_samples = correct_samples + incorrect_samples
    labels_row  = (["Correct"] * len(correct_samples) +
                   ["Incorrect"] * len(incorrect_samples))

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Grad-CAM Heatmaps — ResNet18 Gender Classification",
                 fontsize=16, fontweight="bold")

    for i, (img_pil, cam, true_label, pred_label, probs) in enumerate(all_samples):
        row, col = divmod(i, 3)
        ax = axes[row][col]

        overlay = overlay_cam(img_pil, cam)
        ax.imshow(overlay)

        status = labels_row[i]
        color  = "green" if status == "Correct" else "red"
        title  = (f"{status}\n"
                  f"True: {class_names[true_label]} | "
                  f"Pred: {class_names[pred_label]}\n"
                  f"Conf: {probs[pred_label]*100:.1f}%")
        ax.set_title(title, fontsize=10, color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "gradcam_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also save individual images
    for i, (img_pil, cam, true_label, pred_label, probs) in enumerate(all_samples):
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(img_pil.resize((224, 224)))
        ax1.set_title("Original")
        ax1.axis("off")
        overlay = overlay_cam(img_pil, cam)
        ax2.imshow(overlay)
        ax2.set_title(f"Grad-CAM\nTrue: {class_names[true_label]} | "
                      f"Pred: {class_names[pred_label]}")
        ax2.axis("off")
        plt.tight_layout()
        fname = f"gradcam_{'correct' if i < 3 else 'incorrect'}_{i % 3}.png"
        plt.savefig(os.path.join(args.output_dir, fname),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"All Grad-CAM images saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()