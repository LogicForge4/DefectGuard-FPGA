import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# MODEL
# Exact FP32 architecture used during training
# ============================================================

class HLSResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual


class HLSRestorationNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: 1 x 128 x 128
        self.head = nn.Conv2d(
            1, 32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.res1 = nn.Sequential(
            HLSResidualBlock(32),
            HLSResidualBlock(32),
            HLSResidualBlock(32),
            HLSResidualBlock(32)
        )

        self.expand = nn.Conv2d(
            32, 64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.res2 = nn.Sequential(
            HLSResidualBlock(64),
            HLSResidualBlock(64),
            HLSResidualBlock(64),
            HLSResidualBlock(64)
        )

        self.reduce = nn.Conv2d(
            64, 32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 128 x 128 -> 256 x 256
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

        self.res3 = nn.Sequential(
            HLSResidualBlock(32),
            HLSResidualBlock(32)
        )

        self.tail = nn.Conv2d(
            32, 1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Training model output is in [0, 1]
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.head(x)
        x = self.res1(x)
        x = self.expand(x)
        x = self.relu(x)
        x = self.res2(x)
        x = self.reduce(x)
        x = self.upsample(x)
        x = self.res3(x)
        x = self.tail(x)
        x = self.output_activation(x)
        return x


# ============================================================
# CHECKPOINT
# ============================================================

def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = HLSRestorationNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print("Checkpoint:", checkpoint_path)
    print("Saved epoch:", checkpoint.get("epoch", "N/A"))
    print("Validation loss:", checkpoint.get("val_loss", "N/A"))
    print("Validation PSNR:", checkpoint.get("val_psnr", "N/A"))
    print("Validation MAE:", checkpoint.get("val_mae", "N/A"))

    return model


# ============================================================
# SINGLE IMAGE INFERENCE
# ============================================================

def restore_one(model, input_path, output_path, device):
    noisy = np.load(input_path).astype(np.float32)

    if noisy.ndim != 2:
        raise ValueError(
            f"{input_path.name}: expected a 2-D array, "
            f"got shape {noisy.shape}"
        )

    # Same preprocessing as training:
    # H,W -> 1,H,W -> batch dimension
    tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        prediction = model(tensor)

    # Remove batch/channel dimensions.
    restored = prediction.squeeze(0).squeeze(0).cpu().numpy()

    # Preserve the model's FP32 [0,1] output representation.
    restored = restored.astype(np.float32)

    if restored.shape != (256, 256):
        raise RuntimeError(
            f"{input_path.name}: expected output shape (256,256), "
            f"got {restored.shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, restored)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run HLSRestorationNet inference on a directory "
            "of NoisyLR .npy test images and save restored .npy files."
        )
    )

    # Positional arguments match the competition requirement:
    # evaluate.py <test_images_directory> <output_directory>
    parser.add_argument(
        "test_images",
        type=str,
        help="Directory containing test .npy images."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where restored .npy files are written."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path. "
            "Default: models/best_fp32.pth relative to this script."
        )
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Kept for interface compatibility; inference is saved per image."
    )

    args = parser.parse_args()

    test_dir = Path(args.test_images)
    output_dir = Path(args.output_dir)

    # Resolve default model relative to repository root.
    repo_root = Path(__file__).resolve().parent

    if args.checkpoint is None:
        checkpoint_path = repo_root / "models" / "best_fp32.pth"
    else:
        checkpoint_path = Path(args.checkpoint)

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test image directory not found: {test_dir}"
        )

    # Support both:
    #   test_images/
    #       00001.npy
    # and:
    #   test_images/
    #       NoisyLR/
    #           00001.npy
    noisy_files = sorted(test_dir.glob("*.npy"))

    if not noisy_files:
        nested = test_dir / "NoisyLR"
        if nested.is_dir():
            noisy_files = sorted(nested.glob("*.npy"))

    if not noisy_files:
        raise FileNotFoundError(
            f"No .npy files found in {test_dir} "
            f"or {test_dir / 'NoisyLR'}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("HLS RESTORATION TEST INFERENCE")
    print("=" * 80)
    print("Device       :", device)
    print("Test images  :", len(noisy_files))
    print("Checkpoint   :", checkpoint_path)
    print("Output dir   :", output_dir)

    if device.type == "cuda":
        print("GPU          :", torch.cuda.get_device_name(0))

    model = load_model(
        checkpoint_path,
        device
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("STARTING INFERENCE")
    print("=" * 80)

    for index, input_path in enumerate(noisy_files, start=1):
        output_path = output_dir / input_path.name

        restore_one(
            model,
            input_path,
            output_path,
            device
        )

        print(
            f"[{index:04d}/{len(noisy_files):04d}] "
            f"{input_path.name} -> {output_path.name}"
        )

    print("=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print("Input files  :", len(noisy_files))
    print("Output files :", len(list(output_dir.glob("*.npy"))))
    print("Output dir   :", output_dir)
    print("Each output  : 256 x 256 float32 .npy")
    print("Range        : [0, 1] from final Sigmoid")
    print("SUCCESS")


if __name__ == "__main__":
    main()