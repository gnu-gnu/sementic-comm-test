"""
Semantic Communication — Model Training
========================================
학습 후 모델을 스크립트 디렉토리에 저장합니다 (기본값: jscc_model.pt).

Usage:
    python train.py                          # 학습만
    python train.py --compile-trt            # 학습 + TRT 엔진 사전 컴파일
    python train.py --epochs 100 --image image_4k.jpg
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from models import SemanticCommSystem, extract_patches, save_trt_engines

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def train(model, patches, args):
    device = args.device
    model = model.to(device)
    dataset = TensorDataset(patches)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n[Training] epochs={args.epochs}, SNR=[{args.snr_min}, {args.snr_max}] dB, "
          f"device={device}")

    for epoch in range(args.epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            snr_db = torch.empty(1).uniform_(args.snr_min, args.snr_max).item()
            x_hat = model(batch, snr_db)
            loss = F.mse_loss(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        scheduler.step()
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.6f}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train JSCC Semantic Communication Model')
    parser.add_argument('--image', default='image_4k.jpg', help='Training image')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--patch-stride', type=int, default=64)
    parser.add_argument('--bottleneck-ch', type=int, default=16)
    parser.add_argument('--snr-min', type=float, default=0)
    parser.add_argument('--snr-max', type=float, default=20)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='jscc_model.pt', help='Output model path')
    parser.add_argument('--compile-trt', action='store_true',
                        help='Pre-compile TensorRT engines after training')
    args = parser.parse_args()

    image_path = os.path.join(SCRIPT_DIR, args.image)

    print("=== JSCC Model Training ===")
    print(f"Image: {image_path}")
    print(f"Device: {args.device}")

    # Build model
    model = SemanticCommSystem(bottleneck_ch=args.bottleneck_ch)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Extract patches
    print("\nExtracting patches...")
    patches = extract_patches(image_path, args.patch_size, args.patch_stride)
    patches_aug = torch.cat([
        patches,
        patches.flip(dims=[2]),
        patches.flip(dims=[3]),
    ], dim=0)
    print(f"  After augmentation: {patches_aug.shape[0]} patches")

    # Train
    model = train(model, patches_aug, args)

    # Save
    save_path = os.path.join(SCRIPT_DIR, args.output)
    save_data = {
        'model_state_dict': model.state_dict(),
        'bottleneck_ch': args.bottleneck_ch,
        'patch_size': args.patch_size,
        'snr_range': (args.snr_min, args.snr_max),
        'epochs': args.epochs,
    }
    torch.save(save_data, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\nModel saved: {save_path} ({size_mb:.1f} MB)")

    # Pre-compile TensorRT engines
    if args.compile_trt:
        print("\n[TensorRT] Pre-compiling engines...")
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        n_tiles_h = -(-h // args.patch_size)
        n_tiles_w = -(-w // args.patch_size)
        n_tiles = n_tiles_h * n_tiles_w
        save_trt_engines(model, n_tiles, args.patch_size,
                         args.bottleneck_ch, args.device)
        print("[TensorRT] Done.")


if __name__ == '__main__':
    main()
