"""
Semantic Communication — Demo
==============================
사전 학습된 모델을 로드하여 전통적 통신 vs 시멘틱 통신을 비교합니다.

Usage:
    python demo.py                              # FP16, SNR=[0,5,10,15,20]
    python demo.py --backend fp32               # FP32
    python demo.py --backend tensorrt           # TensorRT FP16
    python demo.py --snr -10 -5 0 10 20         # Custom SNR values
"""

import argparse
import io
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import (
    SemanticCommSystem, compute_psnr, compute_ssim,
    reconstruct_jscc, jpeg_through_channel, snr_to_ber,
    _prepare_tiles, power_normalize,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _fmt_time(seconds):
    """Format seconds into human-readable string."""
    ms = seconds * 1000
    if ms >= 1000:
        return f"{ms/1000:.1f}s"
    return f"{ms:.1f}ms"


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = SemanticCommSystem(bottleneck_ch=checkpoint['bottleneck_ch'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    print(f"Model loaded: {model_path}")
    print(f"  bottleneck_ch={checkpoint['bottleneck_ch']}, "
          f"patch_size={checkpoint['patch_size']}, "
          f"trained SNR={checkpoint['snr_range']}, "
          f"epochs={checkpoint['epochs']}")
    return model, checkpoint


def run_stage1(image_path, config, output_dir):
    print("\n" + "=" * 60)
    print("STAGE 1: Bandwidth Efficiency Comparison")
    print("=" * 60)

    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    device = config['device']

    raw_size = w * h * 3

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    jpeg_size = buf.tell()

    patch_size = config['patch_size']
    n_tiles = (-(- h // patch_size)) * (-(-w // patch_size))
    bc = config['bottleneck_ch']
    jscc_size = n_tiles * bc * 4 * 4 * 4  # float32

    print("  Extracting ResNet-50 semantic embedding...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    resnet.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        modules = list(resnet.children())[:-1]
        feat_ext = nn.Sequential(*modules).to(device)
        embedding = feat_ext(preprocess(img).unsqueeze(0).to(device)).squeeze()
    semantic_size = embedding.numel() * 4
    del resnet, feat_ext
    torch.cuda.empty_cache()

    sizes = {
        'Raw Image\n(uncompressed)': raw_size,
        'JPEG\n(quality=85)': jpeg_size,
        f'JSCC\n(bottleneck={bc}ch)': jscc_size,
        'Semantic\n(ResNet-2048d)': semantic_size,
    }

    print(f"\n  {'Method':<30s} {'Size':>12s}  {'Compression':>12s}")
    print(f"  {'-'*56}")
    for name, size in sizes.items():
        label = name.replace('\n', ' ')
        ratio = raw_size / size
        if size >= 1024 * 1024:
            s = f"{size / 1024 / 1024:.1f} MB"
        elif size >= 1024:
            s = f"{size / 1024:.1f} KB"
        else:
            s = f"{size} B"
        print(f"  {label:<30s} {s:>12s}  {ratio:>10.0f}x")

    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(sizes.keys())
    values = [v / 1024 for v in sizes.values()]
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Transmission Size (KB)', fontsize=13)
    ax.set_title('Stage 1: Bandwidth Efficiency — Traditional vs Semantic Communication',
                 fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    for bar, (name, size) in zip(bars, sizes.items()):
        ratio = raw_size / size
        if size >= 1024 * 1024:
            label = f"{size/1024/1024:.1f} MB"
        elif size >= 1024:
            label = f"{size/1024:.1f} KB"
        else:
            label = f"{size} B"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                f"{label}\n({ratio:.0f}x compression)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(top=values[0] * 5)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'stage1_bandwidth.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


def run_stage2(image_path, model, config, output_dir):
    backend = config['backend']
    backend_label = backend.upper() if backend != 'tensorrt' else 'TensorRT'

    print("\n" + "=" * 60)
    print(f"STAGE 2: Channel Robustness Comparison (backend: {backend_label})")
    print("=" * 60)

    device = config['device']
    snr_list = config['snr_eval_list']
    patch_size = config['patch_size']

    original_tensor = transforms.ToTensor()(Image.open(image_path).convert('RGB'))

    results_jscc = {}
    results_jpeg = {}

    for snr_db in snr_list:
        ber = snr_to_ber(snr_db)
        print(f"\n  SNR = {snr_db:2d} dB  (BER = {ber:.2e})")

        recon_jscc, orig = reconstruct_jscc(model, image_path, snr_db, patch_size, device,
                                             backend=config['backend'])
        psnr_jscc = compute_psnr(orig.unsqueeze(0), recon_jscc.unsqueeze(0))
        ssim_jscc = compute_ssim(orig.unsqueeze(0), recon_jscc.unsqueeze(0))
        results_jscc[snr_db] = {'image': recon_jscc, 'psnr': psnr_jscc, 'ssim': ssim_jscc}
        print(f"    JSCC  — PSNR: {psnr_jscc:.2f} dB, SSIM: {ssim_jscc:.4f}")

        recon_jpeg, _, _ = jpeg_through_channel(image_path, snr_db)
        if recon_jpeg.shape != original_tensor.shape:
            recon_jpeg = recon_jpeg[:, :original_tensor.shape[1], :original_tensor.shape[2]]
            if recon_jpeg.shape != original_tensor.shape:
                recon_jpeg = F.interpolate(
                    recon_jpeg.unsqueeze(0),
                    size=(original_tensor.shape[1], original_tensor.shape[2]),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
        psnr_jpeg = compute_psnr(original_tensor.unsqueeze(0), recon_jpeg.unsqueeze(0))
        ssim_jpeg = compute_ssim(original_tensor.unsqueeze(0), recon_jpeg.unsqueeze(0))
        results_jpeg[snr_db] = {'image': recon_jpeg, 'psnr': psnr_jpeg, 'ssim': ssim_jpeg}
        print(f"    JPEG  — PSNR: {psnr_jpeg:.2f} dB, SSIM: {ssim_jpeg:.4f}")

    # Comparison grid
    n_snr = len(snr_list)
    fig, axes = plt.subplots(3, n_snr, figsize=(4 * n_snr, 12))
    for j, snr_db in enumerate(snr_list):
        axes[0, j].imshow(original_tensor.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, j].set_title('Original', fontsize=11)
        axes[0, j].axis('off')
    for j, snr_db in enumerate(snr_list):
        r = results_jpeg[snr_db]
        axes[1, j].imshow(r['image'].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, j].set_title(
            f'Traditional (JPEG)\nSNR={snr_db}dB\n'
            f'PSNR={r["psnr"]:.1f} / SSIM={r["ssim"]:.3f}', fontsize=9)
        axes[1, j].axis('off')
    for j, snr_db in enumerate(snr_list):
        r = results_jscc[snr_db]
        axes[2, j].imshow(r['image'].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[2, j].set_title(
            f'Semantic (JSCC {backend_label})\nSNR={snr_db}dB\n'
            f'PSNR={r["psnr"]:.1f} / SSIM={r["ssim"]:.3f}', fontsize=9)
        axes[2, j].axis('off')
    fig.suptitle(f'Stage 2: Channel Robustness — Traditional vs Semantic ({backend_label})',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, 'stage2_comparison.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # Metrics curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    psnr_j = [results_jscc[s]['psnr'] for s in snr_list]
    psnr_t = [results_jpeg[s]['psnr'] for s in snr_list]
    ssim_j = [results_jscc[s]['ssim'] for s in snr_list]
    ssim_t = [results_jpeg[s]['ssim'] for s in snr_list]

    ax1.plot(snr_list, psnr_j, 'o-', color='#2ecc71', lw=2.5, ms=8, label=f'Semantic (JSCC {backend_label})')
    ax1.plot(snr_list, psnr_t, 's--', color='#e74c3c', lw=2.5, ms=8, label='Traditional (JPEG)')
    ax1.set_xlabel('SNR (dB)'); ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR vs Channel Quality', fontweight='bold')
    ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)

    ax2.plot(snr_list, ssim_j, 'o-', color='#2ecc71', lw=2.5, ms=8, label=f'Semantic (JSCC {backend_label})')
    ax2.plot(snr_list, ssim_t, 's--', color='#e74c3c', lw=2.5, ms=8, label='Traditional (JPEG)')
    ax2.set_xlabel('SNR (dB)'); ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM vs Channel Quality', fontweight='bold')
    ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

    fig.suptitle('Stage 2: Quality Metrics — Graceful Degradation vs Cliff Effect',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, 'stage2_metrics.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_stage3(image_path, model, config, output_dir):
    """Stage 3: E2E speed comparison — JPEG vs JSCC across backends and bandwidths."""
    print("\n" + "=" * 60)
    print("STAGE 3: End-to-End Speed Comparison")
    print("=" * 60)

    device = config['device']
    patch_size = config['patch_size']
    img = Image.open(image_path).convert('RGB')

    # --- JPEG timing ---
    # Encode
    t0 = time.time()
    for _ in range(10):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
    jpeg_encode = (time.time() - t0) / 10
    jpeg_bytes = buf.getvalue()
    jpeg_size = len(jpeg_bytes)

    # Decode
    t0 = time.time()
    for _ in range(10):
        Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    jpeg_decode = (time.time() - t0) / 10

    # --- JSCC timing per backend ---
    tiles, _, H, W, tiles_h, tiles_w, pH, pW = _prepare_tiles(image_path, patch_size, device)
    jscc_size = tiles_h * tiles_w * config['bottleneck_ch'] * 4 * 4 * 4  # float32 bytes

    backends = {}

    # FP32
    model.float()
    tiles_f32 = tiles.float()
    with torch.no_grad():
        _ = model(tiles_f32[:1], 10.0); torch.cuda.synchronize()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5):
            z = model.encoder(tiles_f32)
        torch.cuda.synchronize()
        enc32 = (time.time() - t0) / 5
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5):
            _ = model.decoder(z)
        torch.cuda.synchronize()
        dec32 = (time.time() - t0) / 5
    backends['FP32'] = (enc32, dec32)

    # FP16
    model.half()
    tiles_f16 = tiles.half()
    with torch.no_grad():
        _ = model(tiles_f16[:1], 10.0); torch.cuda.synchronize()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            z = model.encoder(tiles_f16)
        torch.cuda.synchronize()
        enc16 = (time.time() - t0) / 10
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            _ = model.decoder(z)
        torch.cuda.synchronize()
        dec16 = (time.time() - t0) / 10
    backends['FP16'] = (enc16, dec16)
    model.float()

    # TensorRT (if available)
    try:
        from models import _compile_trt
        encoder_trt = _compile_trt(model.encoder, tiles, 'encoder')
        with torch.no_grad():
            z_trt = encoder_trt(tiles)
        decoder_trt = _compile_trt(model.decoder, z_trt, 'decoder')
        with torch.no_grad():
            _ = encoder_trt(tiles); _ = decoder_trt(z_trt); torch.cuda.synchronize()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(10):
                z_trt = encoder_trt(tiles)
            torch.cuda.synchronize()
            enc_trt = (time.time() - t0) / 10
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(10):
                _ = decoder_trt(z_trt)
            torch.cuda.synchronize()
            dec_trt = (time.time() - t0) / 10
        backends['TensorRT'] = (enc_trt, dec_trt)
    except Exception:
        pass

    # --- Print table ---
    print(f"\n  JPEG: encode={jpeg_encode*1000:.1f}ms, decode={jpeg_decode*1000:.1f}ms, "
          f"size={jpeg_size/1024:.1f}KB")
    print(f"  JSCC: size={jscc_size/1024:.1f}KB")
    for name, (enc, dec) in backends.items():
        print(f"    {name}: encode={enc*1000:.1f}ms, decode={dec*1000:.1f}ms, "
              f"total={( enc+dec)*1000:.1f}ms")

    # --- Compute crossover bandwidths and plot ---
    bandwidths_kbps = np.logspace(0, 5, 200)  # 1 Kbps to 100 Mbps
    bandwidths_bps = bandwidths_kbps * 1000 / 8  # bytes/sec

    jpeg_tx = jpeg_size / bandwidths_bps
    jscc_tx = jscc_size / bandwidths_bps
    jpeg_e2e = jpeg_encode + jpeg_tx + jpeg_decode

    fig, ax = plt.subplots(figsize=(12, 6))

    # JPEG line
    ax.plot(bandwidths_kbps / 1000, jpeg_e2e * 1000, 'r--', linewidth=2.5,
            label='Traditional (JPEG)')

    colors = {'FP32': '#e67e22', 'FP16': '#2ecc71', 'TensorRT': '#3498db'}
    crossovers = {}

    for name, (enc, dec) in backends.items():
        jscc_e2e = enc + jscc_tx + dec
        ax.plot(bandwidths_kbps / 1000, jscc_e2e * 1000, '-', linewidth=2.5,
                color=colors.get(name, 'gray'), label=f'Semantic ({name})')

        # Find crossover point
        diff = jpeg_e2e - jscc_e2e
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            cross_bw = bandwidths_kbps[idx] / 1000  # Mbps
            crossovers[name] = cross_bw
            ax.axvline(x=cross_bw, color=colors.get(name, 'gray'), alpha=0.3,
                       linestyle=':')
            ax.annotate(f'{name}\ncrossover\n{cross_bw:.1f} Mbps',
                        xy=(cross_bw, jpeg_e2e[idx] * 1000),
                        fontsize=8, ha='center', va='bottom',
                        color=colors.get(name, 'gray'))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Bandwidth (Mbps)', fontsize=13)
    ax.set_ylabel('End-to-End Latency (ms)', fontsize=13)
    ax.set_title('Stage 3: E2E Latency — Where Semantic Communication Wins',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Add region annotations
    ax.fill_between(bandwidths_kbps / 1000, 0.1, 1e7,
                    where=bandwidths_kbps / 1000 < crossovers.get('FP32', 1e6),
                    alpha=0.05, color='green')
    ax.set_ylim(bottom=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'stage3_speed.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # Summary
    print("\n  Crossover points (JSCC becomes faster than JPEG):")
    for name, bw in crossovers.items():
        print(f"    {name}: < {bw:.1f} Mbps")
    if not crossovers:
        print("    JSCC is faster at all tested bandwidths")

    # --- Save benchmark report ---
    w, h = img.size
    lines = [
        f"# E2E Benchmark Report",
        f"",
        f"## Environment",
        f"- Image: {os.path.basename(image_path)} ({w}x{h})",
        f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"- Tiles: {tiles_h}x{tiles_w} = {tiles_h * tiles_w} ({patch_size}x{patch_size})",
        f"- Stage 2 backend: {config['backend']}",
        f"",
        f"## Data Size",
        f"| Method | Size |",
        f"|--------|------|",
        f"| JPEG (quality=85) | {jpeg_size/1024:.1f} KB |",
        f"| JSCC (bottleneck={config['bottleneck_ch']}ch) | {jscc_size/1024:.1f} KB |",
        f"",
        f"## Encode / Decode Timing",
        f"| Method | Encode | Decode | Total |",
        f"|--------|--------|--------|-------|",
        f"| JPEG | {jpeg_encode*1000:.1f}ms | {jpeg_decode*1000:.1f}ms | {(jpeg_encode+jpeg_decode)*1000:.1f}ms |",
    ]
    for name, (enc, dec) in backends.items():
        lines.append(
            f"| JSCC ({name}) | {enc*1000:.1f}ms | {dec*1000:.1f}ms | {(enc+dec)*1000:.1f}ms |"
        )

    lines += [
        f"",
        f"## E2E Latency (Encode + Transmit + Decode)",
        f"| Bandwidth | JPEG |" + " ".join(f"{n} |" for n in backends) + "",
        f"|-----------|------|" + " ".join("------|" for _ in backends) + "",
    ]
    for bw_label, bw_kbps in [('10 Kbps', 10), ('100 Kbps', 100), ('1 Mbps', 1000),
                               ('10 Mbps', 10000), ('100 Mbps', 100000)]:
        bw_b = bw_kbps * 1000 / 8
        j_e2e = jpeg_encode + jpeg_size / bw_b + jpeg_decode
        row = f"| {bw_label} | {_fmt_time(j_e2e)} |"
        for name, (enc, dec) in backends.items():
            js_e2e = enc + jscc_size / bw_b + dec
            row += f" {_fmt_time(js_e2e)} |"
        lines.append(row)

    lines += [
        f"",
        f"## Crossover Points",
        f"JSCC becomes faster than JPEG below these bandwidths:",
        f"",
    ]
    for name, bw in crossovers.items():
        lines.append(f"- **{name}**: < {bw:.1f} Mbps")
    if not crossovers:
        lines.append("- JSCC is faster at all tested bandwidths")

    report_path = os.path.join(output_dir, 'benchmark.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {report_path}")


def run_stage4(image_path, config, output_dir):
    """Stage 4: Semantic Understanding — Caption + Object Detection + Segmentation."""
    print("\n" + "=" * 60)
    print("STAGE 4: Semantic Understanding")
    print("=" * 60)

    device = config['device']
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    raw_size = w * h * 3

    results = {}

    # --- 1. Image Captioning (BLIP) ---
    print("\n  [Captioning] Loading BLIP...")
    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base").to(device).eval()
    inputs = blip_proc(img, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_proc.decode(caption_ids[0], skip_special_tokens=True)
    caption_bytes = len(caption.encode('utf-8'))
    results['caption'] = {'text': caption, 'size': caption_bytes}
    print(f"    Caption: \"{caption}\"")
    print(f"    Size: {caption_bytes} bytes (compression: {raw_size/caption_bytes:,.0f}x)")
    del blip_model, blip_proc
    torch.cuda.empty_cache()

    # --- 2. Object Detection (Faster R-CNN) ---
    print("\n  [Detection] Loading Faster R-CNN...")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    det_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    det_model = fasterrcnn_resnet50_fpn_v2(weights=det_weights).to(device).eval()
    det_transform = det_weights.transforms()
    img_tensor = det_transform(transforms.ToTensor()(img).unsqueeze(0).to(device))
    with torch.no_grad():
        detections = det_model(img_tensor)[0]

    # Filter: person only (COCO label 1 = person)
    keep = (detections['scores'] > 0.5) & (detections['labels'] == 1)
    boxes = detections['boxes'][keep].cpu()
    labels = detections['labels'][keep].cpu()
    scores = detections['scores'][keep].cpu()
    cat_names = det_weights.meta['categories']

    det_objects = []
    for box, label, score in zip(boxes, labels, scores):
        det_objects.append({
            'class': 'person',
            'confidence': f"{score:.2f}",
            'bbox': [int(x) for x in box.tolist()],
        })

    import json
    det_json = json.dumps(det_objects)
    det_bytes = len(det_json.encode('utf-8'))
    results['detection'] = {'objects': det_objects, 'size': det_bytes,
                            'boxes': boxes, 'labels': labels, 'scores': scores}
    print(f"    Detected {len(det_objects)} persons")
    print(f"    Size: {det_bytes} bytes (compression: {raw_size/det_bytes:,.0f}x)")
    del det_model
    torch.cuda.empty_cache()

    # --- 3. Fire Detection (HSV color filter) + Person Segmentation (DeepLabV3) ---
    print("\n  [Fire + Person Segmentation]")
    import colorsys

    # Fire detection via HSV thresholding
    img_arr_uint8 = np.array(img)
    # Convert RGB to HSV
    img_hsv = np.zeros_like(img_arr_uint8, dtype=np.float32)
    for y in range(img_arr_uint8.shape[0]):
        for x in range(img_arr_uint8.shape[1]):
            r, g, b = img_arr_uint8[y, x] / 255.0
            img_hsv[y, x] = colorsys.rgb_to_hsv(r, g, b)

    # Vectorized HSV conversion is faster
    img_float = img_arr_uint8.astype(np.float32) / 255.0
    r, g, b = img_float[..., 0], img_float[..., 1], img_float[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin
    # Hue
    hue = np.zeros_like(cmax)
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
    sat = np.where(cmax > 0, diff / cmax, 0)
    val = cmax

    # Fire: red-orange-yellow hue (0-60°), high saturation, high value
    fire_mask = ((hue <= 60) | (hue >= 340)) & (sat > 0.3) & (val > 0.4)

    # Person segmentation from DeepLabV3
    print("    Loading DeepLabV3 for person segmentation...")
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    seg_weights = DeepLabV3_ResNet50_Weights.DEFAULT
    seg_model = deeplabv3_resnet50(weights=seg_weights).to(device).eval()
    seg_transform = seg_weights.transforms()
    seg_input = seg_transform(transforms.ToTensor()(img).unsqueeze(0).to(device))
    with torch.no_grad():
        seg_output = seg_model(seg_input)['out']
    seg_map = seg_output.argmax(dim=1).squeeze().cpu()
    seg_cat_names = seg_weights.meta['categories']
    # Person is class 15 in COCO/VOC segmentation
    person_idx = seg_cat_names.index('person') if 'person' in seg_cat_names else 15
    person_mask_seg = (seg_map == person_idx).numpy()
    # Resize to original image size
    person_mask_full = np.array(Image.fromarray(person_mask_seg.astype(np.uint8)).resize(
        (w, h), Image.NEAREST)).astype(bool)

    # Combine into a 2-class map: 0=background, 1=person, 2=fire
    import zlib
    combined_map = np.zeros((h, w), dtype=np.uint8)
    combined_map[fire_mask] = 2
    combined_map[person_mask_full] = 1  # person takes priority over fire
    seg_compressed = zlib.compress(combined_map.tobytes())
    seg_bytes = len(seg_compressed)

    n_persons_seg = person_mask_full.sum()
    n_fire_pixels = fire_mask.sum()
    fire_pct = n_fire_pixels / (h * w) * 100
    results['segmentation'] = {'size': seg_bytes, 'combined_map': combined_map,
                               'fire_mask': fire_mask, 'person_mask': person_mask_full}
    print(f"    Fire area: {fire_pct:.1f}% of image ({n_fire_pixels:,} pixels)")
    print(f"    Person pixels: {n_persons_seg:,}")
    print(f"    Size (compressed map): {seg_bytes/1024:.1f} KB "
          f"(compression: {raw_size/seg_bytes:,.0f}x)")
    del seg_model
    torch.cuda.empty_cache()

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Original
    axes[0, 0].imshow(np.array(img))
    axes[0, 0].set_title('Original Image\n(28.3 MB)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Caption — show text as overlay box at bottom of image
    axes[0, 1].imshow(np.array(img))
    axes[0, 1].set_title(f'Captioning (BLIP) — {caption_bytes} bytes',
                         fontsize=14, fontweight='bold')
    # Wrap long caption text inside image
    axes[0, 1].text(0.5, 0.03, f'"{caption}"',
                    transform=axes[0, 1].transAxes, ha='center', va='bottom',
                    fontsize=10, color='white', style='italic',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
                    wrap=True)
    axes[0, 1].axis('off')

    # Detection — person only
    axes[1, 0].imshow(np.array(img))
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.tolist()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor='lime', facecolor='none')
        axes[1, 0].add_patch(rect)
        axes[1, 0].text(x1, y1 - 5, f"person {score:.2f}",
                        color='lime', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    axes[1, 0].set_title(f'Person Detection (Faster R-CNN)\n'
                         f'{len(det_objects)} persons — {det_bytes} bytes',
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Fire + Person segmentation overlay
    img_arr = np.array(img).astype(np.float32) / 255.0
    dimmed = img_arr * 0.3
    overlay = dimmed.copy()
    combined_map = results['segmentation']['combined_map']
    # Person = green, Fire = red-orange
    person_mask = combined_map == 1
    fire_mask_vis = combined_map == 2
    overlay[person_mask] = np.array([0.0, 1.0, 0.3]) * 0.7 + img_arr[person_mask] * 0.3
    overlay[fire_mask_vis] = np.array([1.0, 0.3, 0.0]) * 0.7 + img_arr[fire_mask_vis] * 0.3
    axes[1, 1].imshow(np.clip(overlay, 0, 1))
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=[0.0, 1.0, 0.3], label='Person'),
        Patch(facecolor=[1.0, 0.3, 0.0], label='Fire'),
    ]
    axes[1, 1].legend(handles=legend_patches, loc='lower right', fontsize=10,
                      framealpha=0.9, facecolor='white')
    axes[1, 1].set_title(f'Fire + Person Segmentation\n'
                         f'Fire: {fire_pct:.1f}% area, '
                         f'Person: {person_mask.sum():,}px — {seg_bytes/1024:.1f} KB',
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    fig.suptitle('Stage 4: Semantic Understanding — What Can Be Transmitted Instead of Pixels',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.25, wspace=0.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, 'stage4_semantic.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # --- Transmission size comparison bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = {
        'Raw Image': raw_size,
        'JPEG (q=85)': os.path.getsize(image_path) if image_path.endswith('.jpg') else raw_size // 25,
        f'JSCC\n({config["bottleneck_ch"]}ch)': (
            (-(-h // config['patch_size'])) * (-(-w // config['patch_size']))
            * config['bottleneck_ch'] * 4 * 4 * 4),
        'Fire+Person\n(Seg+HSV)': seg_bytes,
        'Person Det.\n(Faster R-CNN)': det_bytes,
        'Caption\n(BLIP)': caption_bytes,
    }
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#9b59b6', '#3498db', '#1abc9c']
    names = list(methods.keys())
    values = [v / 1024 for v in methods.values()]
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Transmission Size (KB)', fontsize=13)
    ax.set_title('Full Spectrum: Raw Pixels → Semantic Understanding',
                 fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    for bar, (name, size) in zip(bars, methods.items()):
        ratio = raw_size / size
        if size >= 1024 * 1024:
            label = f"{size/1024/1024:.1f} MB"
        elif size >= 1024:
            label = f"{size/1024:.1f} KB"
        else:
            label = f"{size} B"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                f"{label}\n({ratio:,.0f}x)",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylim(top=values[0] * 5)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'stage4_bandwidth.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Semantic Communication Demo')
    parser.add_argument('--model', default='jscc_model.pt', help='Pretrained model path')
    parser.add_argument('--image', default='image_4k.jpg', help='Input image')
    parser.add_argument('--snr', type=int, nargs='+', default=[0, 5, 10, 15, 20],
                        help='SNR values to evaluate (dB)')
    parser.add_argument('--backend', choices=['fp32', 'fp16', 'tensorrt'], default='fp16',
                        help='Inference backend: fp32, fp16 (default), or tensorrt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model_path = os.path.join(SCRIPT_DIR, args.model)
    image_path = os.path.join(SCRIPT_DIR, args.image)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    output_dir = os.path.join(SCRIPT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("=== Semantic Communication Demo ===")
    print(f"Image: {image_path}")
    print(f"Device: {args.device}")
    print(f"Backend: {args.backend}")
    print(f"Output: {output_dir}")

    # Load pretrained model
    model, checkpoint = load_model(model_path, args.device)

    config = {
        'device': args.device,
        'backend': args.backend,
        'patch_size': checkpoint['patch_size'],
        'bottleneck_ch': checkpoint['bottleneck_ch'],
        'snr_eval_list': args.snr,
    }

    run_stage1(image_path, config, output_dir)
    run_stage2(image_path, model, config, output_dir)
    run_stage3(image_path, model, config, output_dir)
    run_stage4(image_path, config, output_dir)

    print("\n" + "=" * 60)
    print("Demo complete! Output:")
    for f in ['stage1_bandwidth.png', 'stage2_comparison.png',
              'stage2_metrics.png', 'stage3_speed.png', 'benchmark.md',
              'stage4_semantic.png', 'stage4_bandwidth.png']:
        print(f"  - {os.path.join(output_dir, f)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
