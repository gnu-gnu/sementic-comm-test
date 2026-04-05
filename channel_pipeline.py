"""
Channel Pipeline — Full Communication Chain E2E Simulation
============================================================
전통적 통신 체인 (JPEG + 채널 부호화 + 변조) vs 시멘틱 통신 (JSCC)을
동일한 채널 조건에서 비교합니다.

Traditional:
  Image → JPEG → Hamming(7,4) → [BPSK/QPSK/16-QAM] → AWGN → Demod → Hamming Dec → JPEG Dec

Semantic (JSCC):
  Image → CNN Encoder → Power Norm → AWGN → CNN Decoder → Reconstructed

Usage:
    python channel_pipeline.py
    python channel_pipeline.py --snr -10 -5 0 5 10 15 20
    python channel_pipeline.py --modulations bpsk qpsk 16qam
"""

import argparse
import io
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import (
    SemanticCommSystem, compute_psnr, compute_ssim,
    _prepare_tiles, _reassemble, power_normalize,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Modulation / Demodulation
# ============================================================
MODULATION_SCHEMES = {
    'bpsk': {'bits_per_symbol': 1, 'label': 'BPSK'},
    'qpsk': {'bits_per_symbol': 2, 'label': 'QPSK'},
    '16qam': {'bits_per_symbol': 4, 'label': '16-QAM'},
}


def _bits_to_bpsk(bits):
    """BPSK: 0 → -1, 1 → +1"""
    return 2.0 * bits - 1.0


def _bpsk_to_bits(symbols):
    return (symbols > 0).astype(np.float64)


def _bits_to_qpsk(bits):
    """QPSK: 2 bits → complex symbol, normalized to unit power."""
    bits = bits[:len(bits) // 2 * 2].reshape(-1, 2)
    i = 2.0 * bits[:, 0] - 1.0
    q = 2.0 * bits[:, 1] - 1.0
    return (i + 1j * q) / np.sqrt(2)


def _qpsk_to_bits(symbols):
    i_bits = (symbols.real > 0).astype(np.float64)
    q_bits = (symbols.imag > 0).astype(np.float64)
    return np.column_stack([i_bits, q_bits]).ravel()


def _gray_map_16qam():
    """Generate 16-QAM constellation with Gray coding."""
    # 4-bit Gray code mapping to I/Q levels: {-3, -1, +1, +3}
    gray_2bit = [0, 1, 3, 2]  # 00,01,11,10 → -3,-1,+1,+3
    levels = np.array([-3, -1, 1, 3])
    constellation = {}
    for i_idx in range(4):
        for q_idx in range(4):
            symbol_bits = (gray_2bit[i_idx] << 2) | gray_2bit[q_idx]
            constellation[symbol_bits] = (levels[i_idx] + 1j * levels[q_idx]) / np.sqrt(10)
    return constellation


_16QAM_MAP = _gray_map_16qam()
_16QAM_SYMBOLS = np.array([_16QAM_MAP[i] for i in range(16)])


def _bits_to_16qam(bits):
    """16-QAM: 4 bits → complex symbol."""
    n = len(bits) // 4 * 4
    bits = bits[:n].reshape(-1, 4)
    indices = (bits[:, 0] * 8 + bits[:, 1] * 4 + bits[:, 2] * 2 + bits[:, 3]).astype(int)
    return _16QAM_SYMBOLS[indices]


def _16qam_to_bits(symbols):
    """16-QAM hard decision: find closest constellation point."""
    # Compute distance to all 16 points
    dists = np.abs(symbols[:, None] - _16QAM_SYMBOLS[None, :]) ** 2
    indices = np.argmin(dists, axis=1)
    bits = np.array([[(idx >> 3) & 1, (idx >> 2) & 1, (idx >> 1) & 1, idx & 1]
                     for idx in indices], dtype=np.float64)
    return bits.ravel()


def modulate(bits, scheme):
    if scheme == 'bpsk':
        return _bits_to_bpsk(bits)
    elif scheme == 'qpsk':
        return _bits_to_qpsk(bits)
    elif scheme == '16qam':
        return _bits_to_16qam(bits)
    raise ValueError(f"Unknown modulation: {scheme}")


def demodulate(symbols, scheme):
    if scheme == 'bpsk':
        return _bpsk_to_bits(symbols.real if np.iscomplexobj(symbols) else symbols)
    elif scheme == 'qpsk':
        return _qpsk_to_bits(symbols)
    elif scheme == '16qam':
        return _16qam_to_bits(symbols)
    raise ValueError(f"Unknown modulation: {scheme}")


# ============================================================
# AWGN Channel (for modulated symbols)
# ============================================================
def awgn_channel(symbols, snr_db):
    """Add AWGN noise to modulated symbols. Handles real (BPSK) and complex (QPSK/QAM)."""
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = 1.0 / snr_linear
    if np.iscomplexobj(symbols):
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(*symbols.shape)
    return symbols + noise


# ============================================================
# Hamming(7,4) Channel Coding
# ============================================================
# Generator matrix G (4x7) and parity check matrix H (3x7)
_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=np.uint8)

# Syndrome to error position lookup (0 = no error)
_SYNDROME_TABLE = {}
for i in range(7):
    err = np.zeros(7, dtype=np.uint8)
    err[i] = 1
    syndrome = tuple(_H @ err % 2)
    _SYNDROME_TABLE[syndrome] = i


def hamming_encode(data_bits):
    """Encode data bits with Hamming(7,4). Input length must be multiple of 4."""
    n = len(data_bits)
    pad = (4 - n % 4) % 4
    if pad:
        data_bits = np.concatenate([data_bits, np.zeros(pad, dtype=np.uint8)])
    blocks = data_bits.reshape(-1, 4)
    encoded = (blocks @ _G) % 2
    return encoded.ravel(), pad


def hamming_decode(coded_bits, pad=0):
    """Decode Hamming(7,4) coded bits with single-error correction."""
    blocks = coded_bits.reshape(-1, 7).astype(np.uint8)
    decoded = np.zeros((len(blocks), 4), dtype=np.uint8)
    for i, block in enumerate(blocks):
        syndrome = tuple(_H @ block % 2)
        if syndrome in _SYNDROME_TABLE:
            block = block.copy()
            block[_SYNDROME_TABLE[syndrome]] ^= 1
        decoded[i] = block[:4]
    result = decoded.ravel()
    if pad:
        result = result[:-pad]
    return result


# ============================================================
# Traditional Communication Chain
# ============================================================
def traditional_chain(image_path, snr_db, modulation='bpsk', jpeg_quality=85):
    """Full traditional communication chain: JPEG → Hamming(7,4) → Modulation → AWGN → Demod → Decode → JPEG."""
    img = Image.open(image_path).convert('RGB')
    original = transforms.ToTensor()(img)

    # Source encoding: JPEG
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=jpeg_quality)
    jpeg_bytes = buf.getvalue()

    # Convert to bits
    bits = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8))

    # Channel encoding: Hamming(7,4)
    coded_bits, pad = hamming_encode(bits)
    code_rate = 4.0 / 7.0

    # Modulation
    symbols = modulate(coded_bits.astype(np.float64), modulation)

    # Channel: AWGN
    received = awgn_channel(symbols, snr_db)

    # Demodulation
    demod_bits = demodulate(received, modulation)

    # Channel decoding: Hamming
    decoded_bits = hamming_decode(demod_bits.astype(np.uint8), pad)

    # BER calculation
    ber = np.mean(bits != decoded_bits[:len(bits)])

    # Source decoding: JPEG
    decoded_bytes = np.packbits(decoded_bits[:len(bits)]).tobytes()
    try:
        decoded_img = Image.open(io.BytesIO(decoded_bytes)).convert('RGB')
        reconstructed = transforms.ToTensor()(decoded_img)
        if reconstructed.shape != original.shape:
            reconstructed = F.interpolate(
                reconstructed.unsqueeze(0),
                size=(original.shape[1], original.shape[2]),
                mode='bilinear', align_corners=False
            ).squeeze(0)
    except Exception:
        reconstructed = torch.rand_like(original)

    psnr = compute_psnr(original.unsqueeze(0), reconstructed.unsqueeze(0))
    ssim = compute_ssim(original.unsqueeze(0), reconstructed.unsqueeze(0))

    bits_per_symbol = MODULATION_SCHEMES[modulation]['bits_per_symbol']

    return {
        'image': reconstructed,
        'psnr': psnr,
        'ssim': ssim,
        'ber': ber,
        'source_bytes': len(jpeg_bytes),
        'coded_bits': len(coded_bits),
        'symbols': len(symbols),
        'bits_per_symbol': bits_per_symbol,
        'code_rate': code_rate,
        'spectral_efficiency': bits_per_symbol * code_rate,
    }


# ============================================================
# Semantic Communication Chain
# ============================================================
@torch.no_grad()
def semantic_chain(model, image_path, snr_db, patch_size, device):
    """JSCC semantic communication chain."""
    tiles, img_tensor, H, W, tiles_h, tiles_w, pH, pW = _prepare_tiles(
        image_path, patch_size, device)

    # Encode
    if device != 'cpu':
        tiles = tiles.half()
        model.half()

    z = model.encoder(tiles)
    z_norm = power_normalize(z)

    # Channel: AWGN
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = 1.0 / (2.0 * snr_linear) ** 0.5
    z_noisy = z_norm + torch.randn_like(z_norm) * noise_std

    # Decode
    tiles_hat = model.decoder(z_noisy).float().cpu()
    model.float()

    reconstructed = _reassemble(tiles_hat, tiles_h, tiles_w, patch_size, H, W, pH, pW)

    psnr = compute_psnr(img_tensor.unsqueeze(0), reconstructed.unsqueeze(0))
    ssim = compute_ssim(img_tensor.unsqueeze(0), reconstructed.unsqueeze(0))

    total_symbols = z_norm.numel()

    return {
        'image': reconstructed,
        'psnr': psnr,
        'ssim': ssim,
        'symbols': total_symbols,
    }


# ============================================================
# Visualization
# ============================================================
def plot_results(snr_list, trad_results, sem_results, modulations, output_dir):
    """Generate comparison plots."""

    # --- 1. PSNR vs SNR (all modulations + JSCC) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'bpsk': '#e74c3c', 'qpsk': '#e67e22', '16qam': '#9b59b6'}
    for mod in modulations:
        label = MODULATION_SCHEMES[mod]['label']
        psnr_vals = [trad_results[mod][s]['psnr'] for s in snr_list]
        ssim_vals = [trad_results[mod][s]['ssim'] for s in snr_list]
        ax1.plot(snr_list, psnr_vals, 's--', color=colors[mod], lw=2, ms=7,
                 label=f'Traditional ({label})')
        ax2.plot(snr_list, ssim_vals, 's--', color=colors[mod], lw=2, ms=7,
                 label=f'Traditional ({label})')

    psnr_sem = [sem_results[s]['psnr'] for s in snr_list]
    ssim_sem = [sem_results[s]['ssim'] for s in snr_list]
    ax1.plot(snr_list, psnr_sem, 'o-', color='#2ecc71', lw=2.5, ms=8, label='Semantic (JSCC)')
    ax2.plot(snr_list, ssim_sem, 'o-', color='#2ecc71', lw=2.5, ms=8, label='Semantic (JSCC)')

    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs Channel Quality', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('SSIM vs Channel Quality', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Full Communication Chain — Traditional (JPEG+Hamming+Modulation) vs Semantic (JSCC)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(output_dir, 'pipeline_metrics.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # --- 2. BER vs SNR ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod in modulations:
        label = MODULATION_SCHEMES[mod]['label']
        ber_vals = [trad_results[mod][s]['ber'] for s in snr_list]
        # Replace 0 with small value for log scale
        ber_plot = [max(b, 1e-7) for b in ber_vals]
        ax.semilogy(snr_list, ber_plot, 's-', color=colors[mod], lw=2, ms=7, label=label)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('BER vs SNR — Effect of Modulation Scheme', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_ber.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # --- 3. Visual comparison grid ---
    n_snr = len(snr_list)
    n_rows = 1 + len(modulations)  # modulations + JSCC
    fig, axes = plt.subplots(n_rows, n_snr, figsize=(4 * n_snr, 3.5 * n_rows))
    if n_snr == 1:
        axes = axes.reshape(-1, 1)

    for row, mod in enumerate(modulations):
        label = MODULATION_SCHEMES[mod]['label']
        for col, snr in enumerate(snr_list):
            r = trad_results[mod][snr]
            axes[row, col].imshow(r['image'].permute(1, 2, 0).clamp(0, 1).numpy())
            axes[row, col].set_title(
                f'Traditional ({label})\nSNR={snr}dB, BER={r["ber"]:.1e}\n'
                f'PSNR={r["psnr"]:.1f} / SSIM={r["ssim"]:.3f}', fontsize=8)
            axes[row, col].axis('off')

    for col, snr in enumerate(snr_list):
        r = sem_results[snr]
        axes[-1, col].imshow(r['image'].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[-1, col].set_title(
            f'Semantic (JSCC)\nSNR={snr}dB\n'
            f'PSNR={r["psnr"]:.1f} / SSIM={r["ssim"]:.3f}', fontsize=8)
        axes[-1, col].axis('off')

    fig.suptitle('Full Pipeline Comparison — Image Reconstruction at Each SNR',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(output_dir, 'pipeline_comparison.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # --- 4. Spectral efficiency comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    mod_labels = []
    se_vals = []
    psnr_at_10db = []
    for mod in modulations:
        label = MODULATION_SCHEMES[mod]['label']
        se = trad_results[mod][snr_list[0]]['spectral_efficiency']
        mod_labels.append(f'{label}\n({se:.2f} bps/Hz)')
        se_vals.append(se)
        # Find PSNR at SNR=10dB or closest
        closest_snr = min(snr_list, key=lambda s: abs(s - 10))
        psnr_at_10db.append(trad_results[mod][closest_snr]['psnr'])

    x = np.arange(len(modulations))
    bars = ax.bar(x, psnr_at_10db, color=[colors[m] for m in modulations],
                  edgecolor='white', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(mod_labels, fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'Spectral Efficiency vs Quality Trade-off (SNR={closest_snr}dB)',
                 fontsize=13, fontweight='bold')

    # Add JSCC reference line
    jscc_psnr = sem_results[closest_snr]['psnr']
    ax.axhline(y=jscc_psnr, color='#2ecc71', linestyle='--', lw=2,
               label=f'JSCC: {jscc_psnr:.1f}dB')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_spectral.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def save_report(snr_list, trad_results, sem_results, modulations, output_dir):
    """Save markdown report."""
    lines = [
        "# Channel Pipeline Report",
        "",
        "## Communication Chain",
        "",
        "### Traditional",
        "```",
        "Image → JPEG → Hamming(7,4) → [Modulation] → AWGN → Demod → Hamming Dec → JPEG Dec",
        "```",
        "",
        "### Semantic (JSCC)",
        "```",
        "Image → CNN Encoder → Power Norm → AWGN → CNN Decoder → Reconstructed",
        "```",
        "",
        "## Results",
        "",
    ]

    for mod in modulations:
        label = MODULATION_SCHEMES[mod]['label']
        se = trad_results[mod][snr_list[0]]['spectral_efficiency']
        lines += [
            f"### Traditional ({label}) — {se:.2f} bps/Hz",
            "",
            "| SNR (dB) | BER | PSNR (dB) | SSIM |",
            "|----------|-----|-----------|------|",
        ]
        for snr in snr_list:
            r = trad_results[mod][snr]
            lines.append(f"| {snr} | {r['ber']:.2e} | {r['psnr']:.2f} | {r['ssim']:.4f} |")
        lines.append("")

    lines += [
        "### Semantic (JSCC)",
        "",
        "| SNR (dB) | PSNR (dB) | SSIM |",
        "|----------|-----------|------|",
    ]
    for snr in snr_list:
        r = sem_results[snr]
        lines.append(f"| {snr} | {r['psnr']:.2f} | {r['ssim']:.4f} |")

    report_path = os.path.join(output_dir, 'pipeline_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Channel Pipeline E2E Simulation')
    parser.add_argument('--model', default='jscc_model.pt')
    parser.add_argument('--image', default='image_4k.jpg')
    parser.add_argument('--snr', type=int, nargs='+', default=[-10, -5, 0, 5, 10, 15, 20])
    parser.add_argument('--modulations', nargs='+', default=['bpsk', 'qpsk', '16qam'],
                        choices=['bpsk', 'qpsk', '16qam'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model_path = os.path.join(SCRIPT_DIR, args.model)
    image_path = os.path.join(SCRIPT_DIR, args.image)

    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    output_dir = os.path.join(SCRIPT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("=== Channel Pipeline E2E Simulation ===")
    print(f"Image: {image_path}")
    print(f"SNR: {args.snr}")
    print(f"Modulations: {', '.join(args.modulations)}")
    print(f"Output: {output_dir}")

    # Load JSCC model
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=True)
    model = SemanticCommSystem(bottleneck_ch=checkpoint['bottleneck_ch'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device).eval()
    patch_size = checkpoint['patch_size']
    print(f"JSCC model loaded: {model_path}")

    # --- Run traditional chains ---
    trad_results = {}
    for mod in args.modulations:
        label = MODULATION_SCHEMES[mod]['label']
        print(f"\n[Traditional — {label}]")
        trad_results[mod] = {}
        for snr in args.snr:
            r = traditional_chain(image_path, snr, modulation=mod)
            trad_results[mod][snr] = r
            print(f"  SNR={snr:3d}dB  BER={r['ber']:.2e}  "
                  f"PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}  "
                  f"SE={r['spectral_efficiency']:.2f} bps/Hz")

    # --- Run semantic chain ---
    print(f"\n[Semantic — JSCC]")
    sem_results = {}
    for snr in args.snr:
        r = semantic_chain(model, image_path, snr, patch_size, args.device)
        sem_results[snr] = r
        print(f"  SNR={snr:3d}dB  PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}")

    # --- Generate outputs ---
    print("\n[Generating plots...]")
    plot_results(args.snr, trad_results, sem_results, args.modulations, output_dir)
    save_report(args.snr, trad_results, sem_results, args.modulations, output_dir)

    print(f"\n{'='*60}")
    print(f"Done! Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
