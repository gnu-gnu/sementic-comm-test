"""
Semantic Communication — Model Definitions & Utilities
"""

import io
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from scipy.special import erfc


# ============================================================
# Model Definitions
# ============================================================
def power_normalize(x):
    """Normalize transmitted signal to unit average power."""
    b = x.shape[0]
    x_flat = x.view(b, -1)
    power = (x_flat ** 2).mean(dim=1, keepdim=True)
    return (x_flat / torch.sqrt(power + 1e-8)).view_as(x)


class SemanticEncoder(nn.Module):
    def __init__(self, bottleneck_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, bottleneck_ch, 3, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class AWGNChannel(nn.Module):
    def forward(self, x, snr_db):
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = 1.0 / (2.0 * snr_linear) ** 0.5
        noise = torch.randn_like(x) * noise_std
        return x + noise


class SemanticDecoder(nn.Module):
    def __init__(self, bottleneck_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_ch, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class SemanticCommSystem(nn.Module):
    def __init__(self, bottleneck_ch=16):
        super().__init__()
        self.encoder = SemanticEncoder(bottleneck_ch)
        self.channel = AWGNChannel()
        self.decoder = SemanticDecoder(bottleneck_ch)

    def forward(self, x, snr_db):
        z = self.encoder(x)
        z_norm = power_normalize(z)
        z_noisy = self.channel(z_norm, snr_db)
        x_hat = self.decoder(z_noisy)
        return x_hat


# ============================================================
# Metrics
# ============================================================
def compute_psnr(original, reconstructed):
    mse = F.mse_loss(original, reconstructed)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def _gaussian_window(size, sigma, channels, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return window


def compute_ssim(img1, img2, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ============================================================
# Data Utilities
# ============================================================
def extract_patches(image_path, patch_size=128, stride=64):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(img)
    patches = img_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, patch_size, patch_size)
    print(f"  Extracted {patches.shape[0]} patches ({patch_size}x{patch_size}, stride={stride})")
    return patches


# ============================================================
# Channel Simulation Utilities
# ============================================================
def snr_to_ber(snr_db):
    """BPSK over AWGN: BER = 0.5 * erfc(sqrt(SNR_linear / 2))"""
    snr_linear = 10 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(snr_linear / 2))


def corrupt_bitstream(data_bytes, ber):
    """Flip bits in a byte array with given bit error rate."""
    if ber <= 0:
        return data_bytes
    data = bytearray(data_bytes)
    n_bits = len(data) * 8
    n_errors = int(n_bits * ber)
    if n_errors == 0:
        return bytes(data)
    error_positions = np.random.choice(n_bits, size=n_errors, replace=False)
    for pos in error_positions:
        byte_idx = pos // 8
        bit_idx = pos % 8
        data[byte_idx] ^= (1 << bit_idx)
    return bytes(data)


def _prepare_tiles(image_path, patch_size, device):
    """Load image, pad, and extract non-overlapping tiles as a batch."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(img)
    _, H, W = img_tensor.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, pH, pW = img_padded.shape

    tiles_h, tiles_w = pH // patch_size, pW // patch_size
    tiles = img_padded.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    tiles = tiles.permute(1, 2, 0, 3, 4).reshape(-1, 3, patch_size, patch_size).to(device)

    return tiles, img_tensor, H, W, tiles_h, tiles_w, pH, pW


def _reassemble(tiles_hat, tiles_h, tiles_w, patch_size, H, W, pH, pW):
    """Reassemble tiles into a full image."""
    reconstructed = tiles_hat.reshape(tiles_h, tiles_w, 3, patch_size, patch_size)
    reconstructed = reconstructed.permute(2, 0, 3, 1, 4).reshape(3, pH, pW)
    return reconstructed[:, :H, :W]


@torch.no_grad()
def reconstruct_jscc(model, image_path, snr_db, patch_size, device, backend='fp16'):
    """
    Reconstruct image through JSCC system.

    backend:
        'fp32'      — Batch + FP32
        'fp16'      — Batch + FP16 (default)
        'tensorrt'  — TensorRT FP16 compiled
    """
    tiles, img_tensor, H, W, tiles_h, tiles_w, pH, pW = _prepare_tiles(
        image_path, patch_size, device)

    if backend == 'tensorrt':
        tiles_hat = _infer_tensorrt(model, tiles, snr_db)
    elif backend == 'fp32':
        tiles_hat = _infer_fp32(model, tiles, snr_db)
    else:
        tiles_hat = _infer_fp16(model, tiles, snr_db, device)

    reconstructed = _reassemble(tiles_hat, tiles_h, tiles_w, patch_size, H, W, pH, pW)
    return reconstructed, img_tensor


def _infer_fp32(model, tiles, snr_db):
    """Batch + FP32 inference."""
    model.float()
    return model(tiles.float(), snr_db).cpu()


def _infer_fp16(model, tiles, snr_db, device):
    """Batch + FP16 inference."""
    if device != 'cpu':
        tiles = tiles.half()
        model.half()
        result = model(tiles, snr_db).float().cpu()
        model.float()
        return result
    return model(tiles, snr_db).cpu()


# TensorRT compiled modules cache (in-memory)
_trt_cache = {}

# Default directory for saved TRT engines
TRT_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trt_engines')


def _compile_trt(module, sample_input, name):
    """Compile a module with TensorRT FP16. Uses disk cache, then in-memory cache."""
    if name in _trt_cache:
        return _trt_cache[name]

    import torch_tensorrt

    # Try loading pre-compiled engine from disk
    engine_path = os.path.join(TRT_ENGINE_DIR, f'{name}.ep')
    if os.path.exists(engine_path):
        print(f"  Loading TRT engine: {engine_path}")
        compiled = torch.export.load(engine_path).module()
        _trt_cache[name] = compiled
        return compiled

    # Compile from scratch
    print(f"  Compiling TRT engine: {name}...")
    compiled = torch_tensorrt.compile(
        module,
        ir='dynamo',
        inputs=[sample_input],
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
        min_block_size=1,
    )
    _trt_cache[name] = compiled
    return compiled


def save_trt_engines(model, n_tiles, patch_size, bottleneck_ch, device):
    """Pre-compile and save TRT engines for encoder and decoder."""
    import torch_tensorrt

    os.makedirs(TRT_ENGINE_DIR, exist_ok=True)

    # Encoder
    print("  Compiling encoder...")
    sample_tiles = torch.randn(n_tiles, 3, patch_size, patch_size, device=device)
    encoder_trt = torch_tensorrt.compile(
        model.encoder,
        ir='dynamo',
        inputs=[sample_tiles],
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
        min_block_size=1,
    )
    enc_path = os.path.join(TRT_ENGINE_DIR, 'encoder.ep')
    torch_tensorrt.save(encoder_trt, enc_path, inputs=[sample_tiles])
    print(f"  Saved: {enc_path}")

    # Decoder
    print("  Compiling decoder...")
    with torch.no_grad():
        sample_z = model.encoder(sample_tiles)
    decoder_trt = torch_tensorrt.compile(
        model.decoder,
        ir='dynamo',
        inputs=[sample_z],
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
        min_block_size=1,
    )
    dec_path = os.path.join(TRT_ENGINE_DIR, 'decoder.ep')
    torch_tensorrt.save(decoder_trt, dec_path, inputs=[sample_z])
    print(f"  Saved: {dec_path}")


def _infer_tensorrt(model, tiles, snr_db):
    """TensorRT FP16 inference."""
    encoder_trt = _compile_trt(model.encoder, tiles, 'encoder')
    z = encoder_trt(tiles)
    z_norm = power_normalize(z)

    # Channel (AWGN) runs in PyTorch — simple noise addition
    z_noisy = model.channel(z_norm, snr_db)

    decoder_trt = _compile_trt(model.decoder, z_noisy, 'decoder')
    tiles_hat = decoder_trt(z_noisy)
    return tiles_hat.float().cpu()


def jpeg_through_channel(image_path, snr_db, quality=85):
    """Simulate sending JPEG through a noisy channel."""
    img = Image.open(image_path).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    jpeg_bytes = buf.getvalue()
    jpeg_size = len(jpeg_bytes)

    ber = snr_to_ber(snr_db)

    header_end = jpeg_bytes.find(b'\xff\xda')
    if header_end == -1:
        header_end = min(600, len(jpeg_bytes) // 4)

    header = jpeg_bytes[:header_end]
    payload = jpeg_bytes[header_end:]
    corrupted_payload = corrupt_bitstream(payload, ber)
    corrupted_bytes = header + corrupted_payload

    try:
        corrupted_img = Image.open(io.BytesIO(corrupted_bytes)).convert('RGB')
        result_tensor = transforms.ToTensor()(corrupted_img)
    except Exception:
        w, h = img.size
        result_tensor = torch.rand(3, h, w)

    original_tensor = transforms.ToTensor()(img)
    return result_tensor, original_tensor, jpeg_size
