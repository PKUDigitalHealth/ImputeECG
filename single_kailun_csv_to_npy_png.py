"""
单个 Kailuan CSV -> ImputeECG 补全 -> npy + PNG。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from batch_kailun_csv_to_npy import load_model
from test_csv_input import fill_missing_values, impute_ecg
from utils.plot_kailuan import (
    _FS_OUT,
    estimate_fs_hz,
    load_kailun_csv,
    plot_kailun_12lead_input_recon,
    resample_to_fs,
)


def csv_to_fake_and_input(
    csv_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    fs_out: float,
    missing_sentinel: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    raw = load_kailun_csv(csv_path)
    fs_in = estimate_fs_hz(raw.shape[0])

    raw_mask = (raw == missing_sentinel).astype(np.float32)
    raw_filled = fill_missing_values(raw, missing_sentinel)
    x500_filled = resample_to_fs(raw_filled, fs_in, fs_out).astype(np.float32)

    if x500_filled.shape[0] != 5000:
        raise ValueError(f"length is {x500_filled.shape[0]} not 5000")

    mask500 = resample_to_fs(raw_mask, fs_in, fs_out)
    miss_np = (mask500 > 0.5).astype(np.float32)
    x_in_np = np.where(miss_np == 1.0, 0.0, x500_filled).astype(np.float32)

    data_obs_np = np.where(miss_np == 1.0, missing_sentinel, x_in_np)
    data_obs_np = data_obs_np[np.newaxis, ...]
    data_obs_np = np.transpose(data_obs_np, (0, 2, 1))

    fake = impute_ecg(model, data_obs_np, device, sentinel=missing_sentinel)
    fake_np = fake.cpu().numpy()
    fake_np = np.transpose(fake_np, (0, 2, 1))
    return fake_np[0].astype(np.float32), x_in_np, fs_in


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单个 12 导联 CSV -> MAE 补全 -> npy + PNG"
    )
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练好的 checkpoint-*.pth（含 model state_dict）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="输出目录；默认与 CSV 同目录",
    )
    parser.add_argument("--npy", type=str, default="", help="指定输出 npy 路径")
    parser.add_argument("--png", type=str, default="", help="指定输出 png 路径")
    parser.add_argument("--fs_out", type=float, default=_FS_OUT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--missing_sentinel", type=float, default=65535.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv).resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else csv_path.parent
    npy_path = Path(args.npy).resolve() if args.npy else out_dir / f"{csv_path.stem}.npy"
    png_path = (
        Path(args.png).resolve() if args.png else out_dir / f"{csv_path.stem}_recon.png"
    )

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    model = load_model(args.checkpoint, device)

    try:
        fake, x_in, fs_in = csv_to_fake_and_input(
            csv_path,
            model,
            device,
            args.fs_out,
            args.missing_sentinel,
        )
    except Exception as e:
        print(f"Error {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, fake)

    title = (
        f"MAE inpaint; input = resampled (holes->0); "
        f"fs_in={fs_in:.2f} Hz -> {args.fs_out:.0f} Hz\n{csv_path.name}"
    )
    plot_kailun_12lead_input_recon(
        x_in.astype(np.float64),
        fake.astype(np.float64),
        png_path,
        fs=args.fs_out,
        title=title,
        amplitude_scale=None,
    )

    print(f"Saved npy: {npy_path} shape={fake.shape}")
    print(f"Saved png: {png_path}")


if __name__ == "__main__":
    main()
