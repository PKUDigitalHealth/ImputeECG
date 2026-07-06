"""
Kailun standard-format CSV: 12 columns in PTB-XL lead order; fs = num_rows/10;
resample to 500 Hz; multiply amplitude by _AMPLITUDE_SCALE (default 100) after resample for plot / model input.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Same order as unet_1d/utils/plot.py and scripts/plot_ptbxl_masked_sample.py
LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

_DEFAULT_CSV = (
    "/nvme2/chenggf/fangxiaocheng/ImputeECG/ecg_kailun_12/data/数字化/"
    "人工数字化调整1w/第八次/范各庄3x4/标准格式/"
    "艾召林-1976-03-27-男-2021-07-24-09-32-36-范各庄.csv"
)
_DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "kailuan"
_FS_OUT = 500.0
_AMPLITUDE_SCALE = 100.0


def load_kailun_csv(path: str | Path) -> np.ndarray:
    """Load headerless CSV as (T, 12) float64."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    x = np.loadtxt(p, delimiter=",", dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 12:
        raise ValueError(f"expected 12 columns, got shape={x.shape}")
    return x


def estimate_fs_hz(num_rows: int) -> float:
    """Assume ~10 s record length: fs = num_rows / 10."""
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    return float(num_rows) / 10.0


def resample_to_fs(x: np.ndarray, fs_in: float, fs_out: float = _FS_OUT) -> np.ndarray:
    """x: (T, 12); linear interp each column to fs_out; length round(T * fs_out / fs_in)."""
    t, c = x.shape
    if c != 12:
        raise ValueError(f"expected (T, 12), got {x.shape}")
    t_new = max(1, int(round(t * fs_out / fs_in)))
    t_old = np.arange(t, dtype=np.float64) / fs_in
    t_new_axis = np.arange(t_new, dtype=np.float64) / fs_out
    out = np.empty((t_new, c), dtype=np.float64)
    for j in range(c):
        out[:, j] = np.interp(t_new_axis, t_old, x[:, j])
    return out


def plot_kailun_12lead(
    x_500: np.ndarray,
    out_path: str | Path,
    *,
    fs: float = _FS_OUT,
    title: str = "",
) -> None:
    """x_500: (T, 12), columns match LEAD_NAMES."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if x_500.ndim != 2 or x_500.shape[1] != 12:
        raise ValueError(f"expected (T, 12), got {x_500.shape}")

    t_axis = np.arange(x_500.shape[0], dtype=np.float64) / fs
    fig, axes = plt.subplots(
        12,
        1,
        sharex=True,
        figsize=(10, 14),
        constrained_layout=True,
    )
    for i, ax in enumerate(axes):
        ax.plot(t_axis, x_500[:, i], color="k", linewidth=0.6)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(labelleft=False, left=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(0.5)

    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title, fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_kailun_12lead_input_recon(
    original: np.ndarray,
    reconstructed: np.ndarray,
    out_path: str | Path,
    *,
    fs: float = _FS_OUT,
    title: str = "",
    amplitude_scale: float | None = None,
) -> None:
    """
    12x2 grid: left = scaled original (e.g. x100), right = model output (same units, plot as-is).
    Both (T, 12), PTB-XL column order.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if original.shape != reconstructed.shape:
        raise ValueError(f"shape mismatch {original.shape} vs {reconstructed.shape}")
    if original.ndim != 2 or original.shape[1] != 12:
        raise ValueError(f"expected (T, 12), got {original.shape}")

    t_axis = np.arange(original.shape[0], dtype=np.float64) / fs
    fig, axes = plt.subplots(
        12,
        2,
        sharex=True,
        figsize=(14, 14),
        constrained_layout=True,
    )
    for i in range(12):
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]
        y_l = original[:, i]
        y_r = reconstructed[:, i]
        lo = float(min(np.min(y_l), np.min(y_r)))
        hi = float(max(np.max(y_l), np.max(y_r)))
        pad = (hi - lo) * 0.05 + 1e-9
        y0, y1 = lo - pad, hi + pad

        ax_l.plot(t_axis, y_l, color="#0044ff", linewidth=0.6)
        ax_r.plot(t_axis, y_r, color="#ff0000", linewidth=0.6)
        ax_l.set_ylim(y0, y1)
        ax_r.set_ylim(y0, y1)
        ax_l.set_ylabel(LEAD_NAMES[i], rotation=0, ha="right", va="center", fontsize=9)
        for ax in (ax_l, ax_r):
            ax.tick_params(labelleft=False, left=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_linewidth(0.5)
        if i == 0:
            if amplitude_scale is not None and amplitude_scale != 1.0:
                ax_l.set_title(f"Original (x{amplitude_scale:g})", fontsize=10)
            else:
                ax_l.set_title("Original (resampled)", fontsize=10)
            ax_r.set_title("Reconstructed", fontsize=10)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title, fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def run(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    fs_out: float = _FS_OUT,
    amplitude_scale: float = _AMPLITUDE_SCALE,
) -> tuple[float, int, int]:
    """Load, estimate fs, resample, multiply amplitude, plot. Returns (fs_in, T_in, T_out)."""
    x = load_kailun_csv(csv_path)
    fs_in = estimate_fs_hz(x.shape[0])
    x_out = resample_to_fs(x, fs_in, fs_out) * amplitude_scale
    stem = Path(csv_path).stem
    title = f"Kailun 12-lead x{amplitude_scale:g}; fs_in={fs_in:.2f} Hz -> {fs_out:.0f} Hz\n{stem}"
    plot_kailun_12lead(x_out, out_path, fs=fs_out, title=title)
    return fs_in, x.shape[0], x_out.shape[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Kailun CSV -> 500 Hz -> 12-lead PNG (PTB-XL order)")
    p.add_argument("--csv", default=_DEFAULT_CSV, help="input CSV path")
    p.add_argument(
        "--out",
        default="",
        help="output PNG path; default: unet_1d/kailuan/<csv_stem>.png",
    )
    p.add_argument("--out_dir", default=str(_DEFAULT_OUT_DIR), help="default output directory")
    p.add_argument("--fs_out", type=float, default=_FS_OUT, help="target sample rate Hz (default 500)")
    p.add_argument(
        "--scale",
        type=float,
        default=_AMPLITUDE_SCALE,
        help="amplitude multiplier after resample (default 100)",
    )
    args = p.parse_args()

    csv_p = Path(args.csv)
    if args.out:
        out = Path(args.out)
    else:
        out_dir = Path(args.out_dir)
        out = out_dir / f"{csv_p.stem}.png"

    fs_in, tin, tout = run(csv_p, out, fs_out=args.fs_out, amplitude_scale=args.scale)
    print(f"fs_in={fs_in:.4f} Hz, T_in={tin}, T_out={tout}, saved {out}")


if __name__ == "__main__":
    main()
