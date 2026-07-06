"""
CPSC2018：在缺失位置上计算补全结果的 MAE 和 RMSE。
"""
import argparse
import os
import numpy as np
from pathlib import Path

_DEFAULT_GT = "/nvme2/chenggf/fangxiaocheng/ImputeECG/data/cpsc2018/test_data.npy"
_DEFAULT_DATA_DIR = "/nvme2/chenggf/fangxiaocheng/ImputeECG/data/cpsc2018"
_DEFAULT_INFERENCE_DIR = (
    Path(__file__).resolve().parent.parent / "inference_output_cpsc2018"
)


def main() -> None:
    p = argparse.ArgumentParser(description="CPSC2018 GT vs imputed：缺失区域 MAE/RMSE")
    p.add_argument("--gt", default=_DEFAULT_GT, help="GT npy，(N,T,12)")
    p.add_argument("--data_dir", default=_DEFAULT_DATA_DIR, help="masked npy 目录")
    p.add_argument(
        "--inference_dir",
        default=str(_DEFAULT_INFERENCE_DIR),
        help="inference_cpsc2018 输出根目录",
    )
    p.add_argument("--sentinel", type=float, default=65535.0, help="缺失值标记")
    args = p.parse_args()

    if not os.path.isfile(args.gt):
        raise SystemExit(f"找不到 GT: {args.gt}")

    print(f"加载 GT: {args.gt}")
    gt = np.load(args.gt, mmap_mode="r")

    inference_dir = Path(args.inference_dir)
    data_dir = Path(args.data_dir)
    
    test_dirs = [
        "test_data_masked_12x1lead_10s",
        "test_data_masked_2x6lead_5s",
        "test_data_masked_4x3lead_2p5s"
    ]

    for t_dir in test_dirs:
        fake_path = inference_dir / t_dir / "imputed.npy"
        masked_path = data_dir / f"{t_dir}.npy"
        
        if not fake_path.is_file():
            print(f"跳过 {t_dir}，找不到 fake 文件: {fake_path}")
            continue
        if not masked_path.is_file():
            print(f"跳过 {t_dir}，找不到 masked 文件: {masked_path}")
            continue
            
        print(f"处理 {t_dir} ...")
        fk = np.load(fake_path, mmap_mode="r")
        masked = np.load(masked_path, mmap_mode="r")

        if gt.shape != fk.shape or gt.shape != masked.shape:
            print(f"  形状不一致: GT {gt.shape}, fake {fk.shape}, masked {masked.shape}")
            continue

        missing_mask = masked == args.sentinel

        if not np.any(missing_mask):
            print(f"  没有找到缺失部分 (sentinel={args.sentinel})")
            continue

        diff = np.asarray(fk[missing_mask], dtype=np.float64) - np.asarray(
            gt[missing_mask], dtype=np.float64
        )
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        
        print(f"  缺失点数量: {np.sum(missing_mask)}")
        print(f"  Masked MAE : {mae:.4f}")
        print(f"  Masked RMSE: {rmse:.4f}\n")

if __name__ == "__main__":
    main()
