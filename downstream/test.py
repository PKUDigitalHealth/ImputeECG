import os
import csv
import pickle
import warnings
import torch
from torch.utils.data import DataLoader
from dataset import PTBXLDataset
from net1d import Net1D
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import argparse
from tqdm import tqdm


def _macro_auc_over_valid_classes(y_true, y_pred):
    """只在正负样本同时存在的类别上计算 AUC，再做宏平均。"""
    vals = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if np.unique(yt).size < 2:
            continue
        try:
            v = float(roc_auc_score(yt, y_pred[:, i]))
            if np.isfinite(v):
                vals.append(v)
        except ValueError:
            continue
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _macro_auprc_over_valid_classes(y_true, y_pred):
    """只在正负样本同时存在的类别上计算 AUPRC，再做宏平均。"""
    vals = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if np.unique(yt).size < 2:
            continue
        try:
            v = float(average_precision_score(yt, y_pred[:, i]))
            if np.isfinite(v):
                vals.append(v)
        except ValueError:
            continue
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _bootstrap_macro_ci(
    y_true,
    y_pred,
    n_bootstrap=1000,
    seed=42,
    alpha=0.95,
    min_valid=200,
):
    """对 macro AUC/AUPRC/F1 做 bootstrap 置信区间。"""
    n = y_true.shape[0]
    if n <= 1:
        return {
            "auc": None,
            "auprc": None,
            "f1": None,
            "valid_counts": {"auc": 0, "auprc": 0, "f1": 0},
        }

    rng = np.random.default_rng(seed)
    auc_vals = []
    auprc_vals = []
    f1_vals = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            auc_v = _macro_auc_over_valid_classes(yt, yp)
            if np.isfinite(auc_v):
                auc_vals.append(auc_v)

            auprc_v = _macro_auprc_over_valid_classes(yt, yp)
            if np.isfinite(auprc_v):
                auprc_vals.append(auprc_v)

        yp_bin = (yp > 0.5).astype(int)
        f1_v = float(f1_score(yt, yp_bin, average="macro", zero_division=0))
        if np.isfinite(f1_v):
            f1_vals.append(f1_v)

    lo_pct = (1.0 - alpha) / 2.0 * 100.0
    hi_pct = (1.0 + alpha) / 2.0 * 100.0

    def _ci(arr):
        if len(arr) < min_valid:
            return None
        return (
            float(np.percentile(arr, lo_pct)),
            float(np.percentile(arr, hi_pct)),
        )

    return {
        "auc": _ci(auc_vals),
        "auprc": _ci(auprc_vals),
        "f1": _ci(f1_vals),
        "valid_counts": {
            "auc": len(auc_vals),
            "auprc": len(auprc_vals),
            "f1": len(f1_vals),
        },
    }


def _fmt_ci(ci):
    if ci is None:
        return "N/A"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

def _load_label_names(path):
    """每行一个名称，顺序与标签列一致。"""
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                names.append(s)
    return names


def _load_label_names_from_mlb(path, dict_key=""):
    """
    从 pickle / joblib 读取 sklearn MultiLabelBinarizer（或含 classes_ 的对象）。
    顺序与训练时 mlb.fit 后的 classes_ 一致，应对齐 labels.npy 的列。
    """
    obj = None
    err_joblib = err_pickle = None
    try:
        import joblib

        obj = joblib.load(path)
    except Exception as e:
        err_joblib = e
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e2:
            err_pickle = e2
            raise RuntimeError(
                f"无法加载 MLB 文件 {path}: joblib: {err_joblib}; pickle: {err_pickle}"
            ) from err_pickle

    if dict_key:
        if not isinstance(obj, dict):
            raise TypeError(f"--mlb_key 要求根对象为 dict，实际为 {type(obj)}")
        if dict_key not in obj:
            raise KeyError(f"MLB 字典中无键 {dict_key!r}，现有键: {list(obj.keys())[:20]}...")
        obj = obj[dict_key]

    if hasattr(obj, "classes_"):
        classes = obj.classes_
    else:
        raise TypeError(
            f"期望 MultiLabelBinarizer 或含 classes_，实际类型: {type(obj)}"
        )
    arr = np.asarray(classes).ravel()
    return [str(x) for x in arr]


def _metrics_per_class(y_true, y_pred, disease_names):
    """y_true, y_pred: (N, C). 每类 AUC、AUPRC（写 CSV）。"""
    n_classes = y_true.shape[1]
    rows = []
    for i in range(n_classes):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        try:
            auc_i = float(roc_auc_score(yt, yp))
        except ValueError:
            auc_i = float("nan")
        try:
            auprc_i = float(average_precision_score(yt, yp))
        except ValueError:
            auprc_i = float("nan")
        name = disease_names[i] if i < len(disease_names) else str(i)
        rows.append(
            {
                "disease": name,
                "n_positive": int(yt.sum()),
                "auc": auc_i,
                "auprc": auprc_i,
            }
        )
    return rows


def save_metrics_csv(path, per_class_rows, macro_auc, macro_auprc, ci):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fieldnames = [
        "disease",
        "n_positive",
        "auc",
        "auprc",
        "auc_ci_low_95",
        "auc_ci_high_95",
        "auprc_ci_low_95",
        "auprc_ci_high_95",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_class_rows:
            w.writerow(
                {
                    "disease": r["disease"],
                    "n_positive": r["n_positive"],
                    "auc": f"{r['auc']:.6f}",
                    "auprc": f"{r['auprc']:.6f}",
                    "auc_ci_low_95": "",
                    "auc_ci_high_95": "",
                    "auprc_ci_low_95": "",
                    "auprc_ci_high_95": "",
                }
            )
        auc_ci = ci.get("auc")
        auprc_ci = ci.get("auprc")
        w.writerow(
            {
                "disease": "宏平均",
                "n_positive": "",
                "auc": f"{macro_auc:.6f}",
                "auprc": f"{macro_auprc:.6f}",
                "auc_ci_low_95": "" if auc_ci is None else f"{auc_ci[0]:.6f}",
                "auc_ci_high_95": "" if auc_ci is None else f"{auc_ci[1]:.6f}",
                "auprc_ci_low_95": "" if auprc_ci is None else f"{auprc_ci[0]:.6f}",
                "auprc_ci_high_95": "" if auprc_ci is None else f"{auprc_ci[1]:.6f}",
            }
        )

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data: {args.data_npy}")
    print(f"Labels: {args.label_npy}")

    test_dataset = PTBXLDataset(args.data_npy, args.label_npy)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = Net1D(
        in_channels=12, 
        base_filters=64, 
        ratio=1.0, 
        filter_list=[64, 128, 256, 512], 
        m_blocks_list=[2, 2, 2, 2], 
        kernel_size=15, 
        stride=2, 
        groups_width=16, 
        n_classes=71
    ).to(device)
    
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    test_bar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for x, y in test_bar:
            x, y = x.to(device), y.to(device)
            if args.missing_sentinel >= 0:
                x = torch.where(x >= args.missing_sentinel, torch.zeros_like(x), x)
            out = model(x)
            all_preds.append(torch.sigmoid(out).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        auc = _macro_auc_over_valid_classes(all_labels, all_preds)
        if not np.isfinite(auc):
            auc = 0.0

        auprc = _macro_auprc_over_valid_classes(all_labels, all_preds)
        if not np.isfinite(auprc):
            auprc = 0.0
        
    preds_binary = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)

    macro_auc = auc
    macro_auprc = auprc
    macro_f1 = f1
    ci = _bootstrap_macro_ci(
        all_labels,
        all_preds,
        n_bootstrap=max(1, int(getattr(args, "ci_bootstrap", 1000))),
        seed=int(getattr(args, "ci_seed", 42)),
        alpha=0.95,
        min_valid=max(1, int(getattr(args, "ci_min_valid", 200))),
    )
    n_classes = all_labels.shape[1]
    if args.mlb_path:
        disease_names = _load_label_names_from_mlb(args.mlb_path, args.mlb_key)
    elif args.label_names:
        disease_names = _load_label_names(args.label_names)
    else:
        disease_names = [f"class_{i}" for i in range(n_classes)]
    if len(disease_names) != n_classes:
        raise ValueError(
            f"疾病名称数量 {len(disease_names)} 与标签维度 {n_classes} 不一致，请检查 --mlb_path / --label_names"
        )

    per_class_rows = _metrics_per_class(all_labels, all_preds, disease_names)

    print("\n" + "="*30)
    print(f"Test Results:")
    print(f"AUC (Macro):   {macro_auc:.4f}")
    print(f"AUC 95% CI:    {_fmt_ci(ci['auc'])}")
    print(f"AUPRC (Macro): {macro_auprc:.4f}")
    print(f"AUPRC 95% CI:  {_fmt_ci(ci['auprc'])}")
    print(f"F1 Score (Macro): {macro_f1:.4f}")
    print(f"F1 95% CI:     {_fmt_ci(ci['f1'])}")
    print(
        "CI valid bootstrap: "
        f"AUC={ci['valid_counts']['auc']}, "
        f"AUPRC={ci['valid_counts']['auprc']}, "
        f"F1={ci['valid_counts']['f1']}"
    )
    print("="*30)

    if not args.no_csv:
        csv_path = args.csv_path
        if csv_path is None:
            mdir = os.path.dirname(os.path.abspath(args.model_path))
            csv_path = os.path.join(mdir if mdir else ".", "test_per_class_metrics.csv")
        save_metrics_csv(csv_path, per_class_rows, macro_auc, macro_auprc, ci)
        print(f"Per-class and macro metrics saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/nvme2/chenggf/fangxiaocheng/ImputeECG/data/PTBXL-500-all/all/data')
    parser.add_argument(
        '--data_npy',
        type=str,
        default='/nvme2/chenggf/fangxiaocheng/ImputeECG/mae_1d/inference_output/test_data_masked_12x1lead_10s/imputed.npy',
    )
    parser.add_argument(
        '--label_npy',
        type=str,
        default='/nvme2/chenggf/fangxiaocheng/ImputeECG/data/PTBXL-500-all/all/data/test_labels.npy',
    )
    parser.add_argument(
        '--missing_sentinel',
        type=float,
        default=65535.0,
        help='缺失哨兵值，推理前置 0；设为负数则不做转换',
    )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='downstream/weights/best_model.pth')
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help='CSV output path. Default: <model_dir>/test_per_class_metrics.csv',
    )
    parser.add_argument(
        '--mlb_path',
        type=str,
        default='/nvme2/chenggf/fangxiaocheng/ImputeECG/data/PTBXL-500-all/all/data/mlb.pkl',
        help='MultiLabelBinarizer 的 pickle/joblib，用 classes_ 作为 CSV 疾病列；优先于 --label_names',
    )
    parser.add_argument(
        '--mlb_key',
        type=str,
        default='',
        help='若 pickle 根对象是 dict，此项为其中存放 MultiLabelBinarizer 的键名',
    )
    parser.add_argument(
        '--label_names',
        type=str,
        default='',
        help='文本每行一个名称；--mlb_path 为空时用 class_0, class_1, ...',
    )
    parser.add_argument(
        '--ci_bootstrap',
        type=int,
        default=1000,
        help='用于 95% CI 的 bootstrap 次数',
    )
    parser.add_argument(
        '--ci_seed',
        type=int,
        default=42,
        help='bootstrap 随机种子',
    )
    parser.add_argument(
        '--ci_min_valid',
        type=int,
        default=1,
        help='计算 CI 所需的最小有效 bootstrap 次数，不足则输出 N/A',
    )
    parser.add_argument('--no_csv', action='store_true', help='Do not write metrics CSV')
    args = parser.parse_args()
    test(args)
