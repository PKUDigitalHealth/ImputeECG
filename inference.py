import torch
import numpy as np
import argparse
import os
from models import mae_vit_base_patch50

def load_model(checkpoint_path, device):
    """
    加载训练好的 MAE 模型
    """
    model = mae_vit_base_patch50()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # 兼容 DDP 和 单卡模型的 key
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 如果保存时带了 'module.' 前缀，去掉它
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def impute_ecg(model, data_obs, device, sentinel=65535.0):
    """
    对包含缺失值的 ECG 信号进行补全
    data_obs: numpy array 或 torch Tensor，形状为 [B, C, L] (如 [1, 12, 5000])
              缺失部分的值应为 sentinel (默认 65535)
    返回补全后的信号，形状同输入。
    """
    if isinstance(data_obs, np.ndarray):
        data_obs = torch.from_numpy(data_obs).float()
        
    data_obs = data_obs.to(device)
    
    missing_mask = (data_obs == sentinel)
    
    # 将缺失部分置 0
    x_in = torch.where(missing_mask, torch.zeros_like(data_obs), data_obs)
    
    # 将 mask 作为额外通道拼接入输入，使其形状变为 [B, 24, L]
    x_in = torch.cat([x_in, missing_mask.float()], dim=1)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # 推理模式下不需要 target_gt
            _, imputed_signal, _ = model.forward_inpaint(x_in, target_gt=None, missing_mask=missing_mask)
        
    return imputed_signal

def main():
    parser = argparse.ArgumentParser(description="MAE 1D ECG Imputation Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help="训练好的模型权重路径 (.pth)")
    parser.add_argument('--missing_sentinel', type=float, default=65535.0, help="输入数据中表示缺失的特定值 (默认 65535.0)")
    parser.add_argument('--device', type=str, default='cuda:0', help="推理设备 (如 cuda:0 或 cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {args.checkpoint} onto {device}...")
    model = load_model(args.checkpoint, device)
    
    test_files = [
        "test_data_masked_12x1lead_10s",
        "test_data_masked_2x6lead_5s",
        "test_data_masked_4x3lead_2p5s"
    ]
    
    data_dir = "../data/ptbxl"
    output_base_dir = "inference_output"
    
    for test_name in test_files:
        input_data_path = os.path.join(data_dir, f"{test_name}.npy")
        
        print(f"\nProcessing {input_data_path}...")
        data = np.load(input_data_path)
        
        # 处理不同形状的数据，适配模型需要的 [B, C, L] 形状
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0) # 补齐 Batch 维度 -> [1, L, C] 或 [1, C, L]
            
        if data.shape[-1] == 12:
            # 如果是 [B, L, C] (如 [B, 5000, 12])，转换为 [B, C, L] (如 [B, 12, 5000])
            data = np.transpose(data, (0, 2, 1))
            was_transposed = True
        else:
            was_transposed = False
            
        print(f"Input shape for model: {data.shape}")
        
        print("Running imputation...")
        imputed_signal = impute_ecg(model, data, device, sentinel=args.missing_sentinel)
        
        # 转换回 numpy
        result_np = imputed_signal.cpu().numpy()
        
        if was_transposed:
            # 如果原来是 [B, 5000, 12]，把结果再转回去
            result_np = np.transpose(result_np, (0, 2, 1))
            
        # 创建输出目录
        output_dir = os.path.join(output_base_dir, test_name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "imputed.npy")
        print(f"Saving imputed result to {output_path}...")
        np.save(output_path, result_np)
        
    print("\nAll imputation tasks finished successfully!")

if __name__ == '__main__':
    main()
