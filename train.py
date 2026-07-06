import math
import sys
import argparse
import os
import datetime
import time
import json
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datasets import PTBXLDataset
from models import mae_vit_base_patch50
from utils import adjust_learning_rate, NativeScalerWithGradNormCount


def obs_from_sentinel(obs: torch.Tensor, sentinel: float) -> tuple[torch.Tensor, torch.Tensor]:
    """将数据里的缺失标记（如 65535）转为洞内置 0 的输入与逐点缺失掩码 [B,C,L]，1=缺失。"""
    miss = (obs == sentinel).to(torch.bool)
    x = torch.where(obs == sentinel, torch.zeros_like(obs), obs)
    return x, miss


def extra_random_time_holes(obs: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """每 (batch, 导联) 随机连续窗置 0；返回 (x, extra_missing)，extra_missing 为 bool。"""
    x = obs.clone()
    B, C, L = x.shape
    device = obs.device
    k = max(1, int(L * ratio))
    k = min(k, L)
    max_start = L - k
    starts = torch.randint(0, max_start + 1, (B, C), device=device, dtype=torch.long)
    offs = torch.arange(k, device=device, dtype=torch.long).view(1, 1, -1)
    t_idx = starts.unsqueeze(-1) + offs
    extra_missing = torch.zeros(B, C, L, device=device, dtype=torch.bool)
    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, C, k)
    c_idx = torch.arange(C, device=device).view(1, C, 1).expand(B, C, k)
    extra_missing[b_idx, c_idx, t_idx] = True
    x.scatter_(2, t_idx, 0.0) # 网络输入需要置 0
    return x, extra_missing


def get_args_parser():
    parser = argparse.ArgumentParser('MAE 1D pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--mask_ratio', default=0.20, type=float,
                        help='每个导联再叠加一段连续置 0，窗长约 ratio*L；各导联窗起点独立随机。')
    parser.add_argument('--missing_sentinel', default=65535.0, type=float,
                        help='观测 npy 中该值表示缺失；会置 0 进网络并计入 missing 掩码。无该值时仅随机洞起作用。')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/nvme2/chenggf/fangxiaocheng/ImputeECG/data/ptbxl', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to save train_log.txt')
    parser.add_argument('--gpu', default=1, type=int,
                        help='CUDA device id; force single-GPU training on this card')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser

def main(args):
    # Setup logging (must run before any logging.info)
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = os.path.join(args.log_dir, 'train_log.txt')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    if not torch.cuda.is_available():
        logging.error('CUDA is not available; this script expects a GPU.')
        sys.exit(1)
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        logging.error('Invalid --gpu %s (visible CUDA devices: %d)', args.gpu, torch.cuda.device_count())
        sys.exit(1)

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    logging.info(f"Starting MAE pre-training with arguments: {args}")
    logging.info('Single-GPU mode on %s', device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Setup Dataset
    dataset_train = PTBXLDataset(args.data_path, mode='train')
    logging.info(
        "训练数据: train_data_gt.npy + train_data_mask.npy。观测数据中 65535 视为缺失，模型将预测缺失部分并在缺失部分计算主要损失。"
    )

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Setup Model
    model = mae_vit_base_patch50()
    model.to(device)
    model_without_ddp = model

    logging.info("Model = %s" % str(model_without_ddp))

    # Calculate absolute learning rate
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256

    logging.info("base lr: %.2e" % (args.lr * 256 / args.batch_size))
    logging.info("actual lr: %.2e" % args.lr)

    logging.info("accumulate grad iter: %d" % args.accum_iter)
    logging.info("effective batch size: %d" % (args.batch_size * args.accum_iter))

    # Setup Optimizer
    param_groups = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and len(p.shape) >= 2], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and len(p.shape) == 1], "weight_decay": 0.}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScalerWithGradNormCount()

    logging.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # epoch 从 1 计到 args.epochs；学习率调度仍用 0-based 连续值
    for epoch in range(1, args.epochs + 1):
        model.train()
        metric_logger = {"loss": 0.0, "lr": 0.0}
        
        optimizer.zero_grad()
        
        for data_iter_step, batch in enumerate(data_loader_train):
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % args.accum_iter == 0:
                lr_epoch = (epoch - 1) + data_iter_step / len(data_loader_train)
                adjust_learning_rate(optimizer, lr_epoch, args)

            if isinstance(batch, (list, tuple)):
                samples_gt, samples_obs = batch
                samples_gt = samples_gt.to(device, non_blocking=True)
                samples_obs = samples_obs.to(device, non_blocking=True)
                
                # 1. 提取 sentinel 掩码并将 obs 中的 sentinel 置 0
                x_base, natural_miss = obs_from_sentinel(samples_obs, args.missing_sentinel)
                
                # 2. 额外叠加随机时间洞
                x_in, extra_miss = extra_random_time_holes(x_base, args.mask_ratio)
                
                # 3. 合并两次的缺失掩码
                missing_mask = natural_miss | extra_miss
                
                # 将 mask 作为额外通道拼接入输入
                x_in = torch.cat([x_in, missing_mask.float()], dim=1)
                
                # 防护：如果 GT 中本身也存在缺失标记，在计算 loss 时平方会超出 float16 上限导致 inf，故提前置 0
                samples_gt = torch.where(samples_gt == args.missing_sentinel, torch.zeros_like(samples_gt), samples_gt)
            else:
                samples_gt = batch.to(device, non_blocking=True)
                
                # 1. 直接在 GT 上做额外随机时间洞（无 sentinel 掩码）
                x_in, extra_miss = extra_random_time_holes(samples_gt, args.mask_ratio)
                missing_mask = extra_miss

                # 将 mask 作为额外通道拼接入输入
                x_in = torch.cat([x_in, missing_mask.float()], dim=1)
                
                # 防护：如果 GT 中本身也存在缺失标记，在计算 loss 时平方会超出 float16 上限导致 inf，故提前置 0
                samples_gt = torch.where(samples_gt == args.missing_sentinel, torch.zeros_like(samples_gt), samples_gt)

            with torch.amp.autocast('cuda'):
                loss, _, _ = model.forward_inpaint(x_in, samples_gt, missing_mask)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                logging.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= args.accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % args.accum_iter == 0)
            
            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

            metric_logger["loss"] += loss_value
            metric_logger["lr"] = optimizer.param_groups[0]["lr"]

            if (data_iter_step + 1) % 50 == 0:
                avg_loss = metric_logger["loss"] / 50
                log_msg = f"Epoch: [{epoch}] [{data_iter_step}/{len(data_loader_train)}]  loss: {avg_loss:.4f}  lr: {metric_logger['lr']:.6f}"
                logging.info(log_msg)
                
                metric_logger["loss"] = 0.0

        # 每 10 个 epoch 保存（10, 20, …），最后一个 epoch 必存
        if args.output_dir and (epoch % 10 == 0 or epoch == args.epochs):
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{epoch}.pth')
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)