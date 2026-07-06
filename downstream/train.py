import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PTBXLDataset
from net1d import Net1D
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = PTBXLDataset(os.path.join(args.data_dir, 'train_data.npy'), 
                                 os.path.join(args.data_dir, 'train_labels.npy'))
    val_dataset = PTBXLDataset(os.path.join(args.data_dir, 'val_data.npy'), 
                               os.path.join(args.data_dir, 'val_labels.npy'))
                               
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize model
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
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_auc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for x, y in train_bar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for x, y in val_bar:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                
                all_preds.append(torch.sigmoid(out).cpu().numpy())
                all_labels.append(y.cpu().numpy())
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        val_loss /= len(val_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        try:
            val_auc = roc_auc_score(all_labels, all_preds, average='macro')
        except ValueError:
            val_auc = 0.0
            
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model with Val AUC: {best_val_auc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/nvme2/chenggf/fangxiaocheng/ImputeECG/data/PTBXL-500-all/all/data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='downstream/weights/best_model.pth')
    args = parser.parse_args()
    train(args)
