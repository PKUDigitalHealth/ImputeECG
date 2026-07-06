import torch
import torch.nn as nn
import numpy as np
from .pos_embed import get_1d_sincos_pos_embed

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed1D(nn.Module):
    def __init__(self, seq_len=5000, patch_size=50, in_chans=12, embed_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        # 通道数加倍，后一半用于输入掩码
        self.proj = nn.Conv1d(in_chans * 2, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C*2, L)
        x = self.proj(x).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class MaskedAutoencoderViT1D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone for 1D signals
    """
    def __init__(self, seq_len=5000, patch_size=50, in_chans=12,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(seq_len, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x: [B, C, L]
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, L]
        pred: [N, num_patches, patch_size * C]
        mask: [N, num_patches], 0 is keep, 1 is remove
        """
        # reshape imgs to [N, num_patches, patch_size * C]
        B, C, L = imgs.shape
        p = self.patch_embed.patch_size
        num_patches = L // p
        
        target = imgs.reshape(B, C, num_patches, p)
        target = target.permute(0, 2, 3, 1) # [B, num_patches, patch_size, C]
        target = target.reshape(B, num_patches, p * C)

        loss = torch.abs(pred - target)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_full(self, imgs, pred):
        """整段重建与 GT 的 L1 Loss（MAE）。"""
        B, C, L = imgs.shape
        p = self.patch_embed.patch_size
        num_patches = L // p

        target = imgs.reshape(B, C, num_patches, p)
        target = target.permute(0, 2, 3, 1)
        target = target.reshape(B, num_patches, p * C)

        return torch.nn.functional.l1_loss(pred, target)

    def forward_loss_mask(self, imgs, pred, missing_mask):
        """根据 missing_mask 计算损失：直接在缺失部分计算 MAE (L1 Loss)。"""
        B, C, L = imgs.shape
        p = self.patch_embed.patch_size
        num_patches = L // p

        # 还原 pred 到原始形状 [B, C, L]
        pred_full = pred.reshape(B, num_patches, p, C).permute(0, 3, 1, 2).reshape(B, C, L)
        
        # 将 missing_mask 转为 float，作为计算损失的权重掩码
        m = missing_mask.to(pred_full.dtype)
        
        # 计算所有的 L1 Loss
        loss_l1 = torch.nn.functional.l1_loss(pred_full, imgs, reduction="none")
        
        # 仅在 missing 处求均值
        denom = m.sum().clamp_min(1.0)
        loss = (loss_l1 * m).sum() / denom
        
        return loss

    def forward_inpaint(self, x_obs, target_gt=None, missing_mask=None):
        """
        x_obs: 观测输入 [B,C*2,L]（前半部分含 0 洞，后半部分为 mask），不在此函数内再做 patch 随机 mask。
        target_gt: 重建目标 [B,C,L]，仅在训练时需要。如果为 None，则直接返回组合后的波形。
        missing_mask: [B,C,L] 布尔值，标识哪些是缺失部分（65535）。
        """
        latent, mask, ids_restore = self.forward_encoder(x_obs, mask_ratio=0.0)
        pred = self.forward_decoder(latent, ids_restore)
        
        # 在推理/测试阶段，如果没有给出 target_gt，只返回预测结果
        if target_gt is None:
            B, num_patches, _ = pred.shape
            p = self.patch_embed.patch_size
            C = self.patch_embed.proj.in_channels // 2  # 还原真实通道数(即12)
            
            # 1. 还原 pred 到原始形状 [B, C, L]
            pred_full = pred.reshape(B, num_patches, p, C).permute(0, 3, 1, 2).reshape(B, C, -1)
            
            if missing_mask is not None:
                original_signal = x_obs[:, :C, :]
                final_signal = torch.where(missing_mask, pred_full, original_signal)
                return None, final_signal, mask
            return None, pred_full, mask
        
        if missing_mask is not None:
            loss = self.forward_loss_mask(target_gt, pred, missing_mask)
        else:
            loss = self.forward_loss_full(target_gt, pred)
            
        return loss, pred, mask

    def forward(self, imgs, mask_ratio=0.20):
        """
        在常规的前向传播（如 pre-train）中，imgs 依然是 [B, C, L]
        此时需要内部生成随机 mask，并将 mask 作为一个通道拼接进去
        """
        # 1. 扩充通道：将掩码信息拼接到输入中
        # imgs: [B, C, L]
        B, C, L = imgs.shape
        
        # 为了复用之前的 encoder 逻辑，我们需要调整这里的代码。
        # 由于输入现在需要是 2*C 通道，我们先执行随机 masking：
        
        # 我们需要一种方式将 [B, C, L] 和生成的 mask 结合起来
        # 注意原来的 forward_encoder 内部会调用 random_masking
        
        # 这是一个兼容性的补丁：由于我们把 patch_embed 改成了接受 2*C 通道
        # 所以这里的输入也必须是 2*C 通道
        
        # 简单起见，我们在这里构造一个全 0 的 mask 通道拼接上去
        # 真正的 random_masking 还是在 forward_encoder 里发生（作用于 patch 级别）
        dummy_mask = torch.zeros_like(imgs)
        imgs_extended = torch.cat([imgs, dummy_mask], dim=1)
        
        latent, mask, ids_restore = self.forward_encoder(imgs_extended, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_vit_base_patch50(**kwargs):
    model = MaskedAutoencoderViT1D(
        patch_size=50, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=nn.LayerNorm, **kwargs)
    return model