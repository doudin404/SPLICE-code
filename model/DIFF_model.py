
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from net.diffusion import TransformerDiffusion
from diffusers import DDPMScheduler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

class SPLICE_Diffusion(pl.LightningModule):
    def __init__(self,
                 feature_dim: int = 1+15+256,
                 emb_size: int = 512,
                 n_layers: int = 8,
                 n_heads: int = 16,
                 dropout: float = 0.1,
                 timesteps: int = 1000,
                 lr: float = 1e-5,
                 warmup_epochs: int = 10,
                 num_classes: int = 1,
                 feat_only=False,
                 ):  # 新增num_classes参数
        super().__init__()
        self.feature_dim = feature_dim
        self.feat_only = feat_only
        # 条件嵌入
        self.cond_embd = nn.Embedding(num_classes, emb_size)
        # 定义去噪模型
        self.model = TransformerDiffusion(
            feature_dim=feature_dim,
            emb_size=emb_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        # 定义调度器
        self.scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule="linear"
        )
        self.timesteps = timesteps
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes

    def forward(self, x, t):
        cond= torch.zeros((x.shape[0],), device=self.device,dtype=torch.int)  # (B,)
        cond_embd = self.cond_embd(cond)
        return self.model(x, t,cond_embd)
    
    def test(self,t,x):
        """使用add_noise加噪,使用_predict_noise预测噪声,计算均方误差"""
        noise = torch.randn_like(x)
        t_batch = torch.full((1,), t, device=self.device, dtype=torch.long)
        noisy = self.scheduler.add_noise(x, noise, t_batch)
        noise_pred = self(noisy, t_batch)
        mse = F.mse_loss(noise_pred, noise, reduction='mean')
        print(f"1:t={t}, MSE={mse.item()}")
        
    def compute_loss(self, batch, stage):
        seqs = batch[:]  # (B, S, feature_dim)
        #cond = batch['cond']  # (B,)
        
        #seqs[...,0:1]*=0.5
        #seqs[...,1:]=seqs[...,0:1]
        B, S, _ = seqs.shape
        t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(seqs)
        if self.feat_only:
            noise[..., :16] = 0
        noisy = self.scheduler.add_noise(seqs, noise, t)
        #cond_embd = self.cond_embd(cond)
        noise_pred = self(noisy, t)
        
        mse = F.mse_loss(noise_pred, noise, reduction='none')  # (B, S, D)
        weights = 1#torch.where(seqs[...,0]>0, 1, 0.0005).unsqueeze(-1)  # (B, S, 1)
        loss = (mse * weights).mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # 定义一个 LambdaLR，用来在前 warmup_epochs 内做线性增大
        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epochs:
                # 线性 warm-up
                return float(current_epoch + 1) / float(self.warmup_epochs)
            return 1

        warmup_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "epoch",      # 按 epoch 更新
                "frequency": 1,
            },
        }