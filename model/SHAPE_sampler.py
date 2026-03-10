from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

def _write_obj(path: Path, verts: torch.Tensor, faces: torch.Tensor) -> None:
    v = verts.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    with open(path, "w") as fp:
        for x, y, z in v:
            fp.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in f:
            fp.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")


class ShapeSampler:
    """
    统一生成：严格全步去噪 & 去噪优化步数解释（无条件版本）

    规则：
      1) **无 init_data** -> 从纯噪声采样，并且严格跑满 `timesteps`（例如 1000 步）。忽略 steps/noise_t。
      2) **有 init_data** -> 进行去噪优化；`steps` 表示在 1000 步中的**尾部 k 步**（从 T-1 向 0 走 k 步）。
         可与 fixed_mask 叠加以实现“部分去噪优化”。

    依赖扩散模型具备：
      - forward(x, t) 或 model(x, t)（等价于噪声预测器）
      - scheduler.step(model_output, t, sample) -> obj(prev_sample=...)
      - timesteps: int (通常 1000)
      - feature_dim: int  (应等于 1+15+feat_dim)
      - (可选) scheduler.add_noise(x0, noise, t)

    依赖解码模型具备：
      - reconstruct_from_gaussians(gaussians[N,15], codes[N,C], need_sqrt=True) -> (verts, faces)
    """

    def __init__(self, diffusion_model, decoder_model, save_dir: str = "sample_export"):
        self.dm = diffusion_model
        self.dec = decoder_model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = (
            getattr(decoder_model, "device", None)
            or getattr(diffusion_model, "device", None)
            or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.dm.timesteps,
            beta_schedule='linear',
            clip_sample=False,
        )

    # ---------------- 基础工具 ----------------
    @staticmethod
    def _split(sample: torch.Tensor):
        L, D = sample.shape
        if D < 16:
            raise ValueError(f"D 至少 16 (1+15)，当前 {D}")
        flags = sample[:, 0]
        params15 = sample[:, 1:16]
        feats = sample[:, 16:]
        valid = (flags > 0) & (feats.abs().sum(dim=-1) > 0)
        return params15[valid], feats[valid]

    def _full_schedule(self) -> list[int]:
        T = int(self.dm.timesteps)
        return list(range(T - 1, -1, -1))

    # def _tail_k_schedule(self, k: int) -> list[int]:
    #     T = int(self.dm.timesteps)
    #     k = int(max(1, min(k, min(T, 1000))))
    #     start = T - 1
    #     end = T - 1 - k
    #     return list(range(start, end, -1)) + ([0] if end > 0 else [])
    def _to_zero_schedule(self, t_start: int) -> list[int]:
        """从给定 t_start 一路去噪到 0：t_start, t_start-1, ..., 0"""
        T = int(self.dm.timesteps)
        t_start = int(max(0, min(t_start, T - 1)))
        return list(range(t_start, -1, -1))

    # def _predict_noise(self, x: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
    #     """统一调用噪声预测：优先调用 diffusion.forward(x,t)，否则退回 model(x,t) 或 diffusion_module(x,t)。"""
    #     dm = self.dm
    #     # 优先 LightningModule 的 forward
    #     try:
    #         return dm(x, t_batch)
    #     except Exception:
    #         pass
    #     if hasattr(dm, "model") and callable(dm.model):
    #         return dm.model(x, t_batch)
    #     if hasattr(dm, "diffusion_module") and callable(dm.diffusion_module):
    #         return dm.diffusion_module(x, t_batch)
    #     raise AttributeError("无法找到噪声预测函数：期望 forward(x,t) 或 model(x,t) 或 diffusion_module(x,t)")

    # ---------------- 统一入口 ----------------
    @torch.no_grad()
    def generate(
        self,
        *,
        batch_size: Optional[int] = None,
        max_len: Optional[int] = None,
        init_data: Optional[torch.Tensor] = None,   # [B, L, D]
        fixed_mask: Optional[torch.Tensor] = None,  # [B, L]
        noise_t: Optional[int] = None,              # 可选：先把 init_data 加噪到 t=noise_t 再去噪
        name_prefix: str = "gen",
        clamp_mode: str = "auto",                  # "auto" | "x0" | "xt"
    ) -> List[Path]:
        

        # x = torch.randn(batch_size, max_len, self.dm.feature_dim, device=self.device)

        # for t in reversed(range(1000)):
        #     t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        #     noise_pred = self.dm(x, t_batch)
        #     step = self.scheduler.step(noise_pred, t, x)
        #     x = step.prev_sample


        # final = x.detach().cpu().numpy()
        # return (final, intermediates) if return_intermediates else final
        dm = self.dm
        has_add = hasattr(dm.scheduler, "add_noise")

        # 1) 准备 batch 尺寸与初值
        if init_data is None:
            if batch_size is None or max_len is None:
                raise ValueError("未提供 init_data 时，必须给出 batch_size 与 max_len。")
            B = int(batch_size)
            L = int(max_len)
            D = int(dm.feature_dim)
            x = torch.randn(B, L, D, device=self.device)
            # 严格全步：忽略 steps/noise_t
            t_list = self._full_schedule()
            init_for_clamp = None
        else:
            init_data = init_data.to(self.device)
            if init_data.ndim != 3:
                raise ValueError(f"init_data 形状应为 [B,L,D]，当前 {tuple(init_data.shape)}")
            B, L, D = init_data.shape
            x = init_data.clone()
            # 去噪优化：noise_t 表示 k，取尾部 k 步；未给 noise_t 则默认为全步
            t_list = self._to_zero_schedule(noise_t if noise_t is not None else int(dm.timesteps))
            init_for_clamp = init_data

        # 2) 掩码与 clamp 策略
        if fixed_mask is not None:
            fixed_mask = fixed_mask.to(self.device).bool()
            if tuple(fixed_mask.shape) != (B, L):
                raise ValueError(f"fixed_mask 形状应为 {(B,L)}，当前 {tuple(fixed_mask.shape)}")
        else:
            fixed_mask = torch.zeros(B, L, dtype=torch.bool, device=self.device)

        if clamp_mode not in ("auto", "x0", "xt"):
            raise ValueError("clamp_mode 仅支持 'auto'|'x0'|'xt'")
        use_xt = (clamp_mode == "xt") or (clamp_mode == "auto" and has_add)
        if use_xt:
            base_noise = torch.randn_like(x)

        # 3) 可选：若提供 init_data 且需要先加噪到 noise_t，再去噪
        if init_for_clamp is not None and noise_t is not None and has_add:
            t0 = int(max(0, min(int(noise_t), int(dm.timesteps) - 1)))
            noise = torch.randn_like(x)
            t_batch0 = torch.full((B,), t0, device=self.device, dtype=torch.long)
            x = self.scheduler.add_noise(x, noise, t_batch0)

        # feats = x[..., 16:]  # [B, L, feat_dim]
        # pos = x[..., :16]  # [B, L, 16]，包含 flags + params
        #x

        #4) 反向迭代
        for t in t_list:
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.dm(x, t_batch)
            step = self.scheduler.step(noise_pred, t, x)
            x = step.prev_sample

            # x_fixed_t = self.scheduler.add_noise(init_for_clamp,base_noise if t!=0 else torch.zeros_like(base_noise), t_batch)
            # x[...,:16]=x_fixed_t[...,:16]
            # x[...,16:]=x_fixed_t[...,16:]
            #x[...,:]=x_fixed_t[...,:]

            # if init_for_clamp is not None and fixed_mask.any():
            #     if use_xt and has_add:
            #         x_fixed_t = self.scheduler.add_noise(init_for_clamp[...,:16], base_noise, t_batch)
            #         x = torch.where(fixed_mask.unsqueeze(-1), x_fixed_t, x)
            #     else:
            #         x = torch.where(fixed_mask.unsqueeze(-1), init_for_clamp[...,:16], x)

        #x=torch.cat([x, feats], dim=-1)  # 恢复 feats
        #x = init_data.clone()
        x[...,16:]/=0.1
        x[..., 4:13]/=1
        x[..., 1:4]/= 5
        x[..., 13:16]/= 5

        # 5) 解码并保存
        paths: List[Path] = []
        for i in range(B):
            g, c = self._split(x[i])
            if g.numel() == 0:
                continue
            v, f = self.dec.reconstruct_from_gaussians(g, c, need_sqrt=False)
            dst = self.save_dir / f"{name_prefix}_{i:04d}.obj"
            _write_obj(dst, v, f)
            paths.append(dst)
        return paths
    # def test(self,t):
    #     """使用add_noise加噪,使用_predict_noise预测噪声,计算均方误差"""
    #     x = torch.randn(1, 1000, 1+15+256, device=self.device)
    #     noise = torch.randn_like(x)
    #     t_batch = torch.full((1,), t, device=self.device, dtype=torch.long)
    #     noisy = self.self.scheduler.add_noise(x, noise, t_batch)
    #     noise_pred = self._predict_noise(noisy, t_batch)
    #     mse = F.mse_loss(noise_pred, noise, reduction='mean')
    #     print(f"1:t={t}, MSE={mse.item()}")
    # def test2(self,t):
    #     x = torch.randn(1, 1000, 1+15+256, device=self.device)
    #     noise = torch.randn_like(x)
    #     t_batch = torch.full((1,), t, device=self.device, dtype=torch.long)
    #     noisy = self.self.scheduler.add_noise(x, noise, t_batch)
    #     noise_pred = noisy[:]
    #     mse = F.mse_loss(noise_pred, noise, reduction='mean')
    #     print(f"2:t={t}, MSE={mse.item()}")



