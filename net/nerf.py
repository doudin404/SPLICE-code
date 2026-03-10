import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    实现 NeRF 中的位置编码 γ(p) 并可选地通过线性层投影到任意输出维度。

    参数:
      num_freqs: 频率数量 L
      in_dim: 输入维度（如坐标维度 3）
      out_dim: 投影后输出维度，如果为 None 则不投影
      include_input: 是否保留原始输入
    """
    def __init__(
        self,
        num_freqs: int,
        in_dim: int = 3,
        out_dim: Optional[int] = None,
        include_input: bool = True,
    ):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.in_dim = in_dim
        self.include_input = include_input
        # 生成频率带 [2^0*pi, 2^1*pi, ..., 2^{L-1}*pi]
        freqs = 2.0 ** torch.arange(num_freqs).float() * math.pi
        self.register_buffer('freq_bands', freqs)
        # 计算编码维度
        self.embed_dim = (in_dim if include_input else 0) + 2 * num_freqs * in_dim
        # 可选投影层
        if out_dim is not None:
            self.proj = nn.Linear(self.embed_dim, out_dim)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        emb = torch.cat(out, dim=-1)
        if self.proj is not None:
            emb = self.proj(emb)
        return emb


class NeRF(nn.Module):
    """
    基于 NeRF 论文的密度预测 MLP 结构：
      - 8 层隐藏层，宽度 256，skip connection 加入输入
      - 仅输出 density (σ)
    """
    def __init__(
        self,
        depth: int = 8,
        width: int = 256,
        input_ch: int = 60,
        skips: list = [4],
    ):
        super(NeRF, self).__init__()
        self.depth = depth
        self.width = width
        self.skips = skips

        # 位置分支的线性层
        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(input_ch, width))
        for i in range(1, depth):
            if i in skips:
                self.pts_linears.append(nn.Linear(width + input_ch, width))
            else:
                self.pts_linears.append(nn.Linear(width, width))

        # density（σ）输出层
        self.sigma_layer = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., input_ch) 位置编码后的输入
        返回: (..., 1) 仅 density (σ)
        """
        h = x
        for i, layer in enumerate(self.pts_linears):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)

        sigma = self.sigma_layer(h)
        return sigma


# 示例用法
if __name__ == "__main__":
    # L=10 对坐标编码
    pos_encoder = PositionalEncoding(num_freqs=10)
    model = NeRF()

    # 假设有一批 xyz 坐标
    xyz = torch.rand(1024, 3)
    pe_xyz = pos_encoder(xyz)       # (1024, 60)
    sigma = model(pe_xyz)           # (1024, 1)
    print(sigma.shape)  # 应为 torch.Size([1024, 1])