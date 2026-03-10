import torch
import torch.nn as nn
import spconv.pytorch as sp

class SparseVoxelDownsampleModule(nn.Module):
    def __init__(self, in_channels=1,mid_channels=256, out_channels=256, num_layers=8):
        super(SparseVoxelDownsampleModule, self).__init__()
        # Compute intermediate channels
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        channels = [in_channels]
        for i in range(num_layers):
            if channels[-1] < 8:
                next_c = min(8, mid_channels)
            elif channels[-1] < mid_channels:
                next_c = min(channels[-1] * 2, mid_channels)
            else:
                next_c = mid_channels
            channels.append(next_c)
        channels = [int(c) for c in channels]
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.layers.append(
                sp.SparseSequential(
                    sp.SparseConv3d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm1d(channels[i+1]),
                    nn.ReLU()
                )
            )
        # Add a final linear layer (applied to features)
        self.final_linear = nn.Linear(channels[-1], out_channels)

    def forward(self, x):
        # 统一居中：按每个 batch 计算质心并移动到网格中心
        coords = x.indices  # (N, 4)
        batch_idxs = coords[:, 0]
        xyz = coords[:, 1:].float()
        spatial_shape = x.spatial_shape
        center = torch.tensor([s // 2 for s in spatial_shape], device=coords.device, dtype=torch.float)

        unique_batches = batch_idxs.unique()
        for b in unique_batches:
            mask = (batch_idxs == b)
            if mask.sum() > 0:
                mean = xyz[mask].mean(dim=0)
                shift = center - mean
                xyz[mask] += shift

        # 更新坐标并重建 SparseConvTensor
        new_coords = torch.cat([batch_idxs.unsqueeze(1), xyz.round().to(coords.dtype)], dim=1)
        x = sp.SparseConvTensor(
            features=x.features,
            indices=new_coords,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )

        for layer in self.layers:   
            x = layer(x)
        # Apply final linear layer to features using replace_feature
        x = x.replace_feature(self.final_linear(x.features))
        return x


class SparseVoxelDownsampleModule2(nn.Module):
    def __init__(self, in_channels=1, mid_channels=256, out_channels=256, num_layers=8):
        super(SparseVoxelDownsampleModule2, self).__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # 构建卷积层列表 (stride=1) 和共享的最大池化层
        channels = [in_channels]
        for i in range(num_layers):
            prev = channels[-1]
            if prev < 8:
                nxt = min(8, mid_channels)
            elif prev < mid_channels:
                nxt = min(prev * 2, mid_channels)
            else:
                nxt = mid_channels
            channels.append(nxt)

        self.conv_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.conv_layers.append(
                sp.SparseSequential(
                    sp.SparseConv3d(
                        channels[i], channels[i+1], kernel_size=3, stride=1, padding=1, bias=False
                    ),
                    nn.BatchNorm1d(channels[i+1]),
                    nn.ReLU()
                )
            )

        # 最大池化，用于下采样
        self.pool = sp.SparseMaxPool3d(kernel_size=2, stride=2)

        # 最后的线性层
        self.final_linear = nn.Linear(channels[-1], out_channels)

    def forward(self, x):
        # 统一居中：按每个 batch 计算质心并移动到网格中心
        coords = x.indices  # (N, 4)
        batch_idxs = coords[:, 0]
        xyz = coords[:, 1:].float()
        spatial_shape = x.spatial_shape
        center = torch.tensor([s // 2 for s in spatial_shape], device=coords.device, dtype=torch.float)

        unique_batches = batch_idxs.unique()
        for b in unique_batches:
            mask = (batch_idxs == b)
            if mask.sum() > 0:
                mean = xyz[mask].mean(dim=0)
                shift = center - mean
                xyz[mask] += shift

        # 更新坐标并重建 SparseConvTensor
        new_coords = torch.cat([batch_idxs.unsqueeze(1), xyz.round().to(coords.dtype)], dim=1)
        x = sp.SparseConvTensor(
            features=x.features,
            indices=new_coords,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )

        # 依次进行卷积和最大池化
        for conv in self.conv_layers:
            x = conv(x)
            x = self.pool(x)

        # 应用线性层到 features
        x = x.replace_feature(self.final_linear(x.features))
        return x
