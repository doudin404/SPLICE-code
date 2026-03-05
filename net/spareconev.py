import spconv.pytorch as spconv
from torch import nn
import torch


class SparseVoxelConv3D(nn.Module):
    def __init__(
        self, input_channels, num_layers, features, spatial_shape, output_channels
    ):
        super(SparseVoxelConv3D, self).__init__()

        self.spatial_shape = spatial_shape  # 空间形状，例如 [64, 64, 64]
        self.num_layers = num_layers  # 卷积层数

        layers = []
        current_channels = input_channels

        # 构建多层稀疏卷积层
        for i in range(num_layers):
            # 输出特征大小
            out_channels = features[i] if i < len(features) else output_channels
            # 添加稀疏卷积层
            layers.append(
                spconv.SparseConv3d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            # 添加批量归一化层
            layers.append(nn.BatchNorm1d(out_channels))
            # 添加激活函数层
            layers.append(nn.ReLU())
            current_channels = out_channels

        # 最后一层卷积，确保输出通道数量匹配
        if current_channels != output_channels:
            layers.append(
                spconv.SparseConv3d(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm1d(output_channels))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(output_channels, output_channels))

        # 转换为稠密张量
        layers.append(spconv.ToDense())

        self.network = spconv.SparseSequential(*layers)
        self.layers=layers

    def forward(self, x):
        # x 应该是一个 SparseConvTensor
        y = self.network(x)
        return y.squeeze(-1).squeeze(-1).squeeze(-1)

class PartsToVoxelEncoder(nn.Module):
    def __init__(self, input_channels, num_layers, features, spatial_shape, output_channels):
        super(PartsToVoxelEncoder, self).__init__()
        self.spatial_shape=spatial_shape
        self.voxel_encoder = SparseVoxelConv3D(input_channels, num_layers, features, spatial_shape, output_channels)

    def forward(self, parts_voxels, parts_labels, remove_zero_labels=False):
        batch_sizes = [len(batch) for batch in parts_voxels]  # 记录每个 batch 的长度
        all_coords = []  # 存放所有扁平化后的坐标
        all_labels = []  # 存放所有扁平化后的标签
        all_batch_indices = []  # 存放每个点的批次索引
        current_batch_index = 0  # 当前的批次索引

        for batch_idx, (batch_voxels, labels_batch) in enumerate(zip(parts_voxels, parts_labels)):
            for coords, labels in zip(batch_voxels, labels_batch):
                # 如果 remove_zero_labels 为 True，则过滤掉 labels 为 0 的体素
                if remove_zero_labels:
                    mask = labels != 0  # 创建掩码，只保留标签不为 0 的体素
                    coords = coords[mask]  # 只保留标签不为 0 的坐标
                    labels = labels[mask]  # 只保留标签不为 0 的标签
                all_coords.append(coords)  # 添加坐标
                # 为坐标添加batch维度
                part_batch_indices = torch.full(
                    (coords.size(0), 1),
                    current_batch_index,
                    dtype=torch.int32,
                    device=coords.device,
                )
                all_batch_indices.append(part_batch_indices)
                
                # 处理标签
                all_labels.append(labels.float().unsqueeze(1))  # 将标签转换为浮点数并添加一个维度

                current_batch_index += 1  # 更新批次索引

        coords = torch.cat(all_coords, 0)
        batch_indices = torch.cat(all_batch_indices, 0)
        labels = torch.cat(all_labels, 0)

        # 将批次索引合并到坐标中
        new_coords = torch.cat([batch_indices, coords], 1)

        # 设置所有有效体素的features为1
        features = torch.ones((new_coords.size(0), 1), dtype=torch.float32, device=new_coords.device)
        
        # 将标签信息与 features 合并
        features = torch.cat([features, labels], dim=1)

        # 使用spconv将特征和坐标转换为SparseConvTensor
        x = spconv.SparseConvTensor(features, new_coords, self.spatial_shape, current_batch_index)

        # 编码
        encoded_features = self.voxel_encoder(x)

        # 将编码后的特征分割回原始的batch和part格式
        # 按照每个batch的长度分割
        encoded_parts = torch.split(encoded_features, batch_sizes, dim=0)

        return encoded_parts  # 返回编码后的数据，格式与输入相同
