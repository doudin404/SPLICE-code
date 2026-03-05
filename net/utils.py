import torch
from torch import nn
import numpy as np
from skimage import measure
from typing import Union
import trimesh
import pyvista as pv


class SineLayer(nn.Module):
    """
    From the siren repository
    https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
    """

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.output_channels = out_features
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class CatSineLayer(nn.Module):
    """
    New network layer that includes an additional positional encoding operation.
    """

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        # Adjust the output features to leave space for the original coordinates
        self.pos_encoder = SineLayer(
            in_features, out_features - in_features, bias, is_first, omega_0
        )

    def forward(self, coords):
        # Apply the positional encoder
        pos = self.pos_encoder(coords)
        # Concatenate the original coordinates and positional encoding along the last dimension
        pos = torch.cat((coords, pos), dim=-1)
        return pos


class CatMlpLayer(nn.Module):
    """
    A network layer that includes an additional MLP operation.
    """

    def __init__(self, in_features, out_features, bias=True,is_first=True):
        super().__init__()
        hidden_features = out_features * 2
        # MLP consists of two linear layers with a ReLU activation in between
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features - in_features, bias=bias),
        )

    def forward(self, coords):
        # Apply the MLP
        mlp_out = self.mlp(coords)
        # Concatenate the original coordinates and MLP output along the last dimension
        out = torch.cat((coords, mlp_out), dim=-1)
        return out


class CatMlpSineLayer(nn.Module):
    """
    A network layer that includes an additional MLP operation and a SineLayer operation.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
    ):
        super().__init__()
        hidden_features = out_features
        # MLP consists of two linear layers with a ReLU activation in between
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_features, (out_features - in_features) // 2, bias=bias),
        )
        # SineLayer to apply sinusoidal transformation
        self.sine_layer = SineLayer(
            in_features,
            (out_features - in_features) - (out_features - in_features) // 2,
            bias,
            is_first,
            omega_0,
        )

    def forward(self, coords):
        # Apply the MLP
        mlp_out = self.mlp(coords)
        # Apply the SineLayer
        sine_out = self.sine_layer(coords)
        # Concatenate the original coordinates, SineLayer output, and MLP output along the last dimension
        out = torch.cat((coords, sine_out, mlp_out), dim=-1)
        return out



class DataPadder:
    def __init__(self, max_parts_length, padding_value=0):
        """
        初始化DataPadder类。
        :param max_parts_length: 批处理中的最大部分数量。
        :param padding_value: 用于填充的值。
        """
        self.max_parts_length = max_parts_length
        self.padding_value = padding_value

    def pad_and_stack(self, encoded_batches):
        """
        对输入的批次进行填充和堆叠，并返回填充的数据和填充位置的布尔数组。
        :param encoded_batches: 一个列表，包含需要被填充和堆叠的批次。
        :return: (填充和堆叠后的张量, 填充位置的布尔张量)
        """
        padded_batches = []
        padding_masks = []  # 用于存储填充位置的布尔数组
        for batch in encoded_batches:
            n_parts = len(batch)
            if n_parts < self.max_parts_length:
                padding_size = self.max_parts_length - n_parts
                # 创建与batch相同维度的填充张量，只是在第一个维度进行填充
                padding_shape = (padding_size, *batch.shape[1:])
                padding_tensor = torch.full(
                    padding_shape,
                    self.padding_value,
                    dtype=batch.dtype,
                    device=batch.device,
                )
                batch_padded = torch.cat((batch, padding_tensor), dim=0)
                # 创建一个布尔张量，标记填充的部分为True，未填充的部分为False
                mask = torch.cat(
                    (
                        torch.zeros(n_parts, dtype=torch.bool, device=batch.device),
                        torch.ones(padding_size, dtype=torch.bool, device=batch.device),
                    ),
                    dim=0,
                )
            else:
                batch_padded = batch
                mask = torch.zeros(
                    self.max_parts_length, dtype=torch.bool, device=batch.device
                )
            padded_batches.append(batch_padded)
            padding_masks.append(mask)

        stacked_batches = torch.stack(padded_batches, dim=0)
        stacked_masks = torch.stack(padding_masks, dim=0)
        return stacked_batches, stacked_masks

    def prepare_data(self, data_batches):
        """
        准备数据：将原始数据批次转换为张量，然后调用pad_and_stack进行处理。
        :param data_batches: 一个列表，其中每个元素都是一批数据的numpy数组。
        :return: 处理后的张量数据和填充位置的布尔数组。
        """
        encoded_batches = [torch.tensor(batch) for batch in data_batches]
        return self.pad_and_stack(encoded_batches)


class OccMeshlizer:
    def __init__(self, occ_func):
        """
        初始化 MeshVisualizer 类。
        :param occ_func: 获取占据概率的函数。
        """
        self.occ_func = occ_func

    def generate_grid_points(self, size=256, range=(-1, 1)):
        # 生成方阵点云
        grid_x, grid_y, grid_z = torch.meshgrid(torch.linspace(*range, size), torch.linspace(*range, size), torch.linspace(*range, size))
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid.reshape(1, -1, 3)

    def get_mesh_from_prediction(self, points, size=256, batch_size=100000):
        # 分批处理点云数据
        num_points = points.shape[1]
        y_hat_full = []

        for start_idx in range(0, num_points, batch_size):
            end_idx = min(start_idx + batch_size, num_points)
            points_batch = points[:, start_idx:end_idx, :]
            y_hat_batch = self.occ_func(x=points_batch)
            y_hat_full.append(y_hat_batch.cpu().detach())

        # 拼接所有批次的预测结果并重构形状为方阵
        y_hat_full = torch.cat(y_hat_full, dim=1)
        y_hat_np = y_hat_full.numpy().reshape(size, size, size)

        # 检查占据概率的范围
        min_value, max_value = np.min(y_hat_np), np.max(y_hat_np)
        level = 0.5  # 默认的等值面水平
        if not (min_value <= level <= max_value):
            # 如果level不在范围内，返回一个简单的立方体
            vertices = np.array([
                [-1, -1, -1],
                [1, -1, -1],
                [1,  1, -1],
                [-1,  1, -1],
                [-1, -1,  1],
                [1, -1,  1],
                [1,  1,  1],
                [-1,  1,  1]
            ])
            faces = np.array([
                [0, 3, 1], [1, 3, 2],
                [0, 4, 7], [0, 7, 3],
                [4, 5, 6], [4, 6, 7],
                [5, 1, 2], [5, 2, 6],
                [2, 3, 7], [2, 7, 6],
                [0, 1, 5], [0, 5, 4]
            ])
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh

        # 使用 marching_cubes 方法将预测结果 (占据概率) 转换为网格
        verts, faces, normals, values = measure.marching_cubes(y_hat_np, level=level)

        # 将点从 [0, size-1] 归一化到 [-1, 1]
        verts = 2 * (verts / (size - 1)) - 1

        # 创建 trimesh 网格对象
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # 返回 trimesh 网格对象
        return mesh
    
    def visualize_with_pyvista(self, mesh):
        # 使用 pyvista 展示网格
        plotter = pv.Plotter()
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        pv_mesh = pv.PolyData(verts, faces)
        plotter.add_mesh(pv_mesh, color=True)
        plotter.show()


class KLRegularizationLayer(nn.Module):
    def __init__(self):
        super(KLRegularizationLayer, self).__init__()

    def forward(self, x):
        # 假设输入的最后一个维度是两倍的潜在空间维度
        # 前一半为均值，后一半为方差的对数值
        mean, log_var = torch.chunk(x, 2, dim=-1)
        
        # 计算标准差
        std = torch.exp(0.5 * log_var)
        
        # 通过重参数化技巧进行采样
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, log_var

def np2th(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray.detach().cpu()
    elif isinstance(ndarray, np.ndarray):
        return torch.tensor(ndarray).float()
    else:
        raise ValueError("Input should be either torch.Tensor or np.ndarray")

def clip_eigenvalues(gaus: Union[torch.Tensor, np.ndarray], eps=1e-4):
    """
    Input:
        gaus: [B,G,16] or [G,16]
    Output:
        gaus_clipped: [B,G,16] or [G,16] torch.Tensor
    """
    gaus = np2th(gaus)
    clipped_gaus = gaus.clone()
    clipped_gaus[..., 13:16] = torch.clamp_min(clipped_gaus[..., 13:16], eps)
    return clipped_gaus


def project_eigenvectors(gaus: Union[torch.Tensor, np.ndarray]):
    """
    Input:
        gaus: [B,G,16] or [G,16]
    Output:
        gaus_projected: [B,G,16] or [1,G,16]
    """
    gaus = np2th(gaus).clone()
    if gaus.ndim == 2:
        gaus = gaus.unsqueeze(0)

    B, G = gaus.shape[:2]
    eigvec = gaus[:, :, 3:12]
    eigvec_projected = get_orthonormal_bases_svd(eigvec)
    gaus[:, :, 3:12] = eigvec_projected
    return gaus


def get_orthonormal_bases_svd(vs: torch.Tensor):
    """
    Implements the solution for the Orthogonal Procrustes problem,
    which projects a matrix to the closest rotation matrix / reflection matrix using SVD.
    Args:
        vs: Tensor of shape (B, M, 9)
    Returns:
        p: Tensor of shape (B, M, 9).
    """
    # Compute SVDs of matrices in batch
    b, m, _ = vs.shape
    vs_ = vs.reshape(b * m, 3, 3)
    U, _, Vh = torch.linalg.svd(vs_)
    # Determine the diagonal matrix to make determinants 1
    sigma = torch.eye(3)[None, ...].repeat(b * m, 1, 1).to(vs_.device)
    det = torch.linalg.det(torch.bmm(U, Vh))  # Compute determinants of UVT
    ####
    # Do not set the sign of determinants to 1.
    # Inputs contain reflection matrices.
    # sigma[:, 2, 2] = det
    ####
    # Construct orthogonal matrices
    p = torch.bmm(torch.bmm(U, sigma), Vh)
    return p.reshape(b, m, 9)