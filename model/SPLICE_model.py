import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as sp
import pytorch_lightning as pl
from net.spconv_nets import SparseVoxelDownsampleModule,SparseVoxelDownsampleModule2
from net.func import (
    split_and_pad_features,
    compute_sdf_from_samples,
    sample_sdf_coords,
    quaternion_to_rotation_matrix,
    ellipsoid_vertices,
    gaussian_nll_loss,
    sample_affines,
    query_part_ids,
    recover_ellipsoid,
    multi_ellipsoid_occupancy,
)
from net.nerf import NeRF, PositionalEncoding
from net.occ_net import PointAttention,TransformerEncoder,TransformerMLP
import os
import numpy as np
import trimesh
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
import matplotlib.pyplot as plt
import open3d as o3d


def compute_sdf_batch_torch(
    psr_volumes: torch.Tensor,
    point_clouds: torch.Tensor,
    sample_num: int = 5000,
    surface_bandwidth: float = 0.01,
) -> (torch.Tensor, torch.Tensor):
    """
    批量生成采样坐标及对应的 SDF。
    """
    coords_random = sample_sdf_coords(
        point_clouds, "random", sample_num // 4, surface_bandwidth
    )
    coords_surface2 = sample_sdf_coords(
        point_clouds, "random", sample_num // 4, surface_bandwidth*2
    )
    coords_surface = sample_sdf_coords(
        point_clouds, "near_surface", sample_num // 2, surface_bandwidth
    )
    coords = torch.cat([coords_random,coords_surface2, coords_surface], dim=1)
    sdf = compute_sdf_from_samples(psr_volumes, point_clouds, coords)
    return coords, sdf


class SPLICE_Module(pl.LightningModule):

    def __init__(
        self,
        lr: float = 1e-4,
        in_channels: int = 1,
        feat_dim: int = 256,
        mlp_dim: int = 256,
        attn_layers: int = 4,
        down_layers: int = 8,
        pos_enc_dim: int = 128,
        self_attn: bool = False,
        self_layers: int = 4,
        conv2: bool = True,
        num_heads=8,
        # ---------- 监督采样 ----------
        sample_num: int = 20000,
        surface_bandwidth: float = 0.01,
        # ---------- 损失权重 ----------
        occ_w: float = 10,
        gaussian_w: float = 1,
        feat_l2_w: float =  1e-3,##1e-5,
        # ---------- 其它 ----------
        export_resolution: int = 128,
        export_batch: int = 32768,
        trans_range: tuple = (-0.5, 0.5),  # (-0.5, 0.5),  # 平移范围
        scale_range: tuple = (-0.5, 0.5),  # (-0.5, 0.5)# log-scale 范围
        rot_range: tuple = (
            -math.pi / 6,
            math.pi / 6,
        ),  # (0,0),#(-math.pi/6, math.pi/6)# 旋转角度范围
        # ---------- 导出 ----------
        save_dir: str = "export/pred",  # 导出目录
        export_mode: str = "SPA",  # "SPA" or "SPC"
        export_shape: bool = True,  # 是否导出形状
        export_gaussians: bool = False,  # 是否导出高斯分布
    ):
        super().__init__()
        pred_dim: int = 3 + 3 + 4 + feat_dim * 2  # (dx,dy,dz,log_r) + 128-feat
        self.save_hyperparameters()
        # ---------------- 网络 -------
        if not conv2:
            self.voxel_net = SparseVoxelDownsampleModule(in_channels,feat_dim, pred_dim, down_layers)
        else:
            self.voxel_net = SparseVoxelDownsampleModule2(in_channels,feat_dim, pred_dim, down_layers)
        self.attn = PointAttention(feat_dim, pos_enc_dim, mlp_dim, attn_layers,num_heads)
        if self_layers > 0:
            if self_attn:
                self.self_attn = TransformerEncoder(feat_dim, mlp_dim, attn_layers,num_heads)
            else:
                self.self_attn = TransformerMLP(feat_dim, mlp_dim, self_layers)
        else:
            self.self_attn = None
        self.decoder = NeRF(depth=4, width=mlp_dim, input_ch=pos_enc_dim * 2)
        # 位置编码
        self.point_enc = PositionalEncoding(num_freqs=10, in_dim=3, out_dim=pos_enc_dim)
        self.el_point_enc = PositionalEncoding(
            num_freqs=10, in_dim=3 * 6, out_dim=feat_dim
        )
        # 可学习参数示例
        #self.learnable_param1 = nn.Parameter(torch.randn(3,pred_dim))

        # 初始化用于收集中间结果的列表
        self.scaled_mu_list = []
        self.rotate_list = []
        self.scaled_vec_list = []
        self.feats_list = []
        self.folders_list = []  # 存储每个样本对应的文件夹名
        self.lr = lr
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def encode_voxel_features(self, sparse_tensor, bbox_centers, bbox_radii):
        # ----------- 体素特征提取 -----------
        out = self.voxel_net(sparse_tensor)
        dense = out.dense()
        x = dense[:,:,0,0,0]
        feat_mean = x[
            :, 3 + 3 + 4 : 3 + 3 + 4 + self.hparams.feat_dim
        ]  # (N, feat_dim)
        feat_logvar = x[
            :, 3 + 3 + 4 + self.hparams.feat_dim :
        ]  # (N, feat_dim)
        # 采样
        eps = torch.randn_like(feat_mean)
        feats = feat_mean + eps * torch.exp(0.5 * feat_logvar)

        mu = x[:, :3]  # (N, 3) 平移
        vec = x[:, 3:6].exp()  # (N, 3) 缩放
        quat = x[:, 6:10]  # (N, 4) 四元数旋转

        # ----------- 体素参数变换 -----------
        rotate = quaternion_to_rotation_matrix(quat)  # (N, 3, 3) 旋转矩阵
        scaled_mu = mu * bbox_radii + bbox_centers  # (N, 3) 平移变换
        scaled_vec = vec * bbox_radii  # (N, 3) 缩放变换
        # scaled_mu = mu
        # scaled_vec = vec

        return feats, scaled_mu, rotate, scaled_vec, feat_mean, feat_logvar
    
    def mix_features(self, feats, scaled_mu, rotate, scaled_vec, batch_layer_ranges, affine_matrices=None,feat_mask=None):
        # ----------- 特征分组与填充 -----------
        padded_feats, pad_mask = split_and_pad_features(feats, batch_layer_ranges)

        # ----------- 椭球特征编码 -----------
        el_points = ellipsoid_vertices(scaled_mu, rotate, scaled_vec).view(-1, 3 * 6).detach()
        if affine_matrices is not None:
            # 计算每个椭球应该用哪个仿射矩阵
            # batch_layer_ranges: [0, 3, 5] -> lens: [3, 2]
            lens = [batch_layer_ranges[i+1] - batch_layer_ranges[i] for i in range(len(batch_layer_ranges)-1)]
            # 生成每个椭球所属的 batch 索引
            batch_ids = []
            for idx, l in enumerate(lens):
                batch_ids.extend([idx] * l)
            batch_ids = torch.tensor(batch_ids, device=el_points.device)
            # el_points: (num_ellipsoids*6, 18)
            # batch_ids: (num_ellipsoids,)
            # 需要扩展到每个椭球的6个顶点
            batch_ids = batch_ids.repeat_interleave(6)
            # 重新组织 el_points 形状
            el_points_reshaped = el_points.view(-1, 6, 3)
            # 取对应的 affine
            R = affine_matrices[batch_ids][:, :3, :3]  # (num_ellipsoids*6, 3, 3)
            t = affine_matrices[batch_ids][:, :3, 3]   # (num_ellipsoids*6, 3)
            el_points_aff = torch.bmm(R, el_points_reshaped.view(-1, 3, 1)).squeeze(-1) + t
            el_points = el_points_aff.view(-1, 3 * 6)
        el_x = self.el_point_enc(el_points)
        padded_el_x, _ = split_and_pad_features(el_x, batch_layer_ranges)
        if feat_mask is None:
            padded_feats = padded_feats + padded_el_x  # (B, N, C)
        else:
            padded_feats = padded_feats * feat_mask[:,None,None] + padded_el_x

        if self.self_attn is not None:
            # ----------- 自注意力机制 -----------
            padded_feats = self.self_attn(padded_feats, pad_mask)

        return padded_feats, pad_mask

    # def mix_features(self, feats, scaled_mu, rotate, scaled_vec, batch_layer_ranges,
    #                 affine_matrices=None, radius=1.75):
    #     """
    #     feats:                        原始特征，形状 (总点数, C)
    #     scaled_mu:                    椭球中心，形状 (E, 3)，E 为所有椭球的总数
    #     rotate:                       椭球旋转矩阵，形状 (E, 3, 3)
    #     scaled_vec:                   椭球半轴长度，形状 (E, 3)
    #     batch_layer_ranges:           划分信息，用于 split_and_pad_features
    #     affine_matrices: (可选) 仿射矩阵列表，形状 (B, 4, 4)，None 时表示不做仿射
    #     radius:                       生成顶点时使用的单位球放缩半径（仅用于对比时，实际不再采样顶点）
    #     """
    #     # ----------- 1. 特征分组与填充 -----------
    #     padded_feats, pad_mask = split_and_pad_features(feats, batch_layer_ranges)

    #     # ----------- 2. 将原始参数 flatten 到 (E,⋯) 方便后续处理 -----------
    #     #    假设 scaled_mu、rotate、scaled_vec 的第一个维度正好等于“所有椭球的数量 E”
    #     E = scaled_mu.shape[0]
    #     # scaled_mu_flat: (E, 3)
    #     scaled_mu_flat = scaled_mu.view(E, 3)
    #     # rotate_flat: (E, 3, 3)
    #     rotate_flat = rotate.view(E, 3, 3)
    #     # scaled_vec_flat: (E, 3)
    #     scaled_vec_flat = scaled_vec.view(E, 3)

    #     # ----------- 3. 如果没有 affine_matrices，直接用原参数  -----------
    #     if affine_matrices is None:
    #         # new_mu = scaled_mu_flat
    #         new_mu = scaled_mu_flat                         # (E, 3)
    #         # new_rotate = rotate_flat
    #         new_rotate = rotate_flat                         # (E, 3, 3)
    #         # new_eval = scaled_vec_flat
    #         new_eval = scaled_vec_flat                       # (E, 3)

    #     else:
    #         # ----------- 3.1 先根据 batch_layer_ranges 计算每个椭球的 batch_id  -----------
    #         #    生成一个长度为 E 的张量 batch_ids，使得 batch_ids[i] 表示第 i 个椭球属于哪个 batch
    #         lens = [batch_layer_ranges[i+1] - batch_layer_ranges[i]
    #                 for i in range(len(batch_layer_ranges)-1)]
    #         batch_ids_list = []
    #         for b_idx, length in enumerate(lens):
    #             batch_ids_list.extend([b_idx] * length)
    #         batch_ids = torch.tensor(batch_ids_list, device=scaled_mu.device)  # (E,)

    #         # ----------- 3.2 从 affine_matrices 中取出对应的 A_i (3×3) 和 t_i (3,)  -----------
    #         # affine_matrices[batch_ids] 的形状为 (E, 4, 4)
    #         M = affine_matrices[batch_ids]         # (E, 4, 4)
    #         A = M[:, :3, :3]                       # (E, 3, 3)
    #         t = M[:, :3, 3]                        # (E, 3)

    #         # ----------- 3.3 计算新的中心 new_mu = A @ scaled_mu_flat + t  -----------
    #         # scaled_mu_flat: (E,3) -> (E,3,1) -> A @ (...) -> (E,3) + t -> (E,3)
    #         new_mu = torch.bmm(A, scaled_mu_flat.unsqueeze(-1)).squeeze(-1) + t  # (E, 3)

    #         # ----------- 3.4 计算原 shape 矩阵 S0 = R Λ^2 R^T，每条半轴平方在 Λ^2 里  -----------
    #         #    scaled_vec_flat 里面的每个 a_i 是半轴长度，我们需要它的平方 a_i^2
    #         diag_lambda2 = torch.diag_embed(scaled_vec_flat ** 2)    # (E, 3, 3)：对角矩阵
    #         #    rotate_flat: (E, 3, 3)
    #         #    S0 = R @ diag(a^2) @ R^T
    #         S0 = torch.matmul(rotate_flat, torch.matmul(diag_lambda2, rotate_flat.transpose(-2, -1)))  # (E,3,3)

    #         # ----------- 3.5 计算仿射之后的 shape 矩阵 S = A S0 A^T  -----------
    #         #    A: (E,3,3)，S0: (E,3,3)
    #         AS0 = torch.matmul(A, S0)               # (E, 3, 3)
    #         S = torch.matmul(AS0, A.transpose(-2, -1))  # (E, 3, 3)

    #         # ----------- 3.6 对 S 做特征分解，恢复新的主轴方向和半轴长度  -----------
    #         #    S 是对称正定矩阵，可用 torch.linalg.eigh
    #         #    eigh 返回 eigvals(升序) 和 eigvecs（列向量对应各自的特征值）
    #         eigvals, eigvecs = torch.linalg.eigh(S)  # eigvals: (E, 3)，eigvecs: (E, 3, 3)

    #         #    新的半轴长度 new_eval = sqrt(eigvals)
    #         new_eval = torch.sqrt(eigvals)           # (E, 3)

    #         #    新的旋转矩阵 new_rotate 就是 eigvecs：
    #         #    它的第 i 列是对应第 i 个特征值的单位特征向量，也就是新的主轴方向
    #         new_rotate = eigvecs                     # (E, 3, 3)

    #     # ----------- 4. 把 new_mu (3)、new_rotate (9)、new_eval (3) 拼成 15 维，然后补 3 个 0，得到 18 维 -----------
    #     # new_rotate_flatten: (E, 9)
    #     new_rotate_flatten = new_rotate.reshape(E, 9)
    #     # 15 维 = 中心 3 + 旋转 9 + 半轴长 3
    #     ellipsoid_params_15 = torch.cat([new_mu, new_rotate_flatten, new_eval], dim=-1)  # (E, 15)
    #     # 后面补 3 个零凑成 18 维
    #     zeros = ellipsoid_params_15.new_zeros(E, 3)                                     # (E, 3)
    #     el_params_padded = torch.cat([ellipsoid_params_15, zeros], dim=-1)              # (E, 18)

    #     # ----------- 5. 调用编码器，并做分组/填充  -----------
    #     el_param_feats = self.el_point_enc(el_params_padded.detach())  # (E, C')
    #     padded_el_x, _ = split_and_pad_features(el_param_feats, batch_layer_ranges)
    #     padded_feats = padded_feats*0 + padded_el_x                       # (B, N, C)

    #     # ----------- 6. 如果有自注意力，再接上一层  -----------
    #     if self.self_attn is not None:
    #         padded_feats = self.self_attn(padded_feats, pad_mask)

    #     return padded_feats, pad_mask

    def decode(self, padded_feats, pad_mask, query_points, affine_matrices=None,output_attn=False):
        # ----------- 查询点位置编码 -----------
        if affine_matrices is not None:
            # query_points: (B, N, 3)
            B, N, _ = query_points.shape
            R = affine_matrices[:, :3, :3].unsqueeze(1)  # (B,1,3,3)
            t = affine_matrices[:, :3, 3].unsqueeze(1)   # (B,1,3)
            query_points_aff = torch.matmul(R, query_points.unsqueeze(-1)).squeeze(-1) + t
            query_x = self.point_enc(query_points_aff)
            # # --- Begin inserted code ---

            # # 导出先前保存的椭球端点
            # # self.test_el_points 的形状为 (num_ellipsoids, 18)，每个椭球有 6 个顶点，每个顶点 3 个坐标
            # if hasattr(self, "test_el_points") and self.test_el_points is not None:
            #     ellipsoid_points = self.test_el_points.view(-1, 6, 3).reshape(-1, 3).cpu().numpy()
            # else:
            #     ellipsoid_points = np.empty((0, 3), dtype=np.float32)

            # # 读取 query_points_aff 中第一个 batch_idx 的查询点云，形状 (N, 3)
            # query_points_np = query_points_aff[0].detach().cpu().numpy()

            # # 合并两个点云
            # combined_points = np.concatenate([ellipsoid_points, query_points_np], axis=0)

            # # 为区分不同来源的点，分配颜色（红色表示椭球端点，蓝色表示查询点）
            # colors = np.zeros((combined_points.shape[0], 3), dtype=np.uint8)
            # colors[:ellipsoid_points.shape[0]] = [255, 0, 0]      # 红色
            # colors[ellipsoid_points.shape[0]:] = [0, 0, 255]       # 蓝色

            # # 使用 trimesh 构造点云，并导出为 PLY 文件
            # pc = trimesh.points.PointCloud(vertices=combined_points, colors=colors)
            # ply_filename = f"export_pts_epoch_{self.current_epoch}.ply"
            # pc.export(ply_filename)
            # # --- End inserted code ---
        else:
            query_x = self.point_enc(query_points)

        # ----------- 点注意力机制 -----------
        if output_attn:
            # 如果需要输出注意力特征
            attn_feats, scores = self.attn(padded_feats, pad_mask, query_x, return_scores=True)
        else:
            # 如果不需要输出注意力特征
            attn_feats = self.attn(padded_feats, pad_mask, query_x)

        # ----------- Occupancy/SDF 解码 -----------
        decoder_input = torch.cat([attn_feats, query_x], dim=-1)
        pred_occ = self.decoder(decoder_input)  # (B, N, pred_dim)
        if output_attn:
            return pred_occ, attn_feats, scores
        return pred_occ

    def forward(
        self, sparse_tensor, batch_layer_ranges, query_points, bbox_centers, bbox_radii,affine_matrices=None,feat_mask=None
    ):
        feats, scaled_mu, rotate, scaled_vec, feat_mean, feat_logvar = (
            self.encode_voxel_features(sparse_tensor, bbox_centers, bbox_radii)
        )
        padded_feats, pad_mask = self.mix_features(
            feats, scaled_mu, rotate, scaled_vec, batch_layer_ranges,affine_matrices,feat_mask=feat_mask
        )
        pred_occ, attn_feats, scores = self.decode(padded_feats, pad_mask, query_points,affine_matrices,output_attn=True)
        return pred_occ, scaled_mu, rotate, scaled_vec, feats, feat_mean, feat_logvar,padded_feats,attn_feats,pad_mask, scores

    def training_step(self, batch, batch_idx):
        # 计时：数据获取
        t0 = time.time()
        points = batch["points"]  # (B, N, 3)
        occ_coords = batch["occ_coords"]  # (N, 4)
        occ_feats = batch["occ_feats"]  # (N, 1)
        occ_shape = [256,256,256] # (D, H, W)
        batch_size = points.shape[0]
        #vox_bin = batch["vox_bin"]  # (B, D, H, W)
        bbox_centers = batch["occ_bbox_centers"]  # (B, 3)
        bbox_radii = batch["occ_bbox_radii"]  # (B,)
        batch_layer_ranges = batch["batch_layer_ranges"]
        vox_batch_size = occ_coords[:, 0].max() + 1
        affines = self.sample_affines(batch_size)
        feat_mask = (torch.rand(batch_size) < 0.9).to(occ_feats.device)
        sparse_tensor = sp.SparseConvTensor(
            features=occ_feats,
            indices=occ_coords,
            spatial_shape=occ_shape,
            batch_size=vox_batch_size,
        )


        # 创建一个单通道的体素张量，shape=(batch_size, 1, D, H, W)，初始化为 0
        # 0 表示“不属于任何部件”。
        part_vol = torch.zeros((batch_size, *occ_shape),
                            dtype=torch.int16,  # int16 也可以，最后要做 grid_sample 时再转成 float
                            device=occ_feats.device)

        # 遍历每个 batch，把体素对应的部分赋成 part_id+1
        for i in range(len(batch_layer_ranges) - 1):
            start, end = batch_layer_ranges[i], batch_layer_ranges[i + 1]
            mask = (occ_coords[:, 0] >= start) & (occ_coords[:, 0] < end)
            coords_b = occ_coords[mask]               # 形状 (N, 4)：[:,0] 是层索引，[:,1:] 是 (z,y,x)
            
            # 计算出每个体素的局部 part_id，比如用 coords_b[:,0] - start
            # 这样 part_id 的范围就是 [0, end-start-1]，共 end-start 个部件
            local_part_ids = coords_b[:, 0].short() - start   # 形状 (N,)
            
            # 拿到 (z, y, x) 三个维度
            zyx = coords_b[:, 1:].long().T   # 形状 (3, N)
            
            # 把单通道体素的对应位置，写入 “local_part_id + 1”。
            # 之所以要 +1，是因为我们让 0 表示“无部件”
            part_vol[i,zyx[0], zyx[1], zyx[2]] = local_part_ids + 1
        
        coords, sdf = compute_sdf_batch_torch(
            ( (part_vol > 0).float() * 2.0 - 1.0 ).squeeze(1),
            points,
            sample_num=self.hparams.sample_num,
            surface_bandwidth=self.hparams.surface_bandwidth,
        )
        part_ids=query_part_ids(part_vol,coords)
        occ_gt = (sdf <= 0).float().squeeze(-1)

        pt_coords = coords[0].detach().cpu().numpy()  # 点坐标 (N, 3)
        occ_vals = occ_gt[0].detach().cpu().numpy()  # 占据标签 (N,)

        # # 根据占据标签设置颜色：occupied (1) 为红色，不occupied (0) 为蓝色
        # colors = np.zeros((pt_coords.shape[0], 3), dtype=np.uint8)
        # colors[occ_vals > 0.5] = [255, 0, 0]   # 红色：占据
        # colors[occ_vals <= 0.5] = [0, 0, 255]    # 蓝色：非占据

        # # 创建点云并导出为 ply 文件
        # pc = trimesh.points.PointCloud(vertices=pt_coords, colors=colors)
        # ply_filename = f"train_occ_coords_epoch3_{self.current_epoch}.ply"
        # pc.export(ply_filename)

        t1 = time.time()
        data_time = t1 - t0

        # 计时：前向预测
        t2 = time.time()
        pred_occ, mu, rotate, vec, feats, feat_mean, feat_logvar,padded_feats,attn_feats,pad_mask, scores = self(
            sparse_tensor, batch_layer_ranges, coords, bbox_centers, bbox_radii,affine_matrices=affines#,feat_mask=feat_mask
        )  # pred_occ: (B, N, ...)
        logits = pred_occ.squeeze(-1)  # (B, N)
        t3 = time.time()
        pred_time = t3 - t2

        # 计时：损失计算
        t4 = time.time()
        res = occ_shape[0]
        gaussian_loss = 0.0

        el_occ_gt=torch.zeros_like(occ_gt)
        for i in range(len(batch_layer_ranges) - 1):
            start, end = batch_layer_ranges[i], batch_layer_ranges[i + 1]   
            mu_b = mu[start:end]  # (E, 3)
            rotate_b = rotate[start:end]
            vec_b = vec[start:end] # (E, 3)
            el_occ_gt[i]= multi_ellipsoid_occupancy(coords[i], mu_b.detach(), rotate_b.detach(), vec_b.detach(),  radius=1.75)

        occ_gt = torch.where(feat_mask.unsqueeze(1), occ_gt, el_occ_gt)
        loss_occ = F.binary_cross_entropy_with_logits(logits, occ_gt, pos_weight=torch.ones_like(occ_gt)*0.75)
        # # # 将 el_occ_gt 和 coords 绘制成有色点云并保存为 ply 文件

        # # 假设 el_occ_gt (B, N) 和 coords (B, N, 3)，这里取第一个 batch
        # pt_coords = coords[0].detach().cpu().numpy()  # 点坐标 (N, 3)
        # occ_vals = el_occ_gt[0].detach().cpu().numpy()  # 占据标签 (N,)

        # # 根据占据标签设置颜色：occupied (1) 为红色，不occupied (0) 为蓝色
        # colors = np.zeros((pt_coords.shape[0], 3), dtype=np.uint8)
        # colors[occ_vals > 0.5] = [255, 0, 0]   # 红色：占据
        # colors[occ_vals <= 0.5] = [0, 0, 255]    # 蓝色：非占据

        # # 创建点云并导出为 ply 文件
        # pc = trimesh.points.PointCloud(vertices=pt_coords, colors=colors)
        # ply_filename = f"train_occ_coords_epoch_{self.current_epoch}.ply"
        # pc.export(ply_filename)

                # 假设 el_occ_gt (B, N) 和 coords (B, N, 3)，这里取第一个 batch



        for b in range(vox_batch_size):
            mask = occ_coords[:, 0] == b
            occ_coords_b = -1 + 2 * (occ_coords[mask, 1:4].float() / res + 0.5 / res)
            mu_b = mu[b]
            rotate_b = rotate[b]
            vec_b = vec[b]
            # if occ_coords_b.shape[0] > 0:
            #     gaussian_loss += F.mse_loss(mu_b,occ_coords_b.mean(0))
            if occ_coords_b.shape[0] > 0:
                gaussian_loss += gaussian_nll_loss(
                    mu_b, rotate_b.T, vec_b**2, occ_coords_b
                )
        gaussian_loss = gaussian_loss / vox_batch_size

        feat_kl_loss = (
            -0.5
            * torch.sum(1 + feat_logvar - feat_mean.pow(2) - feat_logvar.exp())
            / feat_mean.size(0)
        )
        feat_l2_loss = feat_kl_loss
        attn_loss = 0.0

        # padded_last16 = padded_feats[..., -16:]
        # attn_last16 = attn_feats[..., -16:]

        # padded_last16_t = padded_last16.transpose(1, 2)  
        # similarity = torch.matmul(attn_last16, padded_last16_t)
        # similarity = similarity.masked_fill(~pad_mask.unsqueeze(1), float('-inf'))
        # pred = similarity.permute(0, 2, 1)

        ###################

        pred = scores.permute(0, 4, 1, 2, 3)
        target = part_ids.unsqueeze(1).unsqueeze(1).expand(-1, scores.shape[1], scores.shape[2], -1)
        attn_loss = F.cross_entropy(pred, target, ignore_index=-1)

        # # 计算正确率：先获得预测类别，再对比有效区域（忽略 -1）
        pred_labels = pred.argmax(dim=1)
        valid_mask = target != -1
        correct = (pred_labels == target) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()


        total_loss = (
            self.hparams.occ_w * loss_occ
            + self.hparams.gaussian_w * gaussian_loss
            + self.hparams.feat_l2_w * feat_l2_loss
            + attn_loss*0.1#0.1
        )
        t5 = time.time()
        loss_time = t5 - t4

        # 日志记录
        self.log(
            "occ",
            loss_occ,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "feat_kl",
            feat_kl_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "gaussian",
            gaussian_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "feat_l2",
            feat_l2_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "attn",
            attn_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "attn_acc",
            accuracy,
            prog_bar=True,
            on_epoch=False,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "total",
            total_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        # 记录耗时
        # self.log("time_data", data_time, prog_bar=False, on_epoch=True, on_step=True, batch_size=batch_size, sync_dist=True)
        # self.log("time_pred", pred_time, prog_bar=False, on_epoch=True, on_step=True, batch_size=batch_size, sync_dist=True)
        # self.log("time_loss", loss_time, prog_bar=False, on_epoch=True, on_step=True, batch_size=batch_size, sync_dist=True)

        if self.trainer.is_global_zero \
            and batch_idx == 0 \
            and ((self.current_epoch + 10) % 1 == 0 or self.current_epoch<10):
            # 可视化并保存第一个数据的形状
            mask = (occ_coords[:, 0] >= batch_layer_ranges[0]) & (occ_coords[:, 0] < batch_layer_ranges[1])
            feats_sparse = occ_feats[mask]
            coords_sparse = occ_coords[mask]
            input_shape = occ_shape
            part_num = batch_layer_ranges[1]- batch_layer_ranges[0]
            bbox_centers_vis = bbox_centers[batch_layer_ranges[0] : batch_layer_ranges[1]]
            bbox_radii_vis = bbox_radii[batch_layer_ranges[0] : batch_layer_ranges[1]]
            pts_cloud = points[0]
            self.export_epoch_mesh(
                feats_sparse=feats_sparse,
                coords_sparse=coords_sparse,
                input_shape=input_shape,
                part_num=part_num,
                bbox_centers=bbox_centers_vis,
                bbox_radii=bbox_radii_vis,
                pts_cloud=pts_cloud,
                affine=None,
                filename_suffix="vis"
            )
            []
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        occ_coords = batch["occ_coords"]
        occ_feats = batch["occ_feats"]
        occ_shape = batch.get("occ_shape", [256, 256, 256])
        vox_batch_size = int(occ_coords[:, 0].max().item()) + 1
        sparse_tensor = sp.SparseConvTensor(
            features=occ_feats,
            indices=occ_coords,
            spatial_shape=occ_shape,
            batch_size=vox_batch_size,
        )

        bbox_centers = batch["occ_bbox_centers"]
        bbox_radii = batch["occ_bbox_radii"]

        # 编码体素特征，拿到中间输出
        feats, scaled_mu, rotate, scaled_vec, feat_mean, feat_logvar = \
            self.encode_voxel_features(sparse_tensor, bbox_centers, bbox_radii)

        # 将中间结果推入列表
        self.scaled_mu_list.append(scaled_mu.detach())
        self.rotate_list.append(rotate.detach())
        self.scaled_vec_list.append(scaled_vec.detach())
        self.feats_list.append(feat_mean.detach())
        self.folders_list.append(batch["folders"][0])

    def on_predict_epoch_end(self):
        # 配置的最大长度（若未设置则默认为20）
        max_len = getattr(self.hparams, "max_length", 20)

        def pad(tensor, target_len, dim=0):
            cur_len = tensor.shape[dim]
            if cur_len > target_len:
                # 如果超过上限，抛出错误
                raise ValueError("数据长度超过最大限制")
            elif cur_len < target_len:
                pad_shape = list(tensor.shape)
                pad_shape[dim] = target_len - cur_len
                pad_tensor = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
                return torch.cat([tensor, pad_tensor], dim=dim)
            else:
                return tensor

        # 批量计算所有样本中长度不超过max_len的编号
        valid_indices = [i for i, mu in enumerate(self.scaled_mu_list) if mu.shape[0] <= max_len]
        if len(valid_indices) == 0:
            # 若所有数据都超过上限，则直接返回
            return

        # 根据有效编号过滤数据
        valid_scaled_mu   = [self.scaled_mu_list[i] for i in valid_indices]
        valid_rotate      = [self.rotate_list[i] for i in valid_indices]
        valid_scaled_vec  = [self.scaled_vec_list[i] for i in valid_indices]
        valid_feats       = [self.feats_list[i] for i in valid_indices]
        valid_folders = [self.folders_list[i] for i in valid_indices]

        # 批量填充各项数据
        padded_centers = [pad(mu, max_len, dim=0) for mu in valid_scaled_mu]
        ellipsoid_centers = torch.stack(padded_centers, dim=0)          # (valid_num, max_len, 3)

        padded_rot = [pad(r.view(r.shape[0], -1), max_len, dim=0) for r in valid_rotate]
        ellipsoid_rot = torch.stack(padded_rot, dim=0)                 # (valid_num, max_len, 9)

        padded_r = [pad(v, max_len, dim=0) for v in valid_scaled_vec]
        ellipsoid_r = torch.stack(padded_r, dim=0)                     # (valid_num, max_len, 3)

        # 批量构造 flags：根据每个样本有效长度生成标志矩阵
        valid_lengths = torch.tensor(
            [mu.shape[0] for mu in valid_scaled_mu], device=ellipsoid_centers.device
        )
        num_valid = valid_lengths.shape[0]
        arange_tensor = torch.arange(max_len, device=ellipsoid_centers.device).unsqueeze(0).expand(num_valid, -1)
        flags = (arange_tensor < valid_lengths.unsqueeze(1)).unsqueeze(-1).float()  # (valid_num, max_len, 1)

        # 保存 SPA 模式下的每个样本文件
        if getattr(self.hparams, "export_mode", None) == "SPA":
            for i in range(num_valid):
                # 新增第1行: 混合系数 φ（与 flags 相同，但转换为浮点数，共 16 个）
                L0 = flags[i].float().view(1, -1)         # (1, 16)
                # 第2行: 均值向量 μ，长度为 3x16=48
                L1 = ellipsoid_centers[i].view(1, -1)      # (1, 48)
                # 第3行: 特征值 eigen，长度为 16x3=48（这里取 ellipsoid_r）
                L2 = ellipsoid_r[i].view(1, -1)**2            # (1, 48)
                # 第4行: 精度矩阵参数 p，长度为 9x16=144（这里取 ellipsoid_rot）
                L3 = ellipsoid_rot[i].view(ellipsoid_rot[i].shape[0],3,3).transpose(-2,-1).reshape(1, -1)          # (1, 144)
                # 第5行: 包含标志 included，16 个整数
                L4 = flags[i].long().view(1, -1)           # (1, 16)
                txt_save_path = os.path.join(self.hparams.save_dir, f"{valid_folders[i]}.txt")
                with open(txt_save_path, "w") as f:
                    np.savetxt(f, L0.cpu().numpy(), fmt="%.6f")
                    np.savetxt(f, L1.cpu().numpy(), fmt="%.6f")
                    np.savetxt(f, L2.cpu().numpy(), fmt="%.6f")
                    np.savetxt(f, L3.cpu().numpy(), fmt="%.6f")
                    np.savetxt(f, L4.cpu().numpy(), fmt="%d")
                
                # 每个样本的 feat 单独保存为 .pkl 文件
                padded_feat = pad(valid_feats[i], max_len, dim=0).cpu()
                pkl_save_path = os.path.join(self.hparams.save_dir, f"{valid_folders[i]}.pkl")
                with open(pkl_save_path, 'wb') as f:
                    pickle.dump(padded_feat, f, protocol=pickle.HIGHEST_PROTOCOL)


        # 保存 DIFF 模式的结果：将所有椭球参数以 centers, rot, r 在最后一维堆叠，并保存到一个 pth 文件中
        elif getattr(self.hparams, "export_mode", None) == "DIFF":
            # 将椭球参数按最后一维堆叠: centers (max_len,3), rot (max_len,9), r (max_len,3) -> (max_len,15)
            diff_params = torch.cat([ellipsoid_centers, ellipsoid_rot, ellipsoid_r], dim=-1)  # (valid_num, max_len, 15)
            diff_params_save_path = os.path.join(self.hparams.save_dir, "diff_params.pth")
            torch.save(diff_params.cpu(), diff_params_save_path)

            # 同样将 feat 进行堆叠，每个样本填充到 max_len 后堆叠保存为一个 pth 文件
            padded_valid_feats = [pad(feat, max_len, dim=0) for feat in valid_feats]
            diff_feats = torch.stack(padded_valid_feats, dim=0)  # (valid_num, max_len, feat_dim)
            diff_feats_save_path = os.path.join(self.hparams.save_dir, "diff_feats.pth")
            torch.save(diff_feats.cpu(), diff_feats_save_path)

        if getattr(self.hparams, "export_shape", False):
            for i in range(num_valid):
                # 拼接高斯参数: centers, 旋转矩阵展平, 半轴长
                gaussians = torch.cat([
                    valid_scaled_mu[i], 
                    valid_rotate[i].view(valid_scaled_mu[i].shape[0], -1),
                    valid_scaled_vec[i]
                ], dim=-1)
                # 使用高斯参数和对应的特征重构形状，获得顶点和面
                verts, faces = self.reconstruct_from_gaussians(gaussians, valid_feats[i],need_sqrt=False)
                # 使用 trimesh 保存为 OBJ 文件
                mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())
                obj_save_path = os.path.join(self.hparams.save_dir, f"{valid_folders[i]}.obj")
                mesh.export(obj_save_path, file_type='obj')

        if getattr(self.hparams, "export_gaussians", False):
            for i in range(num_valid):
                # 拼接高斯参数: centers, 旋转矩阵展平, 半轴长
                gaussians = torch.cat([
                    valid_scaled_mu[i], 
                    valid_rotate[i].view(valid_scaled_mu[i].shape[0], -1),
                    valid_scaled_vec[i]
                ], dim=-1)
                # 保存为 pth 文件
                verts, faces = self.visualize_ellipsoids(gaussians,need_sqrt=False)
                # 使用 trimesh 保存为 OBJ 文件
                mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())
                obj_save_path = os.path.join(self.hparams.save_dir, f"{valid_folders[i]}_gaussians.obj")
                mesh.export(obj_save_path, file_type='obj')
        # 清空列表，为下个 epoch 做准备
        self.scaled_mu_list.clear()
        self.rotate_list.clear()
        self.scaled_vec_list.clear()
        self.feats_list.clear()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def sample_affines(self, batch_size):
        return sample_affines(batch_size).to(self.device)
    
    @torch.no_grad()
    def export_epoch_mesh(
        self, feats_sparse, coords_sparse, input_shape,part_num,bbox_centers, bbox_radii, pts_cloud,
        affine=None, filename_suffix=None
    ):

        # 1) 构造稀疏张量
        x_sp0 = sp.SparseConvTensor(
            feats_sparse, coords_sparse,
            spatial_shape=input_shape, batch_size=part_num
        )
        # 2) 把稀疏体素当成点云，并着色
        coords_np   = coords_sparse.cpu().numpy()      # (N,4)
        batch_ids   = coords_np[:,0].astype(int)       # 哪个 batch
        xyz         = coords_np[:,1:4].astype(np.float32)
        scale_vox   = np.array(input_shape) - 1
        xyz         = (xyz/scale_vox)*2.0 - 1.0         # 归一化到[-1,1]

        # 从 tab10 里取颜色
        cmap   = plt.get_cmap('tab10')
        # 只取 rgb 部分，放大到 [0,255]
        rgb     = np.array([cmap(b%10)[:3] for b in batch_ids])*255
        # 拼出 RGBA
        rgba_pts = np.concatenate([
            rgb.astype(np.uint8),
            np.full((rgb.shape[0],1),255, dtype=np.uint8)
        ], axis=1)  # (N,4)

        # 把点云当作一个没有面的 Trimesh
        pc_mesh = trimesh.Trimesh(
            vertices=xyz,
            faces   =np.empty((0,3), dtype=np.int64),
            process=False
        )
        pc_mesh.visual.vertex_colors = rgba_pts


        was_training = self.training
        self.eval()
        # 2) 准备仿射矩阵：如果传入单个 4x4，就扩成 [1,4,4]
        if affine is not None:
            aff = affine.to(self.device).to(x_sp0.features.dtype)
            affines = aff.unsqueeze(0) if aff.dim() == 2 else aff
        else:
            affines = None

        # 3) 生成查询点网格
        R = self.hparams.export_resolution
        grid = torch.linspace(-1, 1, R, device=self.device)
        xx, yy, zz = torch.meshgrid(grid, grid, grid, indexing="ij")
        pts = torch.stack([xx, yy, zz], -1).view(1, -1, 3)  # [1, R^3, 3]

        # 5) 编码并预测高斯
        _, scaled_mu, rotate, scaled_vec, feats, feat_logvar = self.encode_voxel_features(x_sp0,bbox_centers, bbox_radii)

        # 5.5) 导出高斯椭球为PLY
        # 获取每个高斯的中心、旋转、缩放
        centers = scaled_mu.detach().cpu().numpy()  # (N, 3)
        rot_mats = rotate.detach().cpu().numpy()    # (N, 3, 3)
        scales = scaled_vec.detach().cpu().numpy()  # (N, 3)

        ellipsoids = []
        for i, (c, Rmat, s) in enumerate(zip(centers, rot_mats, scales)):
            # 计算当前椭球的六个顶点
            el_points = ellipsoid_vertices(
                torch.tensor(c[None], device=feats_sparse.device, dtype=feats_sparse.dtype),
                torch.tensor(Rmat[None], device=feats_sparse.device, dtype=feats_sparse.dtype),
                torch.tensor(s[None], device=feats_sparse.device, dtype=feats_sparse.dtype)
            ).cpu().numpy().reshape(6, 3)

            # 创建椭球体
            m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            m.apply_scale(s)
            # 先做旋转再平移
            transform = np.vstack([
            np.hstack([Rmat, np.zeros((3,1))]),
            np.array([0,0,0,1])
            ])
            m.apply_transform(transform)
            m.apply_translation(c)

            # 给这个椭球统一着色（取同一个 cmap 颜色）
            color_rgb = (np.array(cmap(i%10)[:3])*255).astype(np.uint8)
            color_rgba= np.concatenate([color_rgb, [255]], axis=0)  # (4,)
            m.visual.vertex_colors = np.tile(
            color_rgba[None,:],
            (m.vertices.shape[0], 1)
            )

            # 展示六个点：用小球体可视化，六个点分别上色，不同椭球的同一个点上一个颜色
            # 预定义六种颜色（可自定义或用tab10前6种）
            point_cmap = plt.get_cmap('tab10')
            point_colors = [(np.array(point_cmap(j)[:3]) * 255).astype(np.uint8) for j in range(6)]
            for j, pt in enumerate(el_points):
                sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.025)
                sphere.apply_translation(pt)
                # 六个点分别上色，不同椭球的同一个点用同一个颜色
                color_rgba_pt = np.concatenate([point_colors[j], [255]], axis=0)
                sphere.visual.vertex_colors = np.tile(
                    color_rgba_pt[None, :],
                    (sphere.vertices.shape[0], 1)
                )
                ellipsoids.append(sphere)

            ellipsoids.append(m)

        # 4) 合并所有几何体
        all_mesh = trimesh.util.concatenate([pc_mesh] + ellipsoids)

        # 5) 导出为单个 PLY
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        out_path = os.path.join(
            self.output_dir,
            f"epoch_{self.current_epoch}{suffix}_vox_and_ellipsoids.ply"
        )
        ply_data = all_mesh.export(file_type='ply')
        mode     = "wb" if isinstance(ply_data, (bytes, bytearray)) else "w"
        with open(out_path, mode) as f:
            f.write(ply_data)
        # 6) 合并高斯特征
        padded_feats, pad_mask = self.mix_features(
            feats, scaled_mu, rotate, scaled_vec,
            batch_layer_ranges=[0, feats.shape[0]],  # 只处理一个 batch
            affine_matrices=affines,
            #feat_mask=torch.ones(1, device=feats.device, dtype=feats.dtype)*0
        )

        # 7) 分批 decode 得到体素场
        preds = []
        flat_pts = pts.view(-1, 3)
        for i in range(0, flat_pts.size(0), self.hparams.export_batch):
            chunk = flat_pts[i : i + self.hparams.export_batch].unsqueeze(0)
            p = self.decode(padded_feats, pad_mask, chunk, affine_matrices=affines)
            preds.append(p.view(-1))
        occ = torch.cat(preds).view(R, R, R).cpu().numpy()

        # 8) marching cubes & 导出 PLY
        min_val, max_val = occ.min(), occ.max()
        level = 0.0 if (min_val <= 0.0 <= max_val) else float(occ.mean())
        verts, faces, normals, _ = marching_cubes(
            occ.astype(np.float32), level=level
        )
        scale = np.array([R - 1, R - 1, R - 1], dtype=np.float32)
        verts = (verts / scale) * 2.0 - 1.0
        pts_np = pts_cloud.cpu().numpy()
        if affine is not None:
            # Apply affine transformation to vertices
            verts_h = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float32)])
            verts = (affine.cpu().numpy() @ verts_h.T).T[:, :3]

            # Apply affine transformation to point cloud
            pts_h = np.hstack([pts_np, np.ones((pts_np.shape[0], 1), dtype=np.float32)])
            pts_np = (affine.cpu().numpy() @ pts_h.T).T[:, :3]
        combined_verts = np.vstack([verts, pts_np])
        combined_faces = faces.copy()
        zero_norms = np.zeros_like(pts_np)
        combined_norms = np.vstack([normals, zero_norms])

        mesh_all = trimesh.Trimesh(
            vertices=combined_verts,
            faces=combined_faces,
            vertex_normals=combined_norms,
            process=False,
        )
        ply_data = mesh_all.export(file_type="ply")
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        path = os.path.join(
            self.output_dir,
            f"epoch_{self.current_epoch}{suffix}.ply"
        )
        mode = "wb" if isinstance(ply_data, (bytes, bytearray)) else "w"

        if was_training:
            self.train()
        with open(path, mode) as f:
            f.write(ply_data)

        self.log("exported", True)

    @torch.no_grad()
    def extract_gaussians(self, x_sparse: sp.SparseConvTensor, bbox_centers: torch.Tensor, bbox_radii: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        不再使用小球逻辑，改为从 voxel 特征中提取高斯参数。
        返回:
        gaussians_list: list[Tensor] 每个 batch 的高斯参数 [N,15]，包含:
            - 中心: (x, y, z) [3]
            - 旋转矩阵: 3x3 展平为 9 个数
            - 半轴长: (l1, l2, l3) [3]
        codes_list: list[Tensor] 每个 batch 的特征编码 [N, C]
        """
        feats, scaled_mu, rotate, scaled_vec, feat_mean, feat_logvar = self.encode_voxel_features(x_sparse, bbox_centers, bbox_radii)
        # flatten旋转矩阵:
        rotate_flat = rotate.view(rotate.shape[0], -1)  # (N, 9)
        gaussians = torch.cat([scaled_mu, rotate_flat, scaled_vec**2], dim=-1)  # (N, 3+9+3=15)
        return [gaussians], [feats]
    
    def visualize_ellipsoids(
        self, gaussians: torch.Tensor, mode: str = "ellipsoid", need_sqrt=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        可视化椭球的方法，根据 mode 参数选择输出方式：
            - "ellipsoid": 为每个高斯生成完整的椭球网格，同时显示六个顶点
            - "points": 为每个高斯生成六个顶点的小球，组成一个网格
        返回:
            verts: Tensor [V, 3]
            faces: Tensor [F, 3]
        """
        try:
            device = gaussians.device
            gaussians_np = gaussians.detach().cpu().numpy()
            centers = gaussians_np[:, :3]
            rot_mats = gaussians_np[:, 3:12].reshape(-1, 3, 3)
            scales = gaussians_np[:, 12:15]
            ellipsoid_meshes = []
            cmap = plt.get_cmap("tab10")
            point_cmap = plt.get_cmap("tab10")
            # 遍历每个高斯参数
            for i, (c, Rmat, s) in enumerate(zip(centers, rot_mats, scales)):
                # 使用 ellipsoid_vertices 计算椭球的 6 个关键顶点
                center_t = torch.tensor(c[None], device=device, dtype=torch.float32)
                Rmat_t = torch.tensor(Rmat[None], device=device, dtype=torch.float32)
                scale_t = torch.tensor(s[None], device=device, dtype=torch.float32)
                el_points = ellipsoid_vertices(center_t, Rmat_t, scale_t)
                el_points = el_points.view(-1, 3).cpu().numpy()  # (6, 3)
    
                if mode == "ellipsoid":
                    # 生成单位球，并按半轴缩放、旋转和平移
                    m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
                    m.apply_scale(s)
                    transform = np.vstack([
                        np.hstack([Rmat, np.zeros((3, 1))]),
                        np.array([0, 0, 0, 1])
                    ])
                    m.apply_transform(transform)
                    m.apply_translation(c)
    
                    # 着色：取 tab10 柱状图对应颜色
                    color_rgb = (np.array(cmap(i % 10)[:3]) * 255).astype(np.uint8)
                    color_rgba = np.concatenate([color_rgb, [255]], axis=0)
                    m.visual.vertex_colors = np.tile(
                        color_rgba[None, :],
                        (m.vertices.shape[0], 1)
                    )
                    ellipsoid_meshes.append(m)
                elif mode == "points":
                    # 仅为每个高斯显示六个顶点的小球
                    point_colors = [
                        (np.array(point_cmap(j)[:3]) * 255).astype(np.uint8)
                        for j in range(6)
                    ]
                    for j, pt in enumerate(el_points):
                        sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
                        sphere.apply_translation(pt)
                        color_rgba_pt = np.concatenate([point_colors[j], [255]], axis=0)
                        sphere.visual.vertex_colors = np.tile(
                            color_rgba_pt[None, :],
                            (sphere.vertices.shape[0], 1)
                        )
                        ellipsoid_meshes.append(sphere)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
            # 合并所有网格
            combined_mesh = trimesh.util.concatenate(ellipsoid_meshes)
            verts = torch.from_numpy(combined_mesh.vertices).float().to(device)
            faces = torch.from_numpy(combined_mesh.faces.astype(np.int64)).long().to(device)
            return verts, faces
        except Exception as e:
            print("可视化椭球失败，返回单位球网格:", e)
            sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            verts = torch.from_numpy(sphere_mesh.vertices).float().to(device)
            faces = torch.from_numpy(sphere_mesh.faces.astype(np.int64)).long().to(device)
            return verts, faces
    @torch.no_grad()
    def reconstruct_from_gaussians(self, gaussians: torch.Tensor, codes: torch.Tensor, affine: torch.Tensor = None,need_sqrt=True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用 SPLICE_Module 的 voxel 特征解码器，通过高斯参数重构 Occupancy 网格，并用 marching_cubes 提取顶点。
        输入:
        gaussians: Tensor [N,15]，每行包含中心(3)、旋转矩阵展平(9)和半轴长(3)
        codes: Tensor [N, C] 对应每个高斯的特征编码
        affine: 可选的仿射矩阵
        返回:
        verts: Tensor [V,3]
        faces: Tensor [F,3]
        如果失败，返回单位球顶点和面。
        """
        device = codes.device
        try:
            # 从 gaussians 中提取高斯参数
            center = gaussians[:, :3]
            rotate = gaussians[:, 3:12].view(-1, 3, 3)
            scaled_vec = gaussians[:, 12:]
            if need_sqrt:
                scaled_vec = scaled_vec.sqrt()
                rotate = rotate.transpose(-1, -2)
            # 构建 batch_layer_ranges 用于特征混合（这里只有一个 batch）
            batch_layer_ranges = [0, codes.shape[0]]
            padded_feats, pad_mask = self.mix_features(codes, center, rotate, scaled_vec, batch_layer_ranges, affine_matrices=affine)
            # 构建网格查询点
            R = self.hparams.export_resolution
            s=1
            grid = torch.linspace(-s, s, R, device=device)
            xx, yy, zz = torch.meshgrid(grid, grid, grid, indexing="ij")
            pts = torch.stack([xx, yy, zz], -1).view(1, -1, 3)
            # 分批 decode 得到 occupancy/SDF
            preds = []
            flat_pts = pts.view(-1, 3)
            for i in range(0, flat_pts.size(0), self.hparams.export_batch):
                chunk = flat_pts[i : i + self.hparams.export_batch].unsqueeze(0)
                p = self.decode(padded_feats, pad_mask, chunk, affine_matrices=affine)
                preds.append(p.view(-1))
            occ = torch.cat(preds).view(R, R, R).cpu().numpy()
            # marching_cubes 提取网格
            min_val, max_val = occ.min(), occ.max()
            level = 0.0 if (min_val <= 0.0 <= max_val) else float(occ.mean())
            verts, faces, normals, _ = marching_cubes(occ.astype(np.float32), level=level)
            scale = np.array([R - 1, R - 1, R - 1], dtype=np.float32)
            verts = (verts / scale) * (2*s) - s
            verts = torch.from_numpy(verts).float()
            faces = torch.from_numpy(faces.astype(np.int64)).long()
            return verts, faces
        except Exception as e:
            print("重构失败，返回单位球:", e)
            sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            verts = torch.from_numpy(sphere_mesh.vertices).float()
            faces = torch.from_numpy(sphere_mesh.faces.astype(np.int64)).long()
        return verts, faces
