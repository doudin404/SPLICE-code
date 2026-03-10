import os
import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from trimesh.creation import icosphere
import open3d as o3d
import time
from skimage.measure import marching_cubes
import json

@torch.no_grad()
def mesh_to_occupancy_voxel_ray(mesh, resolution=128, padding=0.05, super_sample_factor=0):
    """
    从 mesh 生成占用体素网格，支持超采样，并计算超采样区域的平均值。
    采样范围始终为 [-1, 1] 立方体。
    参数:
      mesh: 归一化的 mesh 对象。
      resolution: 体素网格分辨率。
      padding: 对角线扩展比例（无效，始终采样[-1,1]）。
      super_sample_factor: 超采样倍数，2^n 倍。
    返回:
      vox: 超采样后的体素网格 (torch.Tensor, float32, -1~1)。
      min_b: [-1, -1, -1]
      max_b: [1, 1, 1]
    """
    # 固定采样范围为 [-1, 1]
    min_b = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    max_b = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    factor = 2 ** super_sample_factor
    resolution_new = resolution * factor

    # 计算体素中心坐标
    xs = np.linspace(min_b[0], max_b[0], resolution_new, endpoint=False) + (max_b[0] - min_b[0]) / resolution_new / 2
    ys = np.linspace(min_b[1], max_b[1], resolution_new, endpoint=False) + (max_b[1] - min_b[1]) / resolution_new / 2
    zs = np.linspace(min_b[2], max_b[2], resolution_new, endpoint=False) + (max_b[2] - min_b[2]) / resolution_new / 2

    xg, yg = np.meshgrid(xs, ys, indexing='xy')

    # 计算射线的原点和方向
    origins = np.column_stack([xg.ravel(), yg.ravel(), np.full(xg.size, min_b[2] - 1e-4)])
    dirs = np.tile([0.0, 0.0, 1.0], (origins.shape[0], 1))

    # 获取射线与网格的交点
    locs, idx_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)

    # 创建体素网格
    vox = np.zeros((resolution_new, resolution_new, resolution_new), dtype=np.uint8)

    # 按射线分组交点
    buckets = [[] for _ in range(origins.shape[0])]
    for z, ridx in zip(locs[:, 2], idx_ray):
        buckets[ridx].append(z)

    # 对每个分组内的 z 交点进行插值，填充体素网格
    for ridx, zlist in enumerate(buckets):
        if not zlist:
            continue
        zlist.sort()
        if len(zlist) % 2 == 1:
            zlist = zlist[:-1]  # 忽略最后的孤立交点（非流形情况）

        yi = ridx // resolution_new  
        xi = ridx % resolution_new
        for z0, z1 in zip(zlist[0::2], zlist[1::2]):
            k0 = np.searchsorted(zs, z0, side='left')
            k1 = np.searchsorted(zs, z1, side='right')
            vox[xi, yi, k0:k1] = 1

    # 超采样后，计算每个新的体素位置的平均值
    if super_sample_factor > 0:
        vox_t = torch.from_numpy(vox.astype(np.float32))
        vox_t = vox_t.view(resolution, factor, resolution, factor, resolution, factor)
        vox_t = vox_t.mean(dim=(1, 3, 5))
    else:
        vox_t = torch.from_numpy(vox.astype(np.float32))

    # 将体素值从 0~1 映射到 -1~1
    vox_t = vox_t * 2.0 - 1.0

    return vox_t, min_b, max_b

class VoxelDataset(Dataset):
    def __init__(self, root_dir, data_name=None, resolution=256, sample_num=100000):
        self.root_dir = root_dir
        self.resolution = resolution
        self.data_name = data_name
        self.sample_num = sample_num

        if self.data_name == 'partnet':
            self.samples = sorted([folder for folder in os.listdir(root_dir) if folder != 'shapenet'])
            metadata_path = os.path.join(self.root_dir, "shapenet", "partnet_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata_list = json.load(f)
                # Create a mapping from partnet_id to shapenet_id for fast lookup.
                self.partnet_metadata = {item.get("partnet_id"): item.get("shapenet_id") 
                                         for item in metadata_list if item.get("partnet_id") is not None}
                # Filter self.samples to only include those whose corresponding shapenet_id folder exists.
                filtered_samples = []
                for sample in self.samples:
                    shapenet_id = self.partnet_metadata.get(sample)
                    if shapenet_id and os.path.isdir(os.path.join(self.root_dir, "shapenet", shapenet_id)):
                        filtered_samples.append(sample)
                self.samples = sorted(filtered_samples)
            else:
                self.partnet_metadata = None

        elif self.data_name == 'dae_net':
            suffix = '_ours_segmentation_on_GT.ply'
            files = [f for f in os.listdir(root_dir) if f.endswith(suffix)]
            indices = []
            for f in files:
                try:
                    idx = int(f.split('_')[0])
                    indices.append(idx)
                except:
                    continue
            self.samples = [str(i) for i in sorted(indices)]

            vox_txt = os.path.join(self.root_dir, 'shapenet_vox.txt')
            if not os.path.exists(vox_txt):
                raise FileNotFoundError(f"Cannot find {vox_txt}")
            with open(vox_txt, 'r') as f:
                self.vox_list = [line.strip() for line in f if line.strip()]

            self.cache_root = os.path.join(self.root_dir, 'cache')
            os.makedirs(self.cache_root, exist_ok=True)
        else:
            raise ValueError(f"Unsupported data_name: {self.data_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key = self.samples[idx]

        if self.data_name == 'partnet':
            folder_path = os.path.join(self.root_dir, key)
            pc_np = os.path.join(folder_path, 'points.npy')
            occ_coords_np = os.path.join(folder_path, 'occ_coords.npy')
            occ_labels_np = os.path.join(folder_path, 'occ_labels.npy')
            if all(os.path.exists(p) for p in (pc_np, occ_coords_np, occ_labels_np)):
                points = torch.from_numpy(np.load(pc_np)).float()
                occ_coords = torch.from_numpy(np.load(occ_coords_np)).int()
                occ_labels = torch.from_numpy(np.load(occ_labels_np)).int()
            else:
                mesh, points, labels = self._generate_partnet(key, folder_path)
                occ_coords, occ_labels = self._postprocess(mesh, points, labels)

                np.save(pc_np, points)
                points = torch.from_numpy(points).float()
                np.save(occ_coords_np, occ_coords.numpy())
                np.save(occ_labels_np, occ_labels.numpy())
        else:
            ply_file = os.path.join(self.root_dir, f"{key}_ours_segmentation_on_GT.ply")
            cached = os.path.join(self.cache_root, key)
            os.makedirs(cached, exist_ok=True)
            pc_np = os.path.join(cached, 'points.npy')
            occ_coords_np = os.path.join(cached, 'occ_coords.npy')
            occ_labels_np = os.path.join(cached, 'occ_labels.npy')
            if all(os.path.exists(p) for p in (pc_np, occ_coords_np, occ_labels_np)):
                points = torch.from_numpy(np.load(pc_np)).float()
                occ_coords = torch.from_numpy(np.load(occ_coords_np)).int()
                occ_labels = torch.from_numpy(np.load(occ_labels_np)).int()
            else:
                                                # 先加载带颜色的 PLY，得到 ply_pts 和 ply_labels
                ply_pts, ply_labels = self._load_labeled_ply(ply_file)
                # 对 ply_pts 进行归一化，与 PartNet 中 orig_points 归一化方式一致
                mins, maxs = ply_pts.min(0), ply_pts.max(0)
                center_pts = (mins + maxs) / 2
                ply_pts = (ply_pts - center_pts) * (1.95 / np.linalg.norm(maxs - mins))
                # 旋转 ply_pts 180°，绕 Y 轴（使标签与 mesh 对齐）
                ply_pts[:, 0] *= -1
                ply_pts[:, 2] *= -1
                mins, maxs = ply_pts.min(0), ply_pts.max(0)
                center_pts = (mins + maxs) / 2
                ply_pts = (ply_pts - center_pts) * (2.0 / np.linalg.norm(maxs - mins))

                # 再生成 mesh 并统一从 mesh 表面采样固定数量点（mesh 已在 _generate_dae_net_mesh 中归一化）
                mesh = self._generate_dae_net_mesh(key)
                sampled_pts, _ = trimesh.sample.sample_surface(mesh, self.sample_num)
                # # 用 ply_pts 建树，为 sampled_pts 赋标签
                # tree_ply = cKDTree(ply_pts)
                # idxs = tree_ply.query(sampled_pts, k=1)[1]
                # labels = ply_labels[idxs]
                # 之后通用后处理得到 occ_coords 和 occ_labels
                occ_coords, occ_labels = self._postprocess(mesh, ply_pts, ply_labels)
                points = torch.from_numpy(sampled_pts).float()

                np.save(pc_np, sampled_pts)
                np.save(occ_coords_np, occ_coords.numpy())
                np.save(occ_labels_np, occ_labels.numpy())

        unique_labels = torch.unique(occ_labels)
        layers = []
        for lbl in unique_labels.tolist():
            mask = occ_labels == lbl
            coords_label = occ_coords[mask]
            feats_label = torch.ones((coords_label.size(0), 1), dtype=torch.float32)
            min_idx = coords_label.min(dim=0).values.float()
            max_idx = coords_label.max(dim=0).values.float()
            res = self.resolution
            min_xyz = -1 + 2 * (min_idx / res + 0.5 / res)
            max_xyz = -1 + 2 * (max_idx / res + 0.5 / res)
            center = (min_xyz + max_xyz) / 2
            radius = ((max_xyz - min_xyz) / 2).max()
            layers.append({'coords': coords_label, 'feats': feats_label,
                           'label': torch.tensor(lbl, dtype=torch.long),
                           'center': center, 'radius': radius})

        return {'points': points, 'folder': key, 'layers': layers}

    def _load_labeled_ply(self, ply_path):
        # 使用 trimesh 读取带颜色的点云
        mesh = trimesh.load(ply_path, force='mesh')
        pts = mesh.vertices.astype(np.float32)
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3].astype(np.uint8)
        else:
            raise ValueError(f"PLY 文件 {ply_path} 中未找到顶点颜色")
        ct = [tuple(c) for c in colors]
        uniq = list({c for c in ct})
        cmap = {c: i for i, c in enumerate(uniq)}
        lbs = np.array([cmap[c] for c in ct], dtype=np.int64)
        return pts, lbs

    def _generate_partnet(self, key, folder_path):
        shapenet_dir = os.path.join(self.root_dir, 'shapenet')
        obj_files = [f for f in os.listdir(folder_path) if f.endswith('.obj')]
        if obj_files:
            base = os.path.splitext(max(obj_files, key=lambda x: len(os.path.splitext(x)[0])))[0]
        else:
            folder_name = os.path.basename(folder_path)
            base = self.partnet_metadata.get(folder_name)
            if base is None:
                raise ValueError(f"partnet_id {folder_name} not found in metadata.")
        sd = os.path.join(shapenet_dir, base)
        npz = os.path.join(sd, 'psr.npz')
        if os.path.exists(npz):
            psr = np.load(npz)
            vol = psr['psr'] if 'psr' in psr.files else psr[psr.files[0]]
            v, f, n, val = marching_cubes(vol, level=0)
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0,1,0]))
        else:
            mp = os.path.join(folder_path, f"{key}.obj")
            mesh = trimesh.load(mp, force='mesh')
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0,1,0]))
        bb = mesh.bounding_box.bounds
        mesh.apply_translation(-((bb[0]+bb[1])/2))
        mesh.apply_scale(2.0/np.linalg.norm(bb[1]-bb[0]))
        pts_path = os.path.join(folder_path, 'point_sample', 'pts-10000.txt')
        lbl_path = os.path.join(folder_path, 'point_sample', 'label-10000.txt')
        op = np.loadtxt(pts_path)
        ol = np.loadtxt(lbl_path, dtype=np.int64)
        mn, mx = op.min(0), op.max(0)
        op = (op - (mn+mx)/2) * (2.0/np.linalg.norm(mx-mn))
        uniq_lbl = np.unique(ol)
        mpng = {l: l for l in uniq_lbl}
        mpset = set()
        for t in uniq_lbl:
            for s in uniq_lbl:
                if t==s or (s,t) in mpset: continue
                pt_t = op[ol==t]
                pt_s = op[ol==s]
                if pt_t.size and pt_s.size:
                    d = cKDTree(pt_t).query(pt_s, k=1)[0].mean()
                    if d<0.05:
                        mpng = {k:(t if v==s else v) for k,v in mpng.items()}
                        mpset.add((t,s))
        nl = np.vectorize(mpng.get)(ol)
        rm = {old:i for i,old in enumerate(sorted(np.unique(nl)))}
        ol = np.vectorize(rm.get)(nl)
        sp, _ = trimesh.sample.sample_surface(mesh, self.sample_num)
        idxs = cKDTree(op).query(sp, k=1)[1]
        lbs = ol[idxs]
        return mesh, sp, lbs

    def _generate_dae_net_mesh(self, key):
        si = self.vox_list[int(key)]
        sd = os.path.join(self.root_dir, 'shapenet', si)
        npz = os.path.join(sd, 'psr.npz')
        if os.path.exists(npz):
            psr = np.load(npz)
            vol = psr['psr'] if 'psr' in psr.files else psr[psr.files[0]]
            v, f, n, val = marching_cubes(vol, level=0)
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0,1,0]))
        else:
            mp = os.path.join(self.root_dir, key, si + '.obj')
            mesh = trimesh.load(mp, force='mesh')
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0,1,0]))
        bb = mesh.bounding_box.bounds
        mesh.apply_translation(-((bb[0]+bb[1])/2))
        mesh.apply_scale(2.0/np.linalg.norm(bb[1]-bb[0]))
        return mesh

    def _postprocess(self, mesh, sampled_pts, labels):
        voxels, _, _ = mesh_to_occupancy_voxel_ray(mesh, resolution=self.resolution)
        vox_bin = (voxels > 0).cpu().numpy().astype(np.uint8)
        mask = binary_fill_holes(vox_bin)
        coords = np.array(np.nonzero(mask)).T
        coords_f = -1 + 2 * ((coords + 0.5) / self.resolution)
        tree = cKDTree(sampled_pts)
        idxs = tree.query(coords_f, k=1)[1]
        occ_labels = torch.from_numpy(labels[idxs]).int()
        occ_coords = torch.from_numpy(coords).int()
        return occ_coords, occ_labels





def custom_collate_fn(batch):
    # Stack points and voxel grids
    points  = torch.stack([b["points"] for b in batch], dim=0)
    folders = [sample["folder"] for sample in batch]
    # vox_bin = torch.stack([b["vox_bin"] for b in batch], dim=0)

    all_coords = []
    all_feats  = []
    all_centers= []
    all_radii  = []
    batch_layer_ranges = [0]
    offset = 0

    for sample in batch:
        for layer in sample["layers"]:
            coords = layer["coords"]       # [Ki,3]
            feats  = layer["feats"]        # [Ki,1]
            center = layer["center"]       # [3]
            radius = layer["radius"]       # scalar

            N = coords.size(0)
            batch_idx_col = torch.full((N, 1), offset, dtype=coords.dtype)

            all_coords.append(torch.cat([batch_idx_col, coords], dim=1))
            all_feats.append(feats)
            all_centers.append(center)
            all_radii.append(radius)

            offset += 1
        batch_layer_ranges.append(offset)

    # Concatenate lists into tensors
    occ_coords = torch.cat(all_coords, dim=0)
    occ_feats  = torch.cat(all_feats, dim=0)
    batch_layer_ranges = torch.tensor(batch_layer_ranges, dtype=torch.long)
    occ_bbox_centers = torch.stack(all_centers, dim=0)  # [B',3]
    occ_bbox_radii  = torch.stack(all_radii, dim=0).unsqueeze(-1)  # [B',1]

    return {
        'points': points,
        'occ_coords': occ_coords,
        'occ_feats': occ_feats,
        'batch_layer_ranges': batch_layer_ranges,
        "folders": folders,            # 数据所属文件夹（调试用）
        'occ_bbox_centers': occ_bbox_centers,
        'occ_bbox_radii': occ_bbox_radii,
    }


class VoxelRayDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, resolution=256, batch_size=4, num_workers=4,data_name=None):
        super().__init__()
        self.root_dir = root_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = data_name

    def setup(self, stage=None):
        self.dataset = VoxelDataset(self.root_dir, resolution=self.resolution, data_name=self.data_name)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=True,
        )
    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=True,
        )