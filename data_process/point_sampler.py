import os
import trimesh
import numpy as np
import torch
from scipy.spatial import cKDTree
from types import SimpleNamespace
from typing import Generator, List, Any, Tuple
from shape_process.mesh_sampler import MeshSampler
from shape_process.gm_transformer import GMTransformer
from shape_process.visualizationManager import VisualizationManager
from net.model_func import split_point_by_gmm


class InternalPointSamplePipeline:
    """
    这个类用于从形状中采样点云,并写入数据的data.sample(SimpleNamespace)
    需要的点有:
    - 内部采样,以128(也可在初始化时设置参数)的密度在空间内均匀选点，并使用 get_labels 判断其内外，仅保留内部点
    """

    def __init__(
        self,
        input_generator: Generator[SimpleNamespace, None, None],
        resolution: int = 128,
        **cfg,
    ):
        self.input_generator = input_generator
        self.resolution = resolution

    def __iter__(self):
        for data in self.input_generator:
            if not hasattr(data, "sample"):
                data.sample = SimpleNamespace()
            data.sample.internal_sample = self._sample_internal(data.mesh)

            yield data

    def _sample_internal(self, mesh: trimesh.Trimesh) -> np.ndarray:
        bbox_min = mesh.bounds[0]
        bbox_max = mesh.bounds[1]

        bbox_size = bbox_max - bbox_min

        num_samples_per_axis = np.ceil(self.resolution * (bbox_size) / 2.0).astype(int)

        grid_x, grid_y, grid_z = np.meshgrid(
            np.linspace(bbox_min[0], bbox_max[0], num_samples_per_axis[0]).astype(
                np.float32
            ),
            np.linspace(bbox_min[1], bbox_max[1], num_samples_per_axis[1]).astype(
                np.float32
            ),
            np.linspace(bbox_min[2], bbox_max[2], num_samples_per_axis[2]).astype(
                np.float32
            ),
        )

        candidate_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

        sampler = MeshSampler(mesh)
        labels = sampler.get_labels(candidate_points)

        internal_points = candidate_points[labels == 1]

        return internal_points


class SDFPipeline:
    """
    这个类用于计算采样点到形状表面的最短距离，并根据label加上正负号，形成SDF。
    输入为经过PointSamplePipeline处理的数据。
    """

    def __init__(self, input_generator: Generator[SimpleNamespace, None, None]):
        self.input_generator = input_generator

    def __iter__(self):
        for data in self.input_generator:
            data.sample.near_a.sdf = self._compute_sdf(data.sample.near_a, data.mesh)
            data.sample.near_b.sdf = self._compute_sdf(data.sample.near_b, data.mesh)
            data.sample.global_sample.sdf = self._compute_sdf(
                data.sample.global_sample, data.mesh
            )

            yield data

    def _compute_sdf(
        self, sample_data: SimpleNamespace, mesh: trimesh.Trimesh
    ) -> np.ndarray:
        points = sample_data.points
        labels = sample_data.labels
        tree = cKDTree(mesh.vertices)

        # 计算每个采样点到最近表面点的距离
        distances, _ = tree.query(points)

        # 根据标签确定距离的符号，标签true表示在表面内部，false表示在表面外部
        sdf_values = distances * np.where(labels, -1, 1)

        return sdf_values


class PointSamplePipeline:
    """
    这个类用于从形状中采样点云,并写入数据的data.sample(SimpleNamespace)
    需要的点有:
    - 从表面附近采样,有near a 和 near b两种
    - 全局采样,从全体形状的包围盒随机采样
    """

    def __init__(
        self,
        input_generator: Generator[SimpleNamespace, None, None],
        sample_num: int,
        **cfg,
    ):
        self.input_generator = input_generator
        self.sample_num = sample_num

    def __iter__(self):
        for data in self.input_generator:
            if not hasattr(data, "sample"):
                data.sample = SimpleNamespace()
            data.sample.near_a = self._sample_near_a(data.mesh)
            data.sample.near_b = self._sample_near_b(data.mesh)
            data.sample.global_sample = self._sample_global(data.mesh)

            yield data

    def _sample_near_a(self, mesh: trimesh.Trimesh) -> np.ndarray:
        sampler = MeshSampler(mesh)
        points, labels = sampler.sample_points(
            num=self.sample_num, strategy=[("near_a", 1.0)], compute_labels=True
        )
        return SimpleNamespace(points=points, labels=labels)

    def _sample_near_b(self, mesh: trimesh.Trimesh) -> np.ndarray:
        sampler = MeshSampler(mesh)
        points, labels = sampler.sample_points(
            num=self.sample_num, strategy=[("near_b", 1.0)], compute_labels=True
        )
        return SimpleNamespace(points=points, labels=labels)

    def _sample_global(self, mesh: trimesh.Trimesh) -> np.ndarray:
        sampler = MeshSampler(mesh)
        points, labels = sampler.sample_points(
            num=self.sample_num, strategy=[("uniform", 1.0)], compute_labels=True
        )
        return SimpleNamespace(points=points, labels=labels)


class PartPointSamplePipeline:
    """
    这个类用于从形状中获取部件相关的信息.

    - 用data.parts.obb截取data.sample.internal_sample中的点，算出每个parts的中心(重心)data.parts.center。
    - 使用中心位置和data.parts.aabb计算以data.parts.center为中心最小的包含部件的立方体边长data.parts.patch。
    - 在每个patch上内部采样，以64(可选)的分辨率采样点并用get_labels筛选出内部点，保存在data.parts.voxel中。
      data.parts.voxel的内容是(每个内部点的整数坐标)和(浮点坐标的范围，用于还原点的位置)。
      同时,每个obb内的点也会被分别保存.
    """

    def __init__(
        self,
        input_generator: Generator[SimpleNamespace, None, None],
        obj_folder: str,
        tree_name: str,
        sample_num: int,
        min_part_rate: int,
        part_resolution: int = 64,
        **cfg,
    ):
        self.input_generator = input_generator
        self.obj_folder = obj_folder
        self.tree_name = tree_name
        self.sample_num = sample_num
        self.resolution = part_resolution
        self.min_part_rate = min_part_rate

    def __iter__(self):
        for data in self.input_generator:
            # Calculate part centers and patches
            self._calculate_part_centers_and_patches(data)
            if any(len(x) < 100 for x in data.parts.obb_voxel):
                print(data.id,f"obb:",*(len(x) for x in data.parts.obb_voxel))
                continue
            # Sample points in each patch
            self._sample_voxels_in_patches(data)
            
            if any(sum(x) < 100 for x in data.parts.voxel_labels):
                print(data.id,f"aabb:",*(sum(x) for x in data.parts.voxel_labels))
                continue

            yield data

    def _calculate_part_centers_and_patches(self, data: SimpleNamespace):
        data.parts.center = []
        data.parts.patch_size = []
        data.parts.obb_voxel = []

        for i in range(len(data.parts.obb.extents)):
            obb_transform = data.parts.obb.transform[i]
            obb_extents = data.parts.obb.extents[i]

            obb_center = obb_transform[:3, 3]
            obb_axes = obb_transform[:3, :3]

            translation = obb_center
            scale = np.array([1, 1, 1])  # 无缩放
            rotation = obb_axes.T.flatten()

            gm = np.concatenate([translation, scale, rotation])
            transformer = GMTransformer(gm=gm).get_inverse_gm()

            # Transform the internal sample points to the OBB-aligned space
            transformed_points = transformer.transform(data.sample.internal_sample)

            # 计算OBB的实际范围在对齐空间内的长度
            obb_min = -obb_extents / 2
            obb_max = obb_extents / 2

            # Determine points within the OBB in the OBB-aligned space
            obb_filter = np.all(
                (transformed_points >= obb_min) & (transformed_points <= obb_max),
                axis=1,
            )
            obb_points = data.sample.internal_sample[obb_filter]
            if len(obb_points) < self.min_part_rate:
                data.parts.obb_voxel.append(obb_points)
                break

            center = np.mean(obb_points, axis=0)

            center_obb_points = obb_points - center

            distances = np.concatenate(
                [center_obb_points.max(axis=0), -center_obb_points.min(axis=0)]
            )

            # 找到最大距离并乘以2得到包含AABB的立方体边长
            max_distance = np.max(distances)
            patch_length = max_distance * 2*1.2

            data.parts.center.append(center)
            data.parts.patch_size.append(patch_length)
            data.parts.obb_voxel.append(obb_points)

        data.parts.center = np.array(data.parts.center)
        data.parts.patch_size = np.array(data.parts.patch_size)


    def _sample_voxels_in_patches(self, data: SimpleNamespace):
        data.parts.voxel_indices = []
        data.voxel_ranges = []
        data.parts.voxel_labels = []
        all_voxel_points = []
        lengths = []  # 用于记录每一段的长度

        for i in range(len(data.parts.obb.extents)):
            obb_transform = data.parts.obb.transform[i]
            obb_extents = data.parts.obb.extents[i]
            center, patch_length = data.parts.center[i], data.parts.patch_size[i]
            min_bound = center - patch_length / 2
            max_bound = center + patch_length / 2
            voxel_grid, voxel_ranges, voxel_indices = self._create_voxel_grid(
                min_bound, max_bound, self.resolution
            )

            voxel_points = voxel_grid.reshape(-1, 3)
            all_voxel_points.append(voxel_points)
            lengths.append(len(voxel_points))  # 记录当前段的点数

        # 将所有需要计算的点合并为一个大的数组
        all_voxel_points = np.vstack(all_voxel_points)

        # 一次性获取所有点的 labels
        sampler = MeshSampler(data.mesh)
        all_labels = sampler.get_labels(all_voxel_points)

        start_idx = 0
        for i in range(len(data.parts.obb.extents)):
            # 获取当前段的点和标签
            voxel_points = all_voxel_points[start_idx : start_idx + lengths[i]]
            labels = all_labels[start_idx : start_idx + lengths[i]]
            start_idx += lengths[i]  # 更新下一个段的起始索引

            # 找到内部的点和索引
            internal_voxels = voxel_points[labels == 1]
            internal_indices = voxel_indices.reshape(-1, 3)[labels == 1]

            obb_transform = data.parts.obb.transform[i]
            obb_extents = data.parts.obb.extents[i]
            voxel_grid_local = (internal_voxels - obb_transform[:3, 3]) @ obb_transform[
                :3, :3
            ]

            half_extents = obb_extents / 2
            inside_obb = (
                (voxel_grid_local[:, 0] >= -half_extents[0])
                & (voxel_grid_local[:, 0] <= half_extents[0])
                & (voxel_grid_local[:, 1] >= -half_extents[1])
                & (voxel_grid_local[:, 1] <= half_extents[1])
                & (voxel_grid_local[:, 2] >= -half_extents[2])
                & (voxel_grid_local[:, 2] <= half_extents[2])
            )

            data.parts.voxel_labels.append(inside_obb)
            data.parts.voxel_indices.append(internal_indices)

    def _create_voxel_grid(
        self, min_bound: np.ndarray, max_bound: np.ndarray, resolution: int
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        x = np.linspace(min_bound[0], max_bound[0], resolution)
        y = np.linspace(min_bound[1], max_bound[1], resolution)
        z = np.linspace(min_bound[2], max_bound[2], resolution)
        voxel_grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)

        x_indices = np.arange(resolution, dtype=np.int32)
        y_indices = np.arange(resolution, dtype=np.int32)
        z_indices = np.arange(resolution, dtype=np.int32)
        voxel_indices = np.stack(
            np.meshgrid(x_indices, y_indices, z_indices, indexing="ij"), axis=-1
        )

        return voxel_grid, (min_bound, max_bound), voxel_indices


class TestPointSamplePipeline:

    def __init__(
        self,
        input_generator: Generator[SimpleNamespace, None, None],
        **cfg,
    ):
        self.input_generator = input_generator

    def __iter__(self):
        for data in self.input_generator:
            
            data.sample.internal_sample
            data.parts
            data.parts_after
            gms=[]
            for i in range(len(data.parts.obb.extents)):
                obb_transform = data.parts.obb.transform[i]
                obb_extents = data.parts.obb.extents[i]

                obb_center = obb_transform[:3, 3]
                obb_axes = obb_transform[:3, :3]

                translation = obb_center
                scale = obb_extents
                rotation = obb_axes.T.flatten()

                gms.append(np.concatenate([translation, scale, rotation]))
            gms=torch.tensor(np.stack(gms,0))#mu, p, phi, eigen
            gmms=gms[None,None,...,:3],gms[None,None,...,6:].view(1,1,-1,3,3),torch.ones_like(gms[None,None,...,0]),gms[None,None,...,3:6]
            t=split_point_by_gmm(torch.tensor(data.sample.internal_sample), gmms)

            from shape_process.visualizationManager import VisualizationManager
            vm=VisualizationManager()
            for i in t:
                if t[i] is not None:
                    vm.add_points(t[i])
                    #vm.add_gmm(gms[i])
                    vm.add_obb((SimpleNamespace(transform=data.parts.obb.transform[i],extents=data.parts.obb.extents[i])))
            vm.show()
            
            yield data
