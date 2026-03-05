import math
import numpy as np
import trimesh
import torch
import igl
from shape_process.gm_transformer import GMTransformer



def igl_prepare(func):
    """
    装饰器函数,用于在调用函数之前将输入参数转换为适当的格式。

    - 对于 torch.Tensor 类型的参数,转换为 numpy.ndarray。
    - 确保 numpy.ndarray 参数的数据类型为 float64,并且内存布局为行主序。
    - 对于 trimesh.Trimesh 类型的参数,提取其顶点和面,转换为一个包含顶点和面的元组。

    参数:
    - func (function): 需要装饰的函数。

    返回值:
    - function: 装饰后的函数。
    """

    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.detach().cpu().numpy()
            if isinstance(arg, np.ndarray):
                arg = np.ascontiguousarray(arg, dtype=np.float64)
            if isinstance(arg, trimesh.Trimesh):
                vertices = arg.vertices
                faces = arg.faces
                arg = (vertices, faces)
            new_args.append(arg)
        result = func(*new_args, **kwargs)


        return result

    return wrapper


@igl_prepare
def get_fast_inside_outside(mesh, points):
    """
    使用 fast winding number 方法计算点集是否在网格内部或外部。

    参数:
    - mesh (tuple): 一个包含网格顶点和面的元组。
    - points (numpy.ndarray): 要检查的点集。

    返回值:
    - numpy.ndarray: 布尔数组,指示每个点是否在网格外部。
    """
    batch_size = 1000000
    labels = []
    num_batch = (points.shape[0] + batch_size - 1) // batch_size
    for i in range((points.shape[0] + batch_size - 1) // batch_size):
        if i == num_batch - 1:
            pts_in = points[batch_size * i:]
        else:
            pts_in = points[batch_size * i: batch_size * (i + 1)]
        w = igl.fast_winding_number_for_meshes(*mesh, pts_in)
        w = np.greater(w, 0.9)
        labels.append(w)
    return np.concatenate(labels, axis=0)


class MeshSampler:
    """
    网格采样器类,用于在网格上采样点并计算点的标签。

    参数:
    - mesh (trimesh.Trimesh): 目标网格。
    - global_scale (float): 全局缩放比例。默认为 1。
    """

    def __init__(self, mesh, global_scale=1):
        self.mesh = mesh.copy()
        self.global_scale = global_scale
        self.mesh_bb = self._compute_mesh_bounds()
        self.labels = []
        self.points = []

    def __len__(self):
        """
        返回标签的数量。

        返回值:
        - int: 标签的数量。
        """
        return self.labels.shape[1] if self.labels is not None else 0

    def __getitem__(self, item):
        """
        获取指定索引的点和标签。

        参数:
        - item (int): 索引。

        返回值:
        - tuple: 包含点和标签的元组。
        """
        return self.points[:, item], self.labels[:, item]

    def get_labels(self, points):
        """
        计算点集的标签。

        参数:
        - points (numpy.ndarray): 要计算标签的点集。

        返回值:
        - numpy.ndarray: 点集的标签。
        """
        return get_fast_inside_outside(self.mesh, points)

    def sample_points(self, num, strategy, compute_labels=True):
        """
        根据指定的策略采样点,并可选地计算这些点的标签。

        参数:
        - num (int): 需要采样的总点数。
        - strategy (tuple): 一个包含采样策略和其比例的元组。(策略名称,所占比例)。
            - "near_0": 从表面附近采样点,不添加噪声（请确保 compute_labels=False）。
            - "near_a": 从表面附近采样点,并添加较小的噪声（标准差为0.007）。
            - "near_b": 从表面附近采样点,并添加较大的噪声（标准差为0.02）。
            - "uniform": 在整个空间中均匀采样点。
            - "inside": 在整个空间中均匀采样点。但仅保留内部点,重复采样以达到目标点数.
        - compute_labels (bool): 是否为采样的点计算标签。默认为 True,计算标签；如果设置为 False,则不计算标签。

        返回值:
        - 如果 compute_labels 为 True,则返回一个包含所有采样点和它们的标签的元组 (all_points, all_labels)。
        - 如果 compute_labels 为 False,则仅返回所有采样点 all_points。
        """
        points_list = []
        labels_list = [] if compute_labels else None
        total_samples = 0

        for strategy_name, proportion in strategy:
            num_samples = (
                int(num * proportion) if proportion > 0 else num - total_samples
            )

            if strategy_name in ["near_a", "near_b", "near_0"]:
                noise_scale = (
                    0.007
                    if strategy_name == "near_a"
                    else (0.02 if strategy_name == "near_b" else 0)
                )
                points = self._sample_surface_near_points(num_samples, noise_scale)
            elif strategy_name == "uniform":
                points = (
                    np.random.rand(num_samples, 3).astype(np.float32) * 2 - 1
                ) * self.global_scale
            elif strategy_name == "inside":
                points = self._sample_inside_points(num_samples)
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy_name}")

            total_samples += num_samples
            points_list.append(points)

            if compute_labels:
                labels = self.get_labels(points)
                labels_list.append(labels)

            if total_samples >= num or proportion < 0:
                break

        all_points = np.vstack(points_list)
        all_labels = np.concatenate(labels_list) if compute_labels else None

        return (all_points, all_labels) if compute_labels else all_points

    def _sample_surface_near_points(self, num, scale):
        """
        从网格表面附近采样点,并根据需要添加噪声。

        参数:
        - num (int): 需要采样的点数。
        - scale (float): 噪声的标准差。如果为 0,则不添加噪声。

        返回值:
        - numpy.ndarray: 采样的点集。
        """
        near_points, _ = trimesh.sample.sample_surface(self.mesh, num)
        if scale != 0:
            near_points = (
                near_points.astype(np.float32)
                + np.random.randn(*near_points.shape).astype(np.float32) * scale
            )
        return near_points

    def _sample_inside_points(self, num):
        """
        在整个空间中均匀采样点，并仅保留内部点，重复采样以达到目标点数。

        参数:
        - num (int): 需要采样的点数。

        返回值:
        - numpy.ndarray: 采样的点集。
        """
        points_list = []
        while len(points_list) < num:
            points = (np.random.rand(num*10, 3).astype(np.float32) * 2 - 1) * self.global_scale
            inside_mask = self.get_labels(points)
            points_list.append(points[inside_mask])
        all_points = np.vstack(points_list)[:num]
        return all_points

    def _compute_mesh_bounds(self):
        """
        计算网格的边界。

        返回值:
        - tuple: 包含网格最小边界和边界范围的元组。
        """
        bounds = self.mesh.bounds
        return bounds[0], bounds[1] - bounds[0]
