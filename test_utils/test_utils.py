import data_process.point_sampler
from shape_process.mesh_sampler import MeshSampler
from shape_process.gm_transformer import GMTransformer
from net.model_func import split_point_by_gmm
import torch
import trimesh
import os
import re
import pickle

def read_gmm_from_data(data: str):
    """
    从输入数据中读取 GMM 参数。
    第一行三个为一组，第二行三个为一组，第三行九个为一组。
    最后一行不用管，将每个 gm 数据以 [3, 3, 9] 的顺序连接起来。
    如果某个 gm 全为 0，则舍弃这个 gm。
    返回所有 gm 数据堆叠在一起的张量。
    """
    lines = data.strip().split('\n')
    if len(lines) < 3:
        raise ValueError("输入数据不完整，至少需要前三行")

    mu_values = torch.tensor(list(map(float, lines[0].split())))
    eigen_values = torch.tensor(list(map(float, lines[1].split())))
    p_values = torch.tensor(list(map(float, lines[2].split())))
    exist_values = torch.tensor(list(map(float, lines[3].split())))

    num_gms = len(mu_values) // 3
    gm_list = []
    
    for i in range(num_gms):
        mu = mu_values[i * 3:(i + 1) * 3]
        eigen = eigen_values[i * 3:(i + 1) * 3]
        p = p_values[i * 9:(i + 1) * 9]
        

        if exist_values[i]==0:
            continue  # 如果所有值都为0，跳过此 gm

        gm = torch.cat((mu, eigen, p))
        gm_list.append(gm)
    
    if not gm_list:
        return torch.empty((0, 15))  # 如果没有有效的 gm，返回空张量

    return torch.stack(gm_list)

class TransformPoint:
    def __init__(self, before, after):
        """
        初始化变换器类, 使用仿射变换矩阵进行变换。
        每个 GMM 对应一个变换矩阵。
        """
        self.combined_transformers = [self._construct_combined_transform(gm_before, gm_after) for gm_before, gm_after in zip(before, after)]

    def _construct_combined_transform(self, gm_before, gm_after):
        """
        根据给定的 GMM 参数构造联合仿射变换矩阵，避免精度误差。
        变换过程是先应用缩放，再应用旋转，最后应用位移。
        """
        # Construct the inverse transformation matrix (after -> standard)
        translation_after = gm_after[:3]
        scale_after = gm_after[3:6]
        rotation_matrix_after = gm_after[6:].reshape(3, 3)
        inverse_scale_after = 1.0 / scale_after
        inverse_rotation_matrix_after = rotation_matrix_after
        inverse_transform_matrix_after = torch.eye(4)
        inverse_transform_matrix_after[:3, :3] = torch.diag(inverse_scale_after) @ inverse_rotation_matrix_after
        inverse_translation_after = -inverse_transform_matrix_after[:3, :3] @ translation_after
        inverse_transform_matrix_after[:3, 3] = inverse_translation_after

        # Construct the forward transformation matrix (standard -> before)
        translation_before = gm_before[:3]
        scale_before = gm_before[3:6]
        rotation_matrix_before = gm_before[6:].reshape(3, 3).t()
        transform_matrix_before = torch.eye(4)
        transform_matrix_before[:3, :3] = rotation_matrix_before @ torch.diag(scale_before)
        transform_matrix_before[:3, 3] = translation_before

        # Combined transformation matrix: from after -> before
        combined_transform_matrix = transform_matrix_before @ inverse_transform_matrix_after

        return combined_transform_matrix

    def transform(self, points, part_index):
        """
        将多个点从变换后坐标转换为变换前坐标。
        part_index 用于指定当前点对应的 GMM 部件序号。
        支持一次性输入多个点。
        """
        points_homo = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)  # 转换为齐次坐标
        points = points_homo @ self.combined_transformers[part_index].t()
        return points[:, :3]  # 返回原始坐标系下的 3D 点

def get_labels_for_quest_points(candidate_points: torch.Tensor, labels: torch.Tensor, quest_points: torch.Tensor, res: int = 128, candidate_part_indices: torch.Tensor = None, part_index=None) -> torch.Tensor:
    """
    获取 quest_points 对应的 labels。以每个部件所属的点为单位批量获取。
    如果某个点超过了体素范围, 则 label = False.
    如果点不和体素网格对齐，首先将点对齐到最近的格点上。
    candidate_part_indices 用于指定每个 candidate_point 对应的部件。
    quest_part_indices 用于指定每个 quest_point 对应的部件。
    """
    # 获取体素网格的边界
    bbox_min = torch.tensor([-1.0, -1.0, -1.0])
    bbox_max = torch.tensor([1.0, 1.0, 1.0])

    # 计算步长（格点间的距离）
    voxel_size = (bbox_max - bbox_min) / (res - 1)

    aligned_labels = torch.zeros(len(quest_points), dtype=torch.bool)

    # if quest_part_indices is None:
    #     quest_part_indices = torch.zeros(len(quest_points), dtype=torch.long)

    if candidate_part_indices is None:
        candidate_part_indices = torch.zeros(len(candidate_points), dtype=torch.long)

    # 对齐 quest_points 到最近的格点上
    normalized_points = (quest_points - bbox_min) / voxel_size
    aligned_indices = torch.round(normalized_points).to(torch.long)

    # 检查哪些点在边界内
    is_within_bounds = torch.all((aligned_indices >= 0) & (aligned_indices < res), dim=1)

    # 对于在边界内的点，计算其在 labels 中的位置
    valid_indices = aligned_indices[is_within_bounds]
    linear_indices = valid_indices[:, 0] * (res ** 2) + valid_indices[:, 1] * res + valid_indices[:, 2]

    # 获取对应的标签，并检查部件编号是否匹配
    # valid_quest_part_indices = quest_part_indices[is_within_bounds]
    corresponding_candidate_part_indices = candidate_part_indices[linear_indices]
    aligned_labels[is_within_bounds] = (labels[linear_indices] == 1) & (part_index == corresponding_candidate_part_indices)

    return aligned_labels

def generate_voxel_grid_and_labels(mesh: trimesh.Trimesh, res: int = 128) -> tuple:
    """
    生成一个体素网格,并判断每个点是否在形状内部
    """
    # 生成体素网格，范围为 [-1, 1]
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.linspace(-1.0, 1.0, res),
        torch.linspace(-1.0, 1.0, res),
        torch.linspace(-1.0, 1.0, res),
    )

    candidate_points = torch.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], dim=1)

    # 判断每个点是否在形状内部
    sampler = MeshSampler(mesh)
    labels = torch.tensor(sampler.get_labels(candidate_points.numpy()))

    # 保存所有点和对应的标签
    # print("所有点采样完成，共有点数：", len(candidate_points))
    # print("内部点数：", torch.sum(labels))
    return candidate_points, labels

def process_gmm_and_points(gmm, points) -> torch.Tensor:
    """
    输入 gmm 和 点云，得到 part_labels 并返回。
    points 是需要处理的点云数据。
    """
    gms = gmm  # Assuming gmm is already a tensor

    # Extract mu, phi, and eigen from gms
    mu = gms[..., :3]  # Shape: (N, 3)
    eigen = gms[..., 3:6]  # Shape: (N, 3)
    p = gms[..., 6:].view(-1, 3, 3)  # Shape: (N, 3, 3)

    # Create a mask where phi == 0
    phi_zero_mask = ~(gms[..., 3:6].any(-1))  # Shape: (N,)

    # Define standard transformations
    standard_mu = torch.ones_like(mu[0])*100 # Shape: (3,)
    standard_p = torch.eye(3, device=mu.device, dtype=mu.dtype)  # Shape: (3, 3)
    standard_eigen = torch.ones_like(eigen[0])*0.1

    # Update mu and eigen where phi == 0
    mu[phi_zero_mask] = standard_mu
    p[phi_zero_mask] = standard_p
    eigen[phi_zero_mask] = standard_eigen
    phi=torch.where(phi_zero_mask,-torch.inf,1)

    # Compute p: 1 where phi != 0, 0 where phi == 0
    #p = (~phi_zero_mask).to(torch.float)

    # Prepare gmms tuple
    gmms = (
        mu[None, None, ...],             # Shape: (1, 1, N, 3)
        p[None, None, ...],          # Shape: (1, 1, N, 3, 3)
        phi[None, None, ...],
        eigen[None, None, ...],              # Shape: (1, 1, N)
    )

    # Compute part_labels
    part_labels = split_point_by_gmm(points, gmms)
    return part_labels

def make_target(shape, before, after, res=128) -> tuple:
    """
    输入,形状数据obj,编辑前gmm,编辑后gmm.
    主要工作是先按照 before 和 after 划分体素网格, part_labels 代表某个点对应于 gms[i] 这个部件。
    对于 after 中的每一个点, 用 TransformPoint 获取变形前的位置并用 get_labels_for_quest_points 查询。
    返回体素点阵和每个点对应的布尔值标签。
    """
    # 加载形状数据
    mesh = trimesh.load(shape)

    # 使用新的函数生成体素网格和标签
    candidate_points, labels = generate_voxel_grid_and_labels(mesh, res)

    # 使用 GMM 和点云计算 part_labels
    part_labels_before = process_gmm_and_points(before, candidate_points)
    part_labels_after = process_gmm_and_points(after, candidate_points)

    # 创建 TransformPoint 实例，避免重复构造 GMTransformer
    transform_point_instance = TransformPoint(before, after)

    # 初始化结果标签张量
    result_labels = torch.zeros(part_labels_after.shape, dtype=torch.bool)

    # 以部件为单位处理点
    unique_parts = torch.unique(part_labels_after)
    for part_index in unique_parts:
        if part_index == -1:
            continue  # 跳过不属于任何部件的点

        # # 获取属于当前部件的点
        # mask = part_labels_after == part_index
        # points_to_transform = candidate_points[mask]

        # 获取变形前的位置
        transformed_points = transform_point_instance.transform(candidate_points, part_index=part_index)

        # 查询变形前的点是否在形状之中并属于对应部件
        is_inside = get_labels_for_quest_points(candidate_points, labels, transformed_points, res, candidate_part_indices=part_labels_before, part_index=part_index)
        #valid_points = (is_inside & (part_labels_before == part_index))

        # 更新结果标签张量，只要某个点一次为 true 就为 true
        result_labels = result_labels | is_inside
        #if part_index == 3:break


    return candidate_points, result_labels

def process_and_evaluate(folder_path: str, res=128):
    """
    从指定文件夹中寻找形如 ({name}_before.obj, {name}_after.obj, {name}_before.txt, {name}_after.txt) 的一组文件,
    读取后生成编辑前文件的体素网格,生成体素结果,并使用 gmm 信息获取编辑后的预期结果,和实际的结果比对,得到正确率。
    """
    files = os.listdir(folder_path)
    pattern = re.compile(r'^(.*)_before\.obj$')
    processed_files = set()

    for file in files:
        match = pattern.match(file)
        if match:
            name = match.group(1)
            before_obj_file = os.path.join(folder_path, f"{name}_before.obj")
            after_obj_file = os.path.join(folder_path, f"{name}_after.obj")
            before_file = os.path.join(folder_path, f"{name}_before.txt")
            after_file = os.path.join(folder_path, f"{name}_after.txt")

            if os.path.exists(before_obj_file) and os.path.exists(after_obj_file) and os.path.exists(before_file) and os.path.exists(after_file):
                with open(before_file, 'r') as bf:
                    before_data = bf.read()
                with open(after_file, 'r') as af:
                    after_data = af.read()

                before_gmm = read_gmm_from_data(before_data)
                after_gmm = read_gmm_from_data(after_data)

                # 生成编辑前的体素网格和标签
                candidate_points, result_labels = make_target(before_obj_file, before_gmm, after_gmm, res)

                # 加载编辑后的形状并生成标签
                after_mesh = trimesh.load(after_obj_file)
                sampler = MeshSampler(after_mesh)
                actual_labels = torch.tensor(sampler.get_labels(candidate_points.numpy()))

                from shape_process.visualizationManager import VisualizationManager
                vm=VisualizationManager([1,2])
                vm.add_points(candidate_points,actual_labels,subplot=[0,0],colors_to_show=["blue"])
                vm.add_points(candidate_points,result_labels,subplot=[0,1],colors_to_show=["blue"])
                vm.show()

                # 计算准确率
                differences = torch.sum(result_labels != actual_labels)
                gt_positive_count = torch.sum(actual_labels == 1)
                accuracy = 1.0 - (differences.item() / gt_positive_count.item() if gt_positive_count > 0 else 1.0)

                print(f"{name}: 准确率 = {accuracy:.2f}")

                processed_files.add(name)

    print(f"共处理了 {len(processed_files)} 组文件")

# 示例用法
# process_and_evaluate("path/to/your/folder", res=128)
