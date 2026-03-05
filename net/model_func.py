import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from shape_process.gm_transformer import GMTransformer
import random
import h5py
import os
import pickle
import trimesh

def get_gm_support(gm, x):
    dim = x.shape[-1]
    mu, p, phi, eigen = gm
    sigma_det = eigen.prod(-1)
    eigen_inv = 1 / eigen
    sigma_inverse = torch.matmul(
        p.transpose(3, 4), p * eigen_inv[:, :, :, :, None]
    ).squeeze(1)
    phi = torch.softmax(phi, dim=2)
    const_1 = phi / torch.sqrt((2 * np.pi) ** dim * sigma_det)
    distance = x[:, :, None, :] - mu
    mahalanobis_distance = -0.5 * torch.einsum(
        "bngd,bgdc,bngc->bng", distance, sigma_inverse, distance
    )
    const_2, _ = mahalanobis_distance.max(dim=2)  # for numeric stability
    mahalanobis_distance -= const_2[:, :, None]
    support = const_1 * torch.exp(mahalanobis_distance)
    return support, const_2


def gm_log_likelihood_loss(
    gms, x, get_supports: bool = False, mask=None, reduction: str = "mean"
):

    batch_size, num_points, dim = x.shape
    support, const = get_gm_support(gms, x)
    probs = torch.log(support.sum(dim=2)) + const
    # t=test(gms, x)
    if mask is not None:
        probs = probs.masked_select(mask=mask.flatten())
    if reduction == "none":
        likelihood = probs.sum(-1)
        loss = -likelihood / num_points
    else:
        likelihood = probs.sum()
        loss = -likelihood / (probs.shape[0] * probs.shape[1])
    if get_supports:
        return loss, support
    return loss

def flatten_gmm(gmm):
    b, gp, g, _ = gmm[0].shape
    mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmm]
    p = p.reshape(*p.shape[:2], -1)
    z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2)
    return z_gmm


def gm_loss(gm_params, x):
    batch_n = gm_params.shape[0]

    # 解析 gm_params
    mu = gm_params[:, 0:3]  # 前三个是均值
    eigen_sqrt = gm_params[:, 3:6]  # 接下来三个是特征值平方根
    p = gm_params[:, 6:].view(batch_n, 3, 3)  # 后九个是特征向量，重塑并转置

    # 计算协方差矩阵的特征值（平方）
    eigen = eigen_sqrt**2

    # 权重默认为均等
    phi = (
        torch.ones((batch_n, 1), device=gm_params.device) / 1
    )  # 由于只有一个高斯成分，权重为1

    # 封装 gms
    gms = (
        mu[:, None, None, :],
        p[:, None, None, :],
        phi[:, None],
        eigen[:, None, None, :],
    )

    # 计算损失
    loss = gm_log_likelihood_loss(gms, x, reduction="none")
    return loss

def split_mesh_by_gmm(mesh, gmm):
    faces_split = {}
    if type(mesh)==trimesh.Trimesh:
        vs, faces = torch.tensor(mesh.vertices) ,torch.tensor(mesh.faces)
    else:
        vs, faces = mesh
    vs_mid_faces = vs[faces].mean(1)
    gmm=[torch.tensor(arr) for arr in gmm]
    gmm[3]=gmm[3]**2
    _, supports = gm_log_likelihood_loss(gmm, vs_mid_faces.unsqueeze(0), get_supports=True)
    supports = supports[0]
    label = supports.argmax(1)
    for i in range(gmm[1].shape[2]):
        select = label.eq(i)
        if select.any():
            faces_split[i] = faces[select]
        else:
            faces_split[i] = None
    return faces_split

def split_point_by_gmm(points, gmm):
    gmm = list(gmm)
    gmm[3] = gmm[3] ** 2  # 协方差平方，保持与原始代码一致
    
    # 计算每个点的支持度
    _, supports = gm_log_likelihood_loss(gmm, points[None, :], get_supports=True)
    supports = supports[0]
    
    # 为每个点分配一个标签
    label = supports.argmax(1)
    
    return label


def generate_random_transformation(probability=1.0, device="cpu"):
    # 通过调用 generate_k_random_transformations 生成单个变换
    rotations, translations, scales = generate_k_random_transformations(
        1, probability=probability, device=device
    )
    return rotations, translations, scales


def generate_k_random_transformations(k, probability=1.0, device="cpu"):
    # Determine whether to generate random transformations or zero transformations based on the provided probability
    # random_selection = np.random.random(k) < probability

    # Generate random translations or zero translations
    translations = np.where(
        np.random.random((k, 1)) < probability,
        np.random.uniform(-0.3, 0.3, (k, 3)),
        np.zeros((k, 3)),
    )

    # Generate random scales or unity scales
    scales = np.where(
        np.random.random(k) < probability,
        np.random.uniform(0.7, 1.3, k),
        np.ones(k),
    )

    # Generate random rotations or identity matrices
    random_selection = np.random.random(k) < probability
    rotations = R.random(k)
    rotation_matrices = np.array(
        [
            r.as_matrix() if random_selection[i] else np.eye(3)
            for i, r in enumerate(rotations)
        ]
    )

    # Convert to PyTorch tensors
    translations_tensor = torch.tensor(translations, dtype=torch.float32, device=device)
    scales_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
    rotation_matrices_tensor = torch.tensor(
        rotation_matrices, dtype=torch.float32, device=device
    )

    return rotation_matrices_tensor, translations_tensor, scales_tensor


def extract_gmm_parameters(gmms):
    translations = gmms[:, :3]
    scales = gmms[:, 3:6]
    u1s = gmms[:, 6:9]
    u2s = gmms[:, 9:12]
    u3s = gmms[:, 12:15]

    rotations = torch.stack((u1s, u2s, u3s), dim=-1)

    return rotations, translations, scales


def apply_transformation_to_points(points, rotations, translations, scales):

    # 应用变换
    points_transformed = points * scales.unsqueeze(1).unsqueeze(1)
    points_transformed = torch.einsum("kij,kni->knj", rotations, points_transformed)
    points_transformed += translations.unsqueeze(1)

    return points_transformed


def apply_transformation_to_gmms(gmms, rotations, translations, scales):
    gmm_is_zero = torch.all(gmms == 0, dim=-1, keepdim=True)

    # 分解 gmm 组件
    ci = gmms[..., :3]
    si = gmms[..., 3:6]
    u1i, u2i, u3i = gmms[..., 6:9], gmms[..., 9:12], gmms[..., 12:15]

    # 批量应用变换到 gmm 组件
    ci_transformed = ci * scales.view(-1, 1, 1)
    ci_transformed = torch.einsum("kij,kni->knj", rotations, ci_transformed)
    ci_transformed += translations.view(-1, 1, 3)

    u1i_transformed = torch.einsum("kij,kni->knj", rotations, u1i)
    u2i_transformed = torch.einsum("kij,kni->knj", rotations, u2i)
    u3i_transformed = torch.einsum("kij,kni->knj", rotations, u3i)
    si_transformed = torch.einsum("k,knj->knj", scales, si)

    # 重新组合 gmms
    gmms_transformed = torch.cat(
        [
            ci_transformed,
            si_transformed,
            u1i_transformed,
            u2i_transformed,
            u3i_transformed,
        ],
        axis=-1,
    )
    gmms_transformed = torch.where(gmm_is_zero, gmms, gmms_transformed)

    return gmms_transformed


def random_transform(gmms, mask=None, x=None,):
    batch_num, num_parts, _ = gmms.shape
    device = gmms.device
    rotations, translations, scales = generate_k_random_transformations(
        batch_num, probability=1, device=device
    )
    trans_gmms = apply_transformation_to_gmms(gmms, rotations, translations, scales)

    if mask is not None:
        trans_gmms[mask] = 0

    if x is not None:
        trans_x = apply_transformation_to_points(x, rotations, translations, scales)

    return trans_gmms, trans_x


def _make_assist_points(gms, points, mask):
    # 创建一个与 points 大小相同的全0张量
    assist_points = torch.zeros_like(points)

    min_distances = torch.full((points.size(0),), float('inf'), dtype=torch.float32, device=points.device)

    # 生成一个随机的顺序来访问 gms 的项目
    gm_indices = list(range(gms.size(0)))
    random.shuffle(gm_indices)

    # 遍历每一个高斯分布 gm
    for idx in gm_indices:
        if mask[idx]:
            continue
        gm = gms[idx]

        # 对每一个 point 执行 transform(gm, point) 获取 transformed_point
        transformed_points = GMTransformer(gm).get_inverse_gm().transform(points)

        # 将 transformed_points 中位于 [-2, 2] 范围内的点填充到 assist_points 中
        points_mask = (transformed_points >= -2) & (transformed_points <= 2)
        points_mask = points_mask.all(dim=1)  # 只考虑所有维度都在 [-2, 2] 范围内的点

        # 计算每个点与中心的距离
        points_distance = torch.sum((transformed_points**2), -1)
        
        # 仅在新的距离小于当前记录的最小距离时才更新
        update_mask = points_mask & (points_distance < min_distances)

        min_distances[update_mask] = points_distance[update_mask]

        assist_points[points_mask] = transformed_points[points_mask]

    return assist_points


def make_assist_points(gms, train_points, mask):
    batch_size = gms.size(0)
    assist_points = []

    for i in range(batch_size):
        assist_points_batch = _make_assist_points(gms[i], train_points[i], mask[i])
        assist_points.append(assist_points_batch)

    # 将列表中的每一项（形状为 [points_num, 3]）拼接成形状为 [batch, points_num, 3] 的张量
    assist_points = torch.stack(assist_points, dim=0)

    return assist_points

# def _make_attn_labels(gms, points, mask):
#     # 创建一个与 points 大小相同的标签张量，初始值为 -1
#     attn_labels = torch.full((points.size(0),), -1, dtype=torch.long, device=points.device)
    
#     # 创建一个张量来存储每个点的最小距离，初始值为一个大的数值
#     min_distances = torch.full((points.size(0),), float('inf'), dtype=torch.float32, device=points.device)

#     # 生成一个随机的顺序来访问 gms 的项目
#     gm_indices = list(range(gms.size(0)))
#     random.shuffle(gm_indices)

#     # 遍历每一个高斯分布 gm
#     for idx in gm_indices:
#         if mask[idx]:
#             continue
#         gm = gms[idx]

#         # 对每一个 point 执行 transform(gm, point) 获取 transformed_point
#         transformed_points = GMTransformer(gm).get_inverse_gm().transform(points)

#         # 将 transformed_points 中位于 [-2, 2] 范围内的点对应的标签设置为当前 gm 的索引
#         points_mask = (transformed_points >= -2) & (transformed_points <= 2)
#         points_mask = points_mask.all(dim=1)  # 只考虑所有维度都在 [-2, 2] 范围内的点

#         # 计算每个点与中心的距离
#         points_distance = torch.sum((transformed_points**2), -1)
        
#         # 仅在新的距离小于当前记录的最小距离时才更新标签和最小距离
#         update_mask = points_mask & (points_distance < min_distances)

#         # 更新 attn_labels 和 min_distances
#         attn_labels[update_mask] = idx
#         min_distances[update_mask] = points_distance[update_mask]

#     from shape_process.visualizationManager import VisualizationManager
#     vm = VisualizationManager()
#     vm.add_axes_at_origin()
#     for idx in range(gms.size(0)):
#         if mask[idx] or not (attn_labels==idx).any():
#             continue
        
#         vm.add_points(points[attn_labels==idx])
#     # vm.add_obb(list(gm[3:6]))
#     vm.show()

#     return attn_labels

def _make_attn_labels(gms, points, mask):
    batch_n = gms.shape[0]

    # 仅选择未被 mask 掩盖的 GMs
    unmasked_indices = mask == 0  # 找出未被掩盖的索引
    unmasked_gms = gms[unmasked_indices]

    mu = unmasked_gms[..., 0:3]  # 前三个是均值
    eigen_sqrt = unmasked_gms[..., 3:6]  # 接下来三个是特征值平方根
    p = unmasked_gms[..., 6:].view(-1, 3, 3)  # 后九个是特征向量，重塑并转置

    eigen = eigen_sqrt ** 2

    # 计算 phi，仅使用未被掩盖的部分
    phi = (1 - mask.float())[unmasked_indices] / mask.sum(-1)

    # 重新构建新的 GMs，用于计算损失
    gms_processed = (
        mu[None, None, :],
        p[None, None, :],
        phi[None, None, :],
        eigen[None, None, :],
    )

    # 计算损失，只针对未被掩盖的 GMs
    _, supports = gm_log_likelihood_loss(gms_processed, points[None, :], get_supports=True)
    labels = supports[0].argmax(1)

    
    return labels

def make_attn_labels(gms, train_points, mask):
    batch_size = gms.size(0)
    attn_labels = []

    for i in range(batch_size):
        assist_attn_labels = _make_attn_labels(gms[i], train_points[i], mask[i])
        attn_labels.append(assist_attn_labels)

    # 将列表中的每一项（形状为 [points_num, 3]）拼接成形状为 [batch, points_num, 3] 的张量
    attn_labels = torch.stack(attn_labels, dim=0)

    return attn_labels

def min_mse_loss(points, target):
    # Reshape points to [len, n, 3]
    n = points.size(1) // 3
    points = points.reshape(-1, n, 3)

    # Expand target to match points shape [len, n, 3]
    target_expanded = target.unsqueeze(1).expand(-1, n, -1)

    # Calculate MSE
    mse = torch.mean((points - target_expanded) ** 2, dim=2)

    # Find minimum MSE across points
    min_mse = torch.min(mse, dim=1)[0]

    return min_mse


def sdf_loss(y_sdf, y_sdf_gt):
    d = 0.1
    # 对 y_sdf 和 y_sdf_gt 进行截断
    y_sdf_clamped = y_sdf / d
    y_sdf_gt_clamped = torch.clamp(y_sdf_gt, min=-d, max=d) / d

    # 找到超出范围的元素
    sdf_out_of_bounds = (y_sdf.abs() > d) & (y_sdf_gt.abs() > d)

    # 判断是否在同一个方向出界
    same_direction = torch.sign(y_sdf) == torch.sign(y_sdf_gt)

    # 只对未超出范围或不同方向的元素计算损失
    valid_mask = ~sdf_out_of_bounds | ~same_direction
    valid_mask = valid_mask.float()

    # 计算有效位置的损失
    mse_loss = F.mse_loss(
        y_sdf_clamped * valid_mask, y_sdf_gt_clamped * valid_mask, reduction="sum"
    )
    num_valid = valid_mask.sum()

    # 防止除以零的情况
    if num_valid > 0:
        sdf_loss = mse_loss / num_valid
    else:
        sdf_loss = torch.tensor(0.0, requires_grad=True)

    return sdf_loss

def occ_loss(y_occ, y_sdf_gt):
    # 计算 y_occ_gt: y_sdf_gt > 0 的地方为 1, 否则为 0
    y_occ_gt = (y_sdf_gt > 0).float()

    # 找到 y_occ 绝对值大于2的位置
    mask_abs_gt_2 = torch.abs(y_occ) > 2
    
    # 找到 y_occ 和 y_sdf_gt 符号一致的位置
    sign_match = ((y_occ > 0) & (y_sdf_gt > 0)) | ((y_occ < 0) & (y_sdf_gt < 0))
    
    # 创建屏蔽条件的最终掩码
    mask = ~(mask_abs_gt_2 & sign_match)  # 取反，只有符合条件的才被屏蔽为False

    # 计算二进制交叉熵损失，只对mask中为True的位置进行损失计算
    bce_loss = F.binary_cross_entropy_with_logits(y_occ[mask], y_occ_gt[mask], reduction="mean")
    
    return bce_loss

def sdf_accuracy(y_sdf, y_sdf_gt):
    # 计算 y_sdf 和 y_sdf_gt 的符号
    y_sdf_sign = torch.sign(y_sdf)
    y_sdf_gt_sign = torch.sign(y_sdf_gt)
    
    # 筛选出 y_sdf_gt 中的负号位置
    negative_gt_mask = (y_sdf_gt_sign == -1)
    
    # 判断这些位置上 y_sdf 的符号是否也是负号
    correct_negative_predictions = (y_sdf_sign[negative_gt_mask] == -1).float()
    
    # 计算负号预测正确的比例
    negative_accuracy = correct_negative_predictions.mean()
    
    return negative_accuracy

def save_low_accuracy_shapes(batch, output_file):
    """
    Compares the accuracy of each shape in the batch and saves the IDs of shapes
    with accuracy below 0.7 to a specified file.
    
    Parameters:
    - batch (dict): A dictionary containing 'id', 'sample_train_sdfs', and 'sample_train_labels'.
    - output_file (str): Path to the file where IDs with low accuracy will be saved.
    """
    low_accuracy_ids = []
    
    for shape_id, y_sdf, y_sdf_gt in zip(batch['id'], batch['sample_train_sdfs'], batch['sample_train_labels']):
        # Compute accuracy for the current shape
        accuracy = sdf_accuracy(y_sdf, y_sdf_gt)
        
        # Check if accuracy is below 0.7
        if accuracy < 0.7:
            low_accuracy_ids.append(shape_id)
    
    # Save low accuracy IDs to the output file
    with open(output_file, "w") as f:
        for shape_id in low_accuracy_ids:
            f.write(shape_id + "\n")
    
    print(f"Low accuracy shape IDs saved to {output_file}")

def torch_no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
        return result

    return wrapper

def save_h5(data_to_extract, extract_dir):
    all_gms = torch.stack([item[1] for item in data_to_extract]).cpu().numpy()
    all_shape_z = torch.stack([item[2] for item in data_to_extract]).cpu().numpy()

    with h5py.File(os.path.join(extract_dir, "extracted_gms.h5"), "w") as f:
        f.create_dataset("gms", data=all_gms)

    with h5py.File(os.path.join(extract_dir, "extracted_shape_z.h5"), "w") as f:
        f.create_dataset("shape_z", data=all_shape_z)

def save_txt_pkl(data_to_extract, extract_dir):
    for id_, gms, shape_z in data_to_extract:
        gms_path = os.path.join(extract_dir, f"{id_}.txt")
        gms_data = [
            gms[..., :3].flatten(),
            gms[..., 3:6].flatten(),
            gms[..., 6:15].flatten(),
            (shape_z != 0).any(-1).to(torch.int).flatten()
        ]
        with open(gms_path, "w") as gms_file:
            for line in gms_data:
                gms_file.write(" ".join(map(str, line.tolist())) + "\n")
        
        shape_z_path = os.path.join(extract_dir, f"{id_}.pkl")
        with open(shape_z_path, "wb") as pkl_file:
            pickle.dump(shape_z.cpu(), pkl_file)

def save_ids_txt(data_to_extract, extract_dir):
    ids_path = os.path.join(extract_dir, "ids.txt")
    with open(ids_path, "w") as ids_file:
        for id_, _, _ in data_to_extract:
            ids_file.write(str(id_) + "\n")
