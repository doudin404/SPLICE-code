import torch
import math
import torch.nn.functional as F

def split_and_pad_features(features, segments):
    """
    Args:
        features (Tensor): shape (seq_len, feature_dim)
        segments (Tensor or list): 1D, segment boundaries, e.g. [0, 3, 5, 8]
    Returns:
        padded_features (Tensor): (num_segments, max_len, feature_dim)
        pad_mask (Tensor): (num_segments, max_len), 1 for valid, 0 for pad
    """
    if isinstance(segments, torch.Tensor):
        segments = segments.tolist()
    num_segments = len(segments) - 1
    feature_dim = features.size(-1)
    seg_lens = [segments[i+1] - segments[i] for i in range(num_segments)]
    max_len = max(seg_lens)
    padded_features = features.new_zeros((num_segments, max_len, feature_dim))
    pad_mask = torch.zeros((num_segments, max_len), dtype=torch.bool, device=features.device)

    for i in range(num_segments):
        start, end = segments[i], segments[i+1]
        seg_feat = features[start:end]  # (seg_len, feature_dim)
        seg_len = end - start
        padded_features[i, :seg_len] = seg_feat
        pad_mask[i, :seg_len] = 1

    return padded_features, pad_mask

def normalize_coords(coords_int: torch.Tensor, shape) -> torch.Tensor:
    """
    将整型体素坐标归一化到 [-1,1] 区间（体素中心在内部），支持各向异性网格。
    Args:
        coords_int: [N,3] 或 [N,4] 张量，其中最后三维为 (z,y,x)
        shape: int 或 (D,H,W)，网格尺寸
    Returns:
        pts: [N,3] 归一化坐标张量
    """
    # 统一为 (D,H,W)
    if isinstance(shape, int):
        D = H = W = shape
    else:
        D, H, W = shape
    # 提取 z,y,x 并转 float
    zyx = coords_int[..., -3:].float()
    z, y, x = zyx[:, 0], zyx[:, 1], zyx[:, 2]
    # 平移 0.5，再除以尺寸，映射到 [-1,1]，保证中心不在边缘
    return torch.stack([
        (z + 0.5) / D * 2 - 1,
        (y + 0.5) / H * 2 - 1,
        (x + 0.5) / W * 2 - 1,
    ], dim=1)


def sample_sdf_coords(
    point_clouds: torch.Tensor,
    sampling_strategy: str = 'random',
    sample_num: int = 2000,
    surface_bandwidth: float = 0.1
) -> torch.Tensor:
    """
    根据指定策略生成用于 SDF 计算的查询坐标。

    参数:
      point_clouds (B, M, 3): 已归一化到 [-1,1] 的点云。
      sampling_strategy (str): 'random' 或 'near_surface'。
      random_sample_num (int): random 策略时的样本数。
      surface_sample_num (int): near_surface 策略时的样本数。
      surface_bandwidth (float): near_surface 噪声标准差。

    返回:
      coords (B, N, 3): 采样点坐标，N = sample_num
    """
    B, M, _ = point_clouds.shape
    device = point_clouds.device

    if sampling_strategy == 'random':
        # 均匀采样到 [-1.2,1.2]
        coords = torch.rand(B, sample_num, 3, device=device) * 2.4 - 1.2

    elif sampling_strategy == 'near_surface':
        # 在点云附近采样并加高斯噪声
        idx = torch.randint(0, M, (B, sample_num), device=device)
        base_pts = torch.gather(
            point_clouds,
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, surface_sample_num, 3)
        noise = torch.randn_like(base_pts) * surface_bandwidth
        coords = (base_pts + noise).clamp(-1.1, 1.1)

    else:
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

    return coords


def compute_sdf_from_samples(
    psr_volumes: torch.Tensor,
    point_clouds: torch.Tensor,
    coords: torch.Tensor
) -> torch.Tensor:
    """
    给定查询坐标，基于 PSR 隐式场和点云计算 SDF。

    参数:
      psr_volumes (B, D, H, W): PSR 隐式场体素网格。
      point_clouds (B, M, 3): 对应的点云（[-1,1]）。
      coords (B, N, 3): 查询点坐标（[-1.1,1.1]）。

    返回:
      sdf (B, N, 1): 每个查询点对应的 Signed Distance。
    """
    B, D, H, W = psr_volumes.shape
    device = psr_volumes.device
    psr_volumes = psr_volumes.permute(0, 3, 2, 1).contiguous()

    # 1) 最近点距离
    # dists = torch.cdist(coords, point_clouds)      # (B, N, M)
    # min_dists, _ = dists.min(dim=2, keepdim=True)  # (B, N, 1)

    # 2) trilinear 插值取符号
    psr_in = psr_volumes.unsqueeze(1)  # (B,1,D,H,W)
    # grid_sample 要求 grid 形状 (B, N, 1, 1, 3)
    grid = coords.view(B, coords.shape[1], 1, 1, 3)
    samples = F.grid_sample(
        psr_in, grid, align_corners=True,
        mode='bilinear', padding_mode='border'
    )  # (B,1,N,1,1)
    psr_vals = samples.view(B, coords.shape[1], 1)
    signs = torch.sign(psr_vals)

    return signs

def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion(s) to rotation matrix(ces).

    Args:
        quaternion: tensor of shape (..., 4) with components (w, x, y, z).

    Returns:
        Rotation matrix tensor of shape (..., 3, 3).
    """
    # Normalize quaternion to unit length
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q.unbind(dim=-1)

    # Compute products
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    # Rotation matrix elements
    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz

    # Stack into matrix
    mat = torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )
    return mat


def gaussian_nll_loss(
    mu: torch.Tensor,
    rotation_matrix: torch.Tensor,
    eigenvalues: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the negative log-likelihood loss for points under a Gaussian distribution
    parameterized by mean, rotation (eigenvectors), and eigenvalues (diagonal covariance).

    Args:
        mu: tensor of shape (..., d), the mean vector.
        rotation_matrix: tensor of shape (..., d, d), orthonormal eigenvector matrix Q.
        eigenvalues: tensor of shape (..., d), positive eigenvalues \lambda_j.
        points: tensor of shape (..., N, d), set of N points to evaluate.

    Returns:
        Scalar or tensor of shape (...) representing the NLL loss.
    """
    # Ensure shapes broadcast: leading dims of mu, rotation_matrix, eigenvalues must match points
    *batch_dims, N, d = points.shape
    # Center points
    centered = points - mu.unsqueeze(-2)  # (..., N, d)
    # Project to principal axes: y = Q^T (x - mu)
    Qt = rotation_matrix.transpose(-2, -1)  # (..., d, d)
    y = torch.matmul(centered, Qt)  # (..., N, d)

    # Mahalanobis terms: sum(y^2 / lambda)
    inv_eig = 1.0 / eigenvalues.clamp(min=1e-12)
    m_dist = (y.pow(2) * inv_eig.unsqueeze(-2)).sum(dim=-1)  # (..., N)

    # Log-determinant term
    log_det = torch.log(eigenvalues.clamp(min=1e-12)).sum(dim=-1)  # (...)

    # Compute NLL: 0.5 * [N * (d*log(2*pi) + log_det) + sum_i m_i]
    const_term = d * math.log(2 * math.pi)
    term1 = N * (const_term + log_det)  # (...)
    term2 = m_dist.sum(dim=-1)  # (...)
    nll = 0.5 * (term1 + term2)
    nll = nll / N
    return nll

def ellipsoid_vertices(mu: torch.Tensor, eigenvectors: torch.Tensor, eigenvalues: torch.Tensor, num_points: int = 6, radius=1.75) -> torch.Tensor:
    """
    生成椭球表面六个顶点（主轴正负方向上的点）。

    Args:
        mu: (..., 3) 椭球中心
        eigenvectors: (..., 3, 3) 主轴方向（每列为单位向量）
        eigenvalues: (..., 3) 主轴长度（半轴长度）
        num_points: 忽略，仅为接口兼容

    Returns:
        points: (..., 6, 3) 椭球表面六个顶点
    """
    # 单位球六个顶点
    base = torch.eye(3, device=mu.device)  # (3,3)
    dirs = torch.cat([base, -base], dim=0)  # (6,3)
    dirs = dirs*radius
    # 缩放到椭球表面
    scaled = dirs * eigenvalues.unsqueeze(-2)  # (..., 6, 3)
    # 旋转
    rotated = torch.matmul(scaled, eigenvectors.transpose(-2, -1))  # (..., 6, 3)
    # 平移
    points = rotated + mu.unsqueeze(-2)
    return points
def recover_ellipsoid(vertices: torch.Tensor,
                      radius: float = 1.75,
                      eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    根据椭球的六个顶点反推出椭球参数：中心 mu、主轴方向矩阵 eigenvectors 和半轴长度 eigenvalues。
    假设传入的 vertices 形状为 (..., 6, 3)，且与原函数生成时的顺序保持一致：
      前 3 个点分别是主轴 +X、+Y、+Z 方向上的顶点，后 3 个点分别是对应的 -X、-Y、-Z 方向顶点。
    radius 表示原始单位球顶点在进入缩放前的半径（默认为 1.75）。
    eps 用于在计算单位向量时给范数做下限，避免除以零导致 NaN。

    返回：
        mu: (..., 3) 椭球中心
        eigenvectors: (..., 3, 3) 主轴方向矩阵，每一列是对应的单位主轴向量
        eigenvalues: (..., 3) 半轴长度（对应 X、Y、Z 三条主轴）
    """
    # 1. 先计算中心 mu：六个点的平均值
    #    由于正负方向对称，六个顶点相加再除以 6，正负偏移抵消后，剩下的就是中心
    mu = vertices.mean(dim=-2)  # (..., 3)

    # 2. 计算每个顶点相对于中心的偏移向量
    offsets = vertices - mu.unsqueeze(-2)  # (..., 6, 3)

    # 3. 对应的正向偏移向量就在 offsets[..., 0:3, :]
    pos_offsets = offsets[..., :3, :]  # (..., 3, 3)

    # 4. 分别计算三个正向偏移向量的长度，以及对应的单位方向
    #    norm_i = radius * eigenvalues_i，所以 eigenvalues_i = norm_i / radius
    #    单位方向 u_i = pos_offsets_i / max(norm_i, eps)；用 eps 防止除以 0
    #    最终把这三个 u_i 作为列向量拼成矩阵 (..., 3, 3)
    #    把 (norm_i / radius) 堆成 (..., 3)
    # 先把 pos_offsets 拆成三个单独的向量
    v1 = pos_offsets[..., 0, :]  # (..., 3)
    v2 = pos_offsets[..., 1, :]  # (..., 3)
    v3 = pos_offsets[..., 2, :]  # (..., 3)

    # 计算它们的范数
    n1 = torch.linalg.norm(v1, dim=-1)  # (...,)
    n2 = torch.linalg.norm(v2, dim=-1)  # (...,)
    n3 = torch.linalg.norm(v3, dim=-1)  # (...,)

    # 在除法前做 clamp，保证范数至少为 eps，避免 0/0 导致 NaN
    n1_clamped = n1.clamp(min=eps)      # (...,)
    n2_clamped = n2.clamp(min=eps)      # (...,)
    n3_clamped = n3.clamp(min=eps)      # (...,)

    # 半轴长度
    e1 = n1 / radius  # (...,)
    e2 = n2 / radius  # (...,)
    e3 = n3 / radius  # (...,)

    # 对应的单位方向
    u1 = v1 / n1_clamped.unsqueeze(-1)  # (..., 3)
    u2 = v2 / n2_clamped.unsqueeze(-1)  # (..., 3)
    u3 = v3 / n3_clamped.unsqueeze(-1)  # (..., 3)

    # 把三个单位方向按列拼成矩阵：每个单位方向都是 (...,3)，最后拼成 (...,3,3)
    eigenvectors = torch.stack([u1, u2, u3], dim=-1)  # (..., 3, 3)

    # 半轴长度按对应顺序堆成 (...,3)
    eigenvalues = torch.stack([e1, e2, e3], dim=-1)  # (..., 3)

    return mu, eigenvectors, eigenvalues

def sample_affines(
    batch_size: int,
    trans_range=(-0.5, 0.5),
    trans_prob=1,
    scale_range=(-0.5, 0.5),
    scale_prob=1,
    rot_range=(-math.pi/4, math.pi/4),
    rot_prob=0.2,
    device=None,
) -> torch.Tensor:
    """
    Vectorized sampling of random affine transforms including translation, scaling, and rotation.

    Args:
        batch_size: Number of transforms to sample.
        trans_range: Tuple (min, max) for uniform translation along each axis.
        trans_prob: Probability of applying translation; otherwise zero translation.
        scale_range: Tuple (min, max) for uniform log-scale to sample; final scale = exp(log_scale).
        scale_prob: Probability of applying scaling; otherwise scale=1.
        rot_range: Tuple (min, max) for uniform sampling of rotation angle around random axis.
        rot_prob: Probability of applying rotation; otherwise angle=0 (identity rotation).
        device: torch device or string (e.g. 'cuda'); defaults to CPU.

    Returns:
        Tensor of shape (batch_size, 4, 4) of affine matrices.
    """
    device = torch.device(device) if device is not None else torch.device('cpu')
    B = batch_size

    # Translation sampling
    mask_t = torch.rand(B, device=device) < trans_prob
    trans_vals = torch.empty(B, 3, device=device).uniform_(trans_range[0], trans_range[1])
    translations = torch.where(mask_t.view(B, 1), trans_vals, torch.zeros_like(trans_vals))

    # Scale sampling (via log-scale)
    mask_s = torch.rand(B, device=device) < scale_prob
    log_scales = torch.empty(B, device=device).uniform_(scale_range[0], scale_range[1])
    log_scales = torch.where(mask_s, log_scales, torch.zeros_like(log_scales))
    scales = torch.exp(log_scales)

    # Rotation sampling (axis-angle via Rodrigues)
    mask_r = torch.rand(B, device=device) < rot_prob
    angles = torch.empty(B, device=device).uniform_(rot_range[0], rot_range[1])
    angles = torch.where(mask_r, angles, torch.zeros_like(angles))

    # Random unit axes (unused if angle=0)
    axes = torch.randn(B, 3, device=device)
    axes = axes / axes.norm(dim=1, keepdim=True).clamp(min=1e-12)

    # Build skew-symmetric K matrix for each axis
    K = torch.zeros(B, 3, 3, device=device)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] =  axes[:, 1]
    K[:, 1, 0] =  axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] =  axes[:, 0]

    # Identity matrices
    I3 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)

    # Rodrigues' rotation: R = I*cos + (1-cos)*(axis*axis^T) + sin*K
    cos_t = angles.cos().view(B, 1, 1)
    sin_t = angles.sin().view(B, 1, 1)
    outer = axes.unsqueeze(2) * axes.unsqueeze(1)
    R = I3 * cos_t + outer * (1 - cos_t) + sin_t * K

    # Scaling matrices S = scale * I
    S = scales.view(B, 1, 1) * I3

    # Combined linear part: apply scaling then rotation
    M = torch.bmm(R, S)  # (B, 3, 3)

    # Assemble into 4x4 affine
    affines = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    affines[:, :3, :3] = M
    affines[:, :3, 3] = translations

    return affines

@torch.no_grad
def query_part_ids(part_vol: torch.Tensor,
                                 coords: torch.Tensor) -> torch.Tensor:
    """
    part_vol: FloatTensor 或者 IntTensor，形状 (N, 1, D, H, W)
              单通道里存的是 “part_id + 1” 或者 0
    coords  : LongTensor，形状 (N, P, 3)，存的是 (z, y, x) 的体素索引

    返回
    ----
    part_ids: LongTensor，形状 (N, P)，
              值为 [-1, batch_max_parts-1]，其中 -1 表示体素外或者不在部件内，
              否则 [0, batch_max_parts-1] 表示对应的部件编号。
    """
    N, D, H, W = part_vol.shape
    _, P, _       = coords.shape

    # 先把 part_vol 转成 float，用来给 grid_sample
    vol_f = part_vol.float().permute(0, 3, 2, 1).contiguous().unsqueeze(1)  # (N, 1, D, H, W)

    # # 下面把 (z,y,x) 从索引映射到 [-1, 1] 区间
    # # grid_sample 要求输入的最后一维坐标顺序是 (x, y, z)
    # z = coords[..., 0].float() * (2.0 / (D - 1)) - 1.0
    # y = coords[..., 1].float() * (2.0 / (H - 1)) - 1.0
    # x = coords[..., 2].float() * (2.0 / (W - 1)) - 1.0

    # 把 (x,y,z) 拼成 (N, P, 1, 1, 3)
    grid = coords.view(N, P, 1, 1, 3)
    

    # 最近邻采样，padding_mode='zeros' 保证体素外的采样点会得到 0.0
    sampled = F.grid_sample(vol_f,
                            grid,
                            mode='nearest',
                            padding_mode='zeros',
                            align_corners=True)
    # sampled 形状 (N, 1, P, 1, 1) -> 去掉多余维度变成 (N, P)
    sampled = sampled.view(N, P).long()  # 这里转回 LongTensor，里面的值 ∈ [0, batch_max_parts]

    # 现在 sampled[n, p] == 0 说明 “体素外或不在任何部件内”，
    # 否则 sampled[n, p] − 1 就是真正的 part_id。
    part_ids = sampled - 1  # 让索引回到 [-1, batch_max_parts-1] 范围

    return part_ids


def multi_ellipsoid_occupancy(points: torch.Tensor,
                              mus: torch.Tensor,
                              eigenvectors: torch.Tensor,
                              eigenvalues: torch.Tensor,
                              radius: float = 1.75) -> torch.Tensor:
    """
    给定多只椭球和若干采样点，判断每个点是否落在“任意一只椭球的 radius 倍马氏距离”以内。
    如果存在某个椭球使得：
        ((R_j^T @ (p_i - mu_j)) / eigenvalues_j).norm() <= radius
    则结果为 1，否则为 0。

    Args:
        points:        (N, 3)  或 (..., 3)  所有要检测的点
        mus:           (E, 3)  E 只椭球的中心
        eigenvectors:  (E, 3, 3)   每只椭球的旋转矩阵，列向量为主轴单位方向
        eigenvalues:   (E, 3)     每只椭球对应的三个半轴长度
        radius:        标量，表示马氏距离阈值，默认 1.75

    Returns:
        Tensor of shape (N,)（dtype=torch.uint8），第 i 项为 1 表示 points[i]
        距离任意一只椭球的马氏距离 ≤ radius，否则为 0。
    """
    # —— 1. 将输入 reshape 成 (N, 3)；E 只椭球对应 (E, 3), (E, 3, 3), (E, 3) —— #
    pts = points.view(-1, 3)          # (N, 3)
    mus_flat = mus.view(-1, 3)        # (E, 3)
    evects = eigenvectors.view(-1, 3, 3)  # (E, 3, 3)
    evalues = eigenvalues.view(-1, 3)     # (E, 3)

    N = pts.shape[0]
    E = mus_flat.shape[0]

    # —— 2. 计算 offset(i, j, :) = pts[i, :] - mus[j, :] —— #
    #    令 pts.unsqueeze(1) 形状 (N, 1, 3)，mus_flat.unsqueeze(0) 形状 (1, E, 3)
    #    广播后为 (N, E, 3)
    offset = pts.unsqueeze(1) - mus_flat.unsqueeze(0)   # (N, E, 3)

    # —— 3. 将 offset 投影到各个椭球的主轴坐标系 —— #
    # 方法 A：einsum
    #    local[i, j, k] = sum_{m=0..2} offset[i,j,m] * evects[j,m,k]
    #    也就是 offset 的第三维与 evects[j] 的“行”做内积，最后得到 (N, E, 3)
    local = torch.einsum('nej,ejk->nek', offset, evects)  # (N, E, 3)

    # —— 4. 对局部坐标除以半轴长度，得到归一化的坐标 (N, E, 3) —— #
    normalized = local / evalues.unsqueeze(0)   # (N, E, 3)

    # —— 5. 计算每个 (点 i, 椭球 j) 的平方范数 d^2 = Σ_k (normalized[i,j,k]^2) —— #
    squared_norm = (normalized ** 2).sum(dim=-1)  # (N, E)

    # —— 6. 判断 d^2 <= radius^2 —— #
    mask = squared_norm <= (radius ** 2)         # (N, E) 布尔张量

    # —— 7. 如果某个 j 使得 mask[i,j] 为 True，则 points[i] 被标为 1 —— #
    inside_any = mask.any(dim=1)                 # (N,) 布尔张量

    # —— 8. 转成 0/1 整数并返回 —— #
    return 1-inside_any.to(torch.uint8)            # (N,) dtype=torch.uint8