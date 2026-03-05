import numpy as np
import torch
import trimesh
import functools

# from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R


class GMTransformer:
    def __init__(
        self,
        gm=None,
        rotations=None,
        translations=None,
        scales=None,
        do_normalize=False,
        reverse=False,
    ):
        self.do_normalize = do_normalize
        self.reverse = reverse

        if gm is not None:
            if not isinstance(gm, (np.ndarray, torch.Tensor)):
                rotations=gm[1]
                translations=gm[0]
                scales=gm[2]
                self.extract_gm_components_from_components(rotations, translations, scales)
                self.use_torch = isinstance(rotations, torch.Tensor)
                if self.use_torch:
                    self.device = rotations.device
            else:
                if gm.shape[-1] ==10:
                    gm=gquat_to_gm(gm)
                if gm.shape[-1] != 15:
                    raise ValueError("gm input must have exactly 15 parameters.")

                self.gm = gm
                self.use_torch = isinstance(gm, torch.Tensor)
                if self.use_torch:
                    self.device = gm.device
                self.extract_gm_components_from_gm()
        else:
            if not all(
                isinstance(x, (np.ndarray, torch.Tensor))
                for x in [rotations, translations, scales]
            ):
                raise TypeError(
                    "Rotations, translations, and scales must all be PyTorch tensors."
                )

            self.use_torch = True
            self.device = rotations.device
            self.extract_gm_components_from_components(rotations, translations, scales)

    def extract_gm_components_from_gm(self):
        components = self.gm[:15]
        self.ci = components[:3]
        self.λi = components[3:6]
        self.u1i, self.u2i, self.u3i = (
            components[6:9],
            components[9:12],
            components[12:15],
        )
        self.rotation = self.create_rotation_matrix()

    def extract_gm_components_from_components(self, rotations, translations, scales):
        self.rotation = rotations
        self.ci = translations
        self.λi = scales

    def create_rotation_matrix(self):
        # Create rotation matrix from eigenvectors
        rotation = (
            torch.stack((self.u1i, self.u2i, self.u3i), dim=1)
            if self.use_torch
            else np.column_stack((self.u1i, self.u2i, self.u3i))
        )
        if self.use_torch:
            rotation = rotation.to(self.device)
        return rotation

    def normalize_vectors(self, vectors, original_magnitudes):
        if self.use_torch:
            magnitudes = torch.norm(vectors, dim=0, keepdim=True).clamp(min=1e-8)
        else:
            magnitudes = np.linalg.norm(vectors, axis=0, keepdims=True)
            np.clip(magnitudes, 1e-8, None, out=magnitudes)

        normalized_vectors = vectors / magnitudes
        return normalized_vectors * original_magnitudes

    def calculate_magnitudes(self, vectors, keepdims=False):
        if self.use_torch:
            return torch.norm(vectors, dim=0, keepdim=keepdims).to(self.device)
        else:
            return np.linalg.norm(vectors, axis=0, keepdims=keepdims)

    def scale_vectors(self, vectors):
        return self.λi[:, None] * vectors

    def rotate_vectors(self, vectors):
        return self.rotation @ vectors

    def translate_vectors(self, vectors):
        return vectors + self.ci[:, None]

    def transform(self, points):
        if self.use_torch and not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=self.device)
        elif not self.use_torch and not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)

        points = points.T

        if self.do_normalize:
            magnitudes = self.calculate_magnitudes(points, keepdims=False)
        if not self.reverse:
            points = self.scale_vectors(points)
            points = self.rotate_vectors(points)
            points = self.translate_vectors(points)
        else:
            points = self.translate_vectors(points)
            points = self.rotate_vectors(points)
            points = self.scale_vectors(points)

        if self.do_normalize:
            points = self.normalize_vectors(points, magnitudes)

        return points.T  # Convert back to (n_points, n_dims) shape

    def get_inverse_gm(self):
        # Ensure scale factors are not zero
        λi = (
            torch.clamp(self.λi, min=1e-8)
            if self.use_torch
            else np.clip(self.λi, 1e-8, None)
        )
        inverse_λi = 1.0 / λi
        inverse_ci = -self.ci
        # inverse_rotation = self.rotation.T 直接展平才是对的
        new_gm = (
            torch.hstack([inverse_ci, inverse_λi, self.rotation.flatten()])
            if self.use_torch
            else np.hstack([inverse_ci, inverse_λi, self.rotation.flatten()])
        )
        return GMTransformer(new_gm, reverse=not self.reverse)

    def get_transform_normals_gm(self):
        new_gm = torch.zeros_like(self.gm) if self.use_torch else np.zeros_like(self.gm)
        new_gm[3:] = self.gm[3:]  # Keep shapes and orientation the same
        return GMTransformer(new_gm, do_normalize=True)

    def get_uniform_scale_gm(self):
        if self.use_torch:
            max_λ = torch.max(self.λi)
            uniform_λ = torch.tensor([max_λ, max_λ, max_λ], device=self.device)
        else:
            max_λ = np.max(self.λi)
            uniform_λ = np.array([max_λ, max_λ, max_λ])
        new_gm = (
            torch.hstack([self.ci, uniform_λ, self.u1i, self.u2i, self.u3i]).to(
                self.device
            )
            if self.use_torch
            else np.hstack([self.ci, uniform_λ, self.u1i, self.u2i, self.u3i])
        )
        return GMTransformer(new_gm)

    def get_no_rotation_gm(self):
        new_gm = (
            torch.hstack(
                [
                    self.ci,
                    self.λi,
                    torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], device=self.device),
                ]
            )
            if self.use_torch
            else np.hstack([self.ci, self.λi, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])])
        )
        return GMTransformer(new_gm)
    
def fix_gquat(gq):
    return torch.cat((gq[...,:3],gq[...,3:6],gq[...,-4:]/gq[...,-4:].norm(2,-1,keepdim=True)),dim=-1)
def gquat_to_gm(gq):
    return torch.cat((gq[...,:3],torch.exp(gq[...,3:6]),quat_to_rot(gq[...,-4:])),dim=-1)
# def gm_to_gquat(gm):
#     return torch.cat((gm[...,:3],torch.log(gm[...,3:6]),rot_to_quat(gm[...,-9:])),dim=-1)


def quat_to_rot(q):
    shape = q.shape
    q = q.view(-1, 4)
    norms=q.norm(dim=-1, keepdim=True)
    q = q / norms
    q_sq = 2 * q[:, :, None] * q[:, None, :]
    m00 = 1 - q_sq[:, 1, 1] - q_sq[:, 2, 2]
    m01 = q_sq[:, 0, 1] - q_sq[:, 2, 3]
    m02 = q_sq[:, 0, 2] + q_sq[:, 1, 3]

    m10 = q_sq[:, 0, 1] + q_sq[:, 2, 3]
    m11 = 1 - q_sq[:, 0, 0] - q_sq[:, 2, 2]
    m12 = q_sq[:, 1, 2] - q_sq[:, 0, 3]

    m20 = q_sq[:, 0, 2] - q_sq[:, 1, 3]
    m21 = q_sq[:, 1, 2] + q_sq[:, 0, 3]
    m22 = 1 - q_sq[:, 0, 0] - q_sq[:, 1, 1]
    r = torch.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), dim=-1)
    
    zero_mask = norms.squeeze(-1) == 0
    r[zero_mask] = 0
    
    r = r.view(*shape[:-1], 9)

    return r


@functools.lru_cache(10)
def get_rotation_matrix(theta: float, axis: float, degree: bool = False):
    if degree:
        theta = theta * np.pi / 180
    rotate_mat = np.eye(3)
    rotate_mat[axis, axis] = 1
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotate_mat[(axis + 1) % 3, (axis + 1) % 3] = cos_theta
    rotate_mat[(axis + 2) % 3, (axis + 2) % 3] = cos_theta
    rotate_mat[(axis + 1) % 3, (axis + 2) % 3] = sin_theta
    rotate_mat[(axis + 2) % 3, (axis + 1) % 3] = -sin_theta
    return rotate_mat

def quaternion_to_basic(quaternions):
    # 归一化四元数
    norms = torch.norm(quaternions, p=2, dim=-1, keepdim=True)
    q = quaternions / norms

    # 提取归一化后的四元数分量
    a = q[..., 0]
    b = q[..., 1]
    c = q[..., 2]
    d = q[..., 3]

    # 计算旋转矩阵中的元素
    R11 = 1 - 2 * (c**2 + d**2)
    R12 = 2 * (b * c - a * d)
    R13 = 2 * (b * d + a * c)
    R21 = 2 * (b * c + a * d)
    R22 = 1 - 2 * (b**2 + d**2)
    R23 = 2 * (c * d - a * b)
    R31 = 2 * (b * d - a * c)
    R32 = 2 * (c * d + a * b)
    R33 = 1 - 2 * (b**2 + c**2)

    basic = torch.stack([R11, R21, R31, R12, R22, R32, R13, R23, R33], dim=-1)

    # 将norm=0的位置替换为全0
    zero_mask = norms.squeeze(-1) == 0
    basic[zero_mask] = 0

    return basic

    # # 组装旋转矩阵
    # rotation_matrices = torch.stack([
    #     torch.stack([R11, R12, R13], dim=-1),
    #     torch.stack([R21, R22, R23], dim=-1),
    #     torch.stack([R31, R32, R33], dim=-1)
    # ], dim=-2)


# class BatchTransformer:
#     def generate_random_transformation(self, probability=1.0, device="cpu"):
#         # 通过调用 generate_k_random_transformations 生成单个变换
#         rotations, translations, scales = self.generate_k_random_transformations(
#             1, probability=probability, device=device
#         )
#         return rotations, translations, scales

#     def generate_k_random_transformations(self, k, probability=1.0, device="cpu"):
#         # Determine whether to generate random transformations or zero transformations based on the provided probability
#         # random_selection = np.random.random(k) < probability

#         # Generate random translations or zero translations
#         translations = np.where(
#             np.random.random(k) < probability,
#             np.random.uniform(-0.3, 0.3, (k, 3)),
#             np.zeros((k, 3)),
#         )

#         # Generate random scales or unity scales
#         scales = np.where(
#             np.random.random(k) < probability,
#             np.random.uniform(0.7, 1.3, k),
#             np.ones(k),
#         )

#         # Generate random rotations or identity matrices
#         random_selection = np.random.random(k) < probability
#         rotations = R.random(k)
#         rotation_matrices = np.array(
#             [
#                 r.as_matrix() if random_selection[i] else np.eye(3)
#                 for i, r in enumerate(rotations)
#             ]
#         )

#         # Convert to PyTorch tensors
#         translations_tensor = torch.tensor(
#             translations, dtype=torch.float32, device=device
#         )
#         scales_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
#         rotation_matrices_tensor = torch.tensor(
#             rotation_matrices, dtype=torch.float32, device=device
#         )

#         return rotation_matrices_tensor, translations_tensor, scales_tensor

#     def apply_transformation_to_points(self, points, rotations, translations, scales):

#         # 应用变换
#         points_transformed = points * scales.view(-1, 1)
#         points_transformed = torch.einsum("kij,ni->knj", rotations, points_transformed)
#         points_transformed += translations.view(-1, 3)

#         return points_transformed

#     def apply_transformation_to_gms(self, gms, rotations, translations, scales):
#         gm_is_zero = torch.all(gms == 0, dim=-1, keepdim=True)

#         # 分解 gm 组件
#         ci = gms[..., :3]
#         si = gms[..., 3:6]
#         u1i, u2i, u3i = gms[..., 6:9], gms[..., 9:12], gms[..., 12:15]

#         # 批量应用变换到 gm 组件
#         ci_transformed = ci * scales.view(-1, 1)
#         ci_transformed = torch.einsum("kij,ni->knj", rotations, ci_transformed)
#         ci_transformed += translations.view(-1, 3)

#         u1i_transformed = torch.einsum("kij,ni->knj", rotations, u1i)
#         u2i_transformed = torch.einsum("kij,ni->knj", rotations, u2i)
#         u3i_transformed = torch.einsum("kij,ni->knj", rotations, u3i)
#         si_transformed = torch.einsum("k,nj->knj", scales, si)

#         # 重新组合 gms
#         gms_transformed = torch.cat(
#             [
#                 ci_transformed,
#                 si_transformed,
#                 u1i_transformed,
#                 u2i_transformed,
#                 u3i_transformed,
#             ],
#             axis=-1,
#         )
#         gms_transformed = torch.where(gm_is_zero, gms, gms_transformed)

#         return gms_transformed


if __name__ == "__main__":
    # 示例使用
    gm = torch.tensor(
        [1, 2, 3, 0.5, 0.5, 0.5, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    )  # 示例 gm 参数
    transformer = GMTransformer(gm)  # 使用 NumPy
    points = np.random.rand(10, 3)  # 示例点集
    transformed_points = transformer.transform(points)
    inverse_transformed_points = transformer.inverse_transform(transformed_points)
    inverse_gm = transformer.get_inverse_gm()

    print("Transformed Points:", transformed_points)
    print("Inverse Transformed Points:", inverse_transformed_points)
    print("Inverse gm:", inverse_gm)
