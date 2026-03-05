import numpy as np
import torch

class OrthogonalRotationGenerator:
    def __init__(self):
        # 24 Orthogonal Rotation Matrices representing the symmetry of a cube
        self.rotation_matrices = [
            # np.eye(3),
            # -np.eye(3),
            
            # np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            # np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
            # np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
            # np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            
            # np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            # np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            # np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
            # np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            
            # np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            # np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            # np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
            # np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
            
            # np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            # np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]]),
            # np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            # np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]]),
            
            # np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            # np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            # np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]]),
            # np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]),
            # np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
            # np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
            
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
            np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
            
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
            np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
            
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
            np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
            
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
            np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
            np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        ]

    def sample(self, num_samples=1, apply_rotation_prob=1.0):
        sampled_matrices = []
        for _ in range(num_samples):
            if np.random.rand() < apply_rotation_prob:
                # Apply rotation with given probability
                index = np.random.choice(len(self.rotation_matrices))
                sampled_matrices.append(self.rotation_matrices[index])
            else:
                # If not applying rotation, return identity matrix
                sampled_matrices.append(np.eye(3))
        return sampled_matrices

class VoxelRotator:
    def __init__(self, voxel_range=64):
        self.voxel_range = voxel_range
        self.rotation_generator = OrthogonalRotationGenerator()

    def apply_rotation(self, voxel_list, apply_rotation=True):
        rotated_voxels = []
        rotation_matrices = []

        for voxels in voxel_list:
            # Randomly sample a rotation matrix and convert it to torch tensor
            rotation_matrix_np = self.rotation_generator.sample(1, apply_rotation_prob= 0 if apply_rotation else 0)[0]
            rotation_matrix = torch.tensor(rotation_matrix_np, dtype=torch.float32, device=voxels.device)
            rotation_matrices.append(rotation_matrix)

            # Convert the voxel indices to torch tensor and normalize them
            voxel_indices = voxels.float()
            voxel_indices -= (self.voxel_range - 1) / 2.0

            # Apply the rotation using torch.matmul
            rotated_indices=torch.einsum("ij,ni->nj", rotation_matrix, voxel_indices)

            # Shift back to the original coordinate system
            rotated_indices += (self.voxel_range - 1) / 2.0
            rotated_indices = torch.round(rotated_indices).int()

            # Convert back to original device and data type
            rotated_voxels.append(rotated_indices.to(voxels.device, dtype=torch.int32))

        # Convert rotation matrices to torch tensor
        rotation_matrices_tensor = torch.stack(rotation_matrices)

        return rotation_matrices_tensor, rotated_voxels

    def apply_rotation_batch(self, batch_voxel_list, apply_rotation=True):
        batch_rotated_voxels = []
        batch_rotation_matrices = []

        for voxel_list in batch_voxel_list:
            rotation_matrices_tensor, rotated_voxels = self.apply_rotation(voxel_list, apply_rotation=apply_rotation)
            batch_rotated_voxels.append(rotated_voxels)
            batch_rotation_matrices.append(rotation_matrices_tensor)

        # Convert list of rotation matrices into a single tensor
        # batch_rotation_matrices_tensor = torch.cat(batch_rotation_matrices, dim=0)

        return batch_rotation_matrices, batch_rotated_voxels
    
    def inverse_transform_gmm(self, gms, rotation_matrices, mask=None):
        # Apply inverse rotation to gms based on the part_rotation
        gmm_is_zero = torch.all(gms == 0, dim=-1, keepdim=True)

        # Decompose GMM components
        ci = gms[..., :3]
        si = gms[..., 3:6]
        u1i, u2i, u3i = gms[..., 6:9], gms[..., 9:12], gms[..., 12:15]

        # Apply inverse rotations
        #ci_transformed = torch.einsum("knji,kni->knj", rotation_matrices, ci)
        u1i_transformed = torch.einsum("knji,kni->knj", rotation_matrices, u1i)
        u2i_transformed = torch.einsum("knji,kni->knj", rotation_matrices, u2i)
        u3i_transformed = torch.einsum("knji,kni->knj", rotation_matrices, u3i)

        # Reassemble GMMs
        gmms_transformed = torch.cat(
            [
                ci,
                si,
                u1i_transformed,
                u2i_transformed,
                u3i_transformed,
            ],
            axis=-1,
        )
        gmms_transformed = torch.where(gmm_is_zero, gms, gmms_transformed)

        # Apply mask if provided
        if mask is not None:
            gmms_transformed = torch.where(mask.unsqueeze(-1), torch.zeros_like(gmms_transformed), gmms_transformed)

        return gmms_transformed