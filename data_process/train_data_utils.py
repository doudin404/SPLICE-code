import os
import json
from types import SimpleNamespace
from typing import List, Generator, Callable, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from scipy.spatial.transform import Rotation as R
from shape_process.gm_transformer import GMTransformer

data = SimpleNamespace(
    id=None,
    parts=SimpleNamespace(
        center=None,
        obb_voxel=None,
        voxel_indices=None,
        patch_size=None,
    ),
    sample=SimpleNamespace(
        global_sample=SimpleNamespace(points=None, labels=None),
        near_a=SimpleNamespace(points=None, labels=None),
        near_b=SimpleNamespace(points=None, labels=None),
    ),
)


class TrainPointExtractPipeline:
    """
    该生成器类接受一个input_generator，并从中获取形如以下结构的数据：

        data = SimpleNamespace(
            id=None,
            parts=SimpleNamespace(
                center=None,
                obb_voxel=None,
                voxel_indices=None,
                voxel_indices=None,
                patch_size=None,
            ),
            sample=SimpleNamespace(
                global_sample=SimpleNamespace(points=None, labels=None),
                near_a=SimpleNamespace(points=None, labels=None),
                near_b=SimpleNamespace(points=None, labels=None),
            ),
        )

    将global_sample、near_a和near_b中的所有点和对应标签合并并转换为torch.tensor，
    并将结果分别保存在data.sample.train.points和data.sample.train.labels中。
    同时将data.parts下的所有数据全部转换为torch.tensor，obb_voxel和voxel是list，其中每个元素是np.array。
    然后yield处理后的data对象。

    参数:
    - input_generator (Generator): 输入数据生成器，生成包含数据的SimpleNamespace对象。
    """

    def __init__(
        self, input_generator: Generator[SimpleNamespace, None, None], sample_num,type="train",**cfg
    ):
        self.input_generator = input_generator
        self.sample_num = sample_num  # 添加采样率参数
        self.type=type
        self.select=["0086","0299","0575","1842","2181","2378","2454","3557","3876","3898","3934","4150","4437","4470","4655","4835"]

    def __len__(self):
        return len(self.input_generator)

    def __iter__(self):
        for i, data in enumerate(self.input_generator):
            if i==100 and self.type=="val" :break
            # if data.id not in self.select  and self.type=="val" :continue
            all_points = []
            all_labels = []
            all_sdfs = []
            if not hasattr(data.sample.global_sample, "sdf"):
                continue

            for sample_type in ["global_sample", "near_a", "near_b"]if self.type!="val" else ["global_sample"]:
                sample = getattr(data.sample, sample_type)

                if sample.points is not None and sample.labels is not None:
                    all_points.append(sample.points)
                    all_labels.append(sample.labels)
                    all_sdfs.append(sample.sdf)
                delattr(data.sample, sample_type)

            all_points = np.concatenate(all_points, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            all_sdfs = np.concatenate(all_sdfs, axis=0)

            total_points = len(all_points)
            sample_size = self.sample_num if self.type!="val" else 40000
            indices = np.random.choice(total_points, sample_size, replace=False)
            sampled_points = all_points[indices]
            sampled_labels = all_labels[indices]
            sampled_sdfs = all_sdfs[indices]

            data.sample.train = SimpleNamespace()
            data.sample.train.points = torch.tensor(sampled_points)
            data.sample.train.labels = torch.tensor(sampled_labels)
            data.sample.train.sdfs = torch.tensor(sampled_sdfs)

            # Convert data.parts to torch.tensor
            parts_dict = data.parts.__dict__
            for key, value in parts_dict.items():
                if isinstance(value, list):
                    parts_dict[key] = [
                        torch.tensor(v) if isinstance(v, np.ndarray) else v
                        for v in value
                    ]
                elif isinstance(value, np.ndarray):
                    parts_dict[key] = torch.tensor(value)

            data.parts = SimpleNamespace(**parts_dict)

            yield data


def generate_k_random_transformations(k, probability=1.0, device="cpu"):
    # Determine whether to generate random transformations or zero transformations based on the provided probability
    # random_selection = np.random.random(k) < probability

    # Generate random translations or zero translations
    translations = np.where(
        np.random.random(k) < probability,
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


def get_random_transform(probability=1, device="cpu"):
    rotations, translations, scales = generate_k_random_transformations(
        1, probability=probability, device=device
    )
    return GMTransformer(rotations=rotations, translations=translations, scales=scales)
