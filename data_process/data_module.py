# Python标准库
from collections import defaultdict
from types import SimpleNamespace
import os

# 第三方库
import h5py
import pytorch_lightning as pl
import numpy as np
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate

# 项目内部模块
import data_process.data_utils
import data_process.namespace_hdf5
import data_process.train_data_utils
import net
from data_process.train_data_utils import (
    get_random_transform,
    TrainPointExtractPipeline,
)

# data = SimpleNamespace(
#     id=None,  # 字符串直接作为list合并
#     parts=SimpleNamespace(
#         center=None,  # 在新的维度拼成一个张量
#         obb_voxel=None,  # 长度不一样,作为list合并
#         voxel_indices=None,  # 长度不一样,作为list合并
#         patch_size=None,  # 在新的维度拼成一个张量
#     ),
#     sample=SimpleNamespace(
#         train=SimpleNamespace(
#             points=None, labels=None  # 在新的维度拼成一个张量  # 在新的维度拼成一个张量
#         ),
#     ),
# )


class customPipeline:

    def __init__(
        self,
        input_generator,
        **cfg,
    ):
        self.input_generator = input_generator

    def __iter__(self):
        for data in self.input_generator:
            data.sample.near_a.sdf = data.sample.near_a.sdf.astype(np.float32)
            data.sample.near_b.sdf = data.sample.near_b.sdf.astype(np.float32)
            data.sample.global_sample.sdf = data.sample.global_sample.sdf.astype(
                np.float32
            )

            temp = []
            for i in data.parts.obb_voxel:
                temp.append(i.astype(np.float32))
            data.parts.obb_voxel = temp
            
            data.parts.patch_size = data.parts.patch_size.astype(np.float32)
            yield data


class NoodlesDataset(IterableDataset):
    def __init__(self, datas, **cfg):
        self.datas = datas

    def __iter__(self):
        return iter(self.datas)


def pad_sequences(sequences, maxlen, dtype=torch.float32):
    padded_sequences = torch.zeros(
        (len(sequences), maxlen) + sequences[0].shape[1:], dtype=dtype
    )
    for i, seq in enumerate(sequences):
        padded_sequences[i, : len(seq)] = seq
    return padded_sequences


def collate_fn(batch):
    # 创建一个用于存储重构数据的字典
    batched_data = defaultdict(list)

    # 遍历每个数据项
    for item in batch:
        # 处理 id
        batched_data["id"].append(item.id)

        # 处理 parts
        if item.parts.center is not None:
            batched_data["parts_center"].append(item.parts.center)
        if item.parts.obb_voxel is not None:
            batched_data["parts_obb_voxel"].append(item.parts.obb_voxel)
        if item.parts.voxel_indices is not None:
            batched_data["parts_voxel_indices"].append(item.parts.voxel_indices)
        if item.parts.patch_size is not None:
            batched_data["parts_patch_size"].append(item.parts.patch_size)
        if item.parts.voxel_labels is not None:
            batched_data["parts_voxel_labels"].append(item.parts.voxel_labels)

        # 处理 sample
        if item.sample.train.points is not None:
            batched_data["sample_train_points"].append(item.sample.train.points)
        if item.sample.train.labels is not None:
            batched_data["sample_train_labels"].append(item.sample.train.labels)
        if item.sample.train.sdfs is not None:
            batched_data["sample_train_sdfs"].append(item.sample.train.sdfs)

    # 找到需要填充的最大长度
    # max_len_center = max(len(seq) for seq in batched_data["parts_center"])
    # max_len_patch_size = max(len(seq) for seq in batched_data["parts_patch_size"])

    # # 填充 sequences 到最大长度
    # batched_data["parts_center"] = pad_sequences(
    #     batched_data["parts_center"], max_len_center
    # )
    # batched_data["parts_patch_size"] = pad_sequences(
    #     batched_data["parts_patch_size"], max_len_patch_size
    # )

    # 对需要在新维度上拼接的数据使用 default_collate
    for key in [
        "sample_train_points",
        "sample_train_labels",
        "sample_train_sdfs",
    ]:
        batched_data[key] = default_collate(batched_data[key])

    return dict(batched_data)

def save_ids_txt(data_to_extract, extract_dir):
    ids_path = os.path.join(extract_dir, "ids.txt")
    with open(ids_path, "w") as ids_file:
        for data in tqdm.tqdm(data_to_extract):
            ids_file.write(str(data.id) + "\n")

class NoodlesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, sample_num, **cfg):
        super().__init__()
        self.pipe = data_process.data_utils.DataLoader(data_path,shuffle=True)
        self.pipe = data_process.data_utils.FilterPipeline.filter_no_valid_shape(self.pipe)#,required_categories=["StorageFurniture"])  
        # [
        # "Chair",
        # "Table",
        # #"Bed",
        # "StorageFurniture",
        # #"Lamp",
        # ]
        self.piped = customPipeline(self.pipe)
        self.pipe = data_process.train_data_utils.TrainPointExtractPipeline(
            self.piped, sample_num
        )
        self.dataset = NoodlesDataset(self.pipe)

        #save_ids_txt(self.dataset, "extract")

        self.pipe = data_process.train_data_utils.TrainPointExtractPipeline(
            self.piped, sample_num, type="val"
        )
        self.val_dataset = NoodlesDataset(self.pipe)
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset.mode = stage

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, collate_fn=collate_fn)

class FryDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_path,gms_factor=1,shape_z_factor=1, **cfg):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.gms_factor = gms_factor
        self.shape_z_factor = shape_z_factor

    def setup(self, stage=None):
        # Load the datasets from the provided paths
        gms_path = os.path.join(self.data_path, "extracted_gms.h5")
        shape_z_path = os.path.join(self.data_path, "extracted_shape_z.h5")

        with h5py.File(gms_path, "r") as gms_file, h5py.File(
            shape_z_path, "r"
        ) as shape_z_file:
            gms = torch.tensor(gms_file["gms"][:])
            shape_z = torch.tensor(shape_z_file["shape_z"][:])

        # Ensure the data is of the expected shape
        assert shape_z.shape[-1] == 512

        # Concatenate along the last dimension
        concatenated_data = torch.tensor(
            np.concatenate((gms[...,:15], shape_z), axis=-1)
        )  # [sample_num, len, 15 + 512]
        concatenated_data = concatenated_data.squeeze(1)
        #concatenated_data = concatenated_data[...,:15]
        # concatenated_data = torch.zeros_like(concatenated_data)
        # for i in range(concatenated_data.shape[-2]):
        #     for j in range(concatenated_data.shape[-1]):
        #         if i % 4 == 0:
        #             concatenated_data[..., i, j] = 0
        #         if i % 4 == 1:
        #             concatenated_data[..., i, j] = 1
        #         if i % 4 == 2:
        #             concatenated_data[..., i, j] = j % 2
        #         if i % 4 == 3:
        #             concatenated_data[..., i, j] = 1 - (j % 2)

        # []

        # Split the data into train and validation sets
        self.val_data = concatenated_data[:6]  # 前六个数据作为验证集
        self.train_data = concatenated_data[6:]  # 其余数据作为训练集
        self.train_data = torch.tile(self.train_data, (10, 1, 1))

        # Create datasets
        self.train_dataset = self.train_data
        self.val_dataset = self.val_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # Assuming you want to use the whole dataset for testing
        test_dataset = np.concatenate((self.val_data, self.train_data), axis=0)
        return DataLoader(test_dataset, batch_size=1, shuffle=True)

    def split_gms_shapez(self, data):
        """
        将连接后的数据重新拆分回 gms 和 shape_z.
        输入数据格式：[sample_num, len, 15+512]
        返回两个部分，分别是 gms 和 shape_z
        """
        gms = data[:, :, :15]
        shape_z = data[:, :, 15:]
        return gms, shape_z
