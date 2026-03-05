"""
用于测试数据处理模块
"""

import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from data_process.data_utils import (
    DataTraverser,
    ManifoldPlusPipeline,
    ShapeReaderPipeline,
    FilterPipeline,
    TreeExtractPipeline,
    DataCleanPipeline,
    DataLoader,
    PathPipeline,
    ObjFilePipeline,
    JsonFilePipeline,
    GmmFilePipeline,
)
from data_process.point_sampler import (
    PointSamplePipeline,
    PartPointSamplePipeline,
    InternalPointSamplePipeline,
    SDFPipeline,
)
from data_process.data_func import shape_classification
from data_process.namespace_hdf5 import NamespaceHDF5

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def ManifoldPlus(cfg: DictConfig = None) -> None:
    pipeline = DataTraverser(
        **cfg.data,
    )
    pipeline = TreeExtractPipeline(**cfg.data, input_generator=pipeline)
    pipeline = ManifoldPlusPipeline(**cfg.data, input_generator=pipeline)

    for data in tqdm(pipeline):
        []


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def full_pipeline_test(cfg: DictConfig = None) -> None:
    """
    main
    """
    plus_ones=[]
    
    pipeline = DataTraverser(
        **cfg.data,plus_ones=plus_ones,
    )
    pipeline = TreeExtractPipeline(**cfg.data, input_generator=pipeline)
    pipeline = ManifoldPlusPipeline(**cfg.data, input_generator=pipeline)
    

    pipeline = ShapeReaderPipeline(input_generator=pipeline, **cfg.data)
    pipeline = FilterPipeline.filter_low_volume(input_generator=pipeline)
    pipeline = InternalPointSamplePipeline(input_generator=pipeline, **cfg.data)

    pipeline = PartPointSamplePipeline(input_generator=pipeline, **cfg.data)

    pipeline = PointSamplePipeline(input_generator=pipeline, **cfg.data)
    pipeline = SDFPipeline(input_generator=pipeline)

    pipeline = DataCleanPipeline.default(input_generator=pipeline)

    progress_bar = tqdm(pipeline)
    count=0
    def plus_one():
        nonlocal count
        nonlocal progress_bar
        count+=1
        progress_bar.set_postfix({"count": count})
    plus_ones.append(plus_one)

    plus_ones
    for data in progress_bar:
        h5 = NamespaceHDF5(f"{cfg.data.output_dir}/{data.id}.hdf5")
        h5.save_namespace(data)
        progress_bar.set_postfix({"count": count})


class customPipeline:

    def __init__(
        self,
        input_generator,
        **cfg,
    ):
        self.input_generator = input_generator

    def __iter__(self):
        for data in self.input_generator:

            yield data


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def data_custom(cfg: DictConfig = None) -> None:
    pipe = DataLoader(cfg.train.data_path)
    pipe = PathPipeline(pipe, cfg.data.partnet_path)
    pipe = ShapeReaderPipeline(pipe, **cfg.data)
    # pipe = TreeExtractPipeline(**cfg.data, input_generator=pipe)
    pipe = customPipeline(pipe)
    # pipe = InternalPointSamplePipeline(input_generator=pipe, **cfg.data)

    # pipe = SDFPipeline(pipe)
    # pipe = PartPointSamplePipeline(input_generator=pipe, **cfg.data)
    pipe = DataCleanPipeline.default(pipe)

    for data in tqdm(pipe):
        h5 = NamespaceHDF5(f"data/{data.id}.hdf5")
        h5.save_namespace(data)

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def delete_list(cfg: DictConfig = None) -> None:
    """
    main
    """
    # pipeline = DataTraverser(
    #     **cfg.data
    # )
    pipe = DataLoader(cfg.train.data_path)
    def data_filter(input_generator):
        for data in input_generator:
            # 检查 voxel_indices 是否存在或小于 100
            if data.parts.voxel_indices is None:
                yield data
            else:
                for i in zip(data.parts.obb_voxel):
                    if len(i[0])< 10:
                        # if len(i)!=0:
                            # from shape_process.visualizationManager import VisualizationManager
                            # vm=VisualizationManager()
                            # vm.add_points(i,j)
                            # vm.show()
                        yield data
                        break

    pipeline = data_filter(
        pipe
    )

    # 打开一个文本文件以写入模式
    with open("data_ids.txt", "w") as file:
        # 遍历 pipeline 中的数据
        for data in tqdm(pipeline):
            # 将 data.id 写入文本文件，每个 ID 占一行
            file.write(f"{data.id}\n")

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def data_show(cfg: DictConfig = None) -> None:
    pipe = DataLoader(cfg.train.data_path,data_list=["22690"])
    # pipe = PathPipeline(pipe, cfg.data.partnet_path)
    # pipe = ShapeReaderPipeline(pipe, **cfg.data)
    # pipe = TreeExtractPipeline(**cfg.data, input_generator=pipe)
    # pipe = customPipeline(pipe)
    # pipe = InternalPointSamplePipeline(input_generator=pipe, **cfg.data)

    # pipe = SDFPipeline(pipe)
    # pipe = PartPointSamplePipeline(input_generator=pipe, **cfg.data)
    # pipe = DataCleanPipeline.default(pipe)

    for data in tqdm(pipe):

        for i in range(len(data.parts.obb_voxel)):
            from shape_process.visualizationManager import VisualizationManager
            vm=VisualizationManager()
            vm.add_points(data.parts.obb_voxel[i])
            vm.show()

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def add_new_data(cfg: DictConfig = None) -> None:
    """
    main
    """
    plus_ones=[]
    
    pipeline = ObjFilePipeline(**cfg.data)
    pipeline = JsonFilePipeline(input_generator=pipeline,**cfg.data)
    
    pipeline = FilterPipeline.filter_low_volume(input_generator=pipeline)
    pipeline = InternalPointSamplePipeline(input_generator=pipeline, **cfg.data)

    pipeline = PartPointSamplePipeline(input_generator=pipeline, **cfg.data)

    pipeline = PointSamplePipeline(input_generator=pipeline, **cfg.data)
    pipeline = SDFPipeline(input_generator=pipeline)

    pipeline = DataCleanPipeline.default(input_generator=pipeline)

    progress_bar = tqdm(pipeline)
    count=0
    def plus_one():
        nonlocal count
        nonlocal progress_bar
        count+=1
        progress_bar.set_postfix({"count": count})
    plus_ones.append(plus_one)

    for data in progress_bar:
        h5 = NamespaceHDF5(f"D:/data/train_data/{data.id}.hdf5")
        h5.save_namespace(data)
        progress_bar.set_postfix({"count": count})

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def add_spa_data(cfg: DictConfig = None) -> None:
    """
    main
    """
    plus_ones=[]
    
    pipeline = ObjFilePipeline(**cfg.data)
    pipeline = GmmFilePipeline(input_generator=pipeline,**cfg.data)
    
    pipeline = FilterPipeline.filter_low_volume(input_generator=pipeline)
    pipeline = InternalPointSamplePipeline(input_generator=pipeline, **cfg.data)

    pipeline = PartPointSamplePipeline(input_generator=pipeline, **cfg.data)

    pipeline = PointSamplePipeline(input_generator=pipeline, **cfg.data)
    pipeline = SDFPipeline(input_generator=pipeline)

    pipeline = DataCleanPipeline.default(input_generator=pipeline)

    progress_bar = tqdm(pipeline)
    count=0
    def plus_one():
        nonlocal count
        nonlocal progress_bar
        count+=1
        progress_bar.set_postfix({"count": count})
    plus_ones.append(plus_one)

    for data in progress_bar:
        h5 = NamespaceHDF5(f"D:/data/train_data_airplanes/{data.id}.hdf5")
        h5.save_namespace(data)
        progress_bar.set_postfix({"count": count})
    
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def count(cfg: DictConfig = None) -> None:
        pipe = DataLoader(cfg.train.data_path,shuffle=True)
        pipe = FilterPipeline.filter_no_valid_shape(pipe,required_categories=["Table"])  
        print(len(list(pipe)))
        # [
        # "Chair",
        # "Table",
        # #"Bed",
        # "StorageFurniture",
        # #"Lamp",
        # ]
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def data_prepare(cfg: DictConfig = None) -> None:
    """
    main
    """
    plus_ones=[]
    
    pipeline = DataTraverser(
        **cfg.data,plus_ones=plus_ones,
    )
    pipeline = TreeExtractPipeline(**cfg.data, input_generator=pipeline)
    pipeline = ManifoldPlusPipeline(**cfg.data, input_generator=pipeline)
    

    pipeline = ShapeReaderPipeline(input_generator=pipeline, **cfg.data)
    pipeline = FilterPipeline.filter_low_volume(input_generator=pipeline)
    pipeline = InternalPointSamplePipeline(input_generator=pipeline, **cfg.data)

    pipeline = PartPointSamplePipeline(input_generator=pipeline, **cfg.data)

    pipeline = PointSamplePipeline(input_generator=pipeline, **cfg.data)
    pipeline = SDFPipeline(input_generator=pipeline)

    pipeline = DataCleanPipeline.default(input_generator=pipeline)

    progress_bar = tqdm(pipeline)
    count=0
    def plus_one():
        nonlocal count
        nonlocal progress_bar
        count+=1
        progress_bar.set_postfix({"count": count})
    plus_ones.append(plus_one)

    plus_ones
    for data in progress_bar:
        h5 = NamespaceHDF5(f"{cfg.data.output_dir}/{data.id}.hdf5")
        h5.save_namespace(data)
        progress_bar.set_postfix({"count": count})
if __name__ == "__main__":
    #ManifoldPlus()#
    #full_pipeline_test()
    #delete_list()
    #data_show()
    # data_custom()
    #add_new_data()
    #add_spa_data()
    data_prepare()