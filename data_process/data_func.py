"""
一大堆和数据处理有关的函数

"""

import os
import json
from types import SimpleNamespace
import numpy as np
from shape_process.mesh_sampler import MeshSampler


def shape_classification(data: SimpleNamespace):
    """
    形状判别程序。

    参数:
    - data (Data): 包含要处理的网格数据。

    返回值:
    - bool: 如果true样本点占比少于0.4，则返回false，否则返回true。
    """
    mesh = data.mesh

    # 创建 MeshSampler 实例
    sampler = MeshSampler(mesh)

    # 使用 near_a 策略进行采样
    num_samples = 1000000  # 采样点的数量，可以根据需要调整
    strategy = [("near_a", 1.0)]
    points, labels = sampler.sample_points(num_samples, strategy, compute_labels=True)

    # 计算 true 样本点的占比
    true_ratio = np.mean(labels)
    data.true_ratio = true_ratio
    # print(data.id, data.true_ratio)

    # 判别形状特征
    return true_ratio >= 0.4

def is_valid_shape(shape, partnet_path="D:\data\data_v0", required_categories=None, black=False):
    """
    Checks whether the input shape belongs to the required categories.

    Parameters:
    - shape: The input shape object, which should have an 'id' attribute.
    - partnet_path (str): The base path to the PartNet dataset.
    - required_categories (list or None): The list of categories to check against.
    - black (bool): If True, treat required_categories as blacklist; otherwise as whitelist.

    Returns:
    - bool: True if the shape's category is valid, False otherwise.
    """
    if required_categories is None:
        return True
    id_ = shape.id
    meta_path = os.path.join(partnet_path, id_, "meta.json")
    
    # 获取形状的类别
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        meta_data = json.load(meta_file)
        category = meta_data.get("model_cat")
    
    # 检查类别是否在要求的类别中
    if required_categories is None:
        return True
    if not black:
        return category in required_categories
    else:
        return category not in required_categories

def extract_leaf_components(tree: list):
    result = []

    def traverse(node):

        # 如果没有子节点，说明是叶子节点，添加到结果列表
        if "children" not in node or not node["children"]:
            result.extend(node["objs"])
        else:
            # 递归遍历子节点
            for child in node["children"]:
                traverse(child)

    traverse(tree)

    return result

def populate_objs_in_tree(tree: dict) -> list:
    def traverse_and_collect(node):
        # 如果没有子节点，说明是叶子节点，直接返回该节点的 "objs" 列表
        if "children" not in node or not node["children"]:
            return node.get("objs", [])
        else:
            # 否则，递归遍历子节点，收集所有子节点的 "objs"
            all_objs = []
            for child in node["children"]:
                child_objs = traverse_and_collect(child)
                all_objs.extend(child_objs)
            
            # 将收集到的所有 "objs" 列表拼接，并赋值给当前节点的 "objs" 词条
            node["objs"] = all_objs
            return all_objs

    traverse_and_collect(tree)

def extract_parts(data: SimpleNamespace, tree_name: str, obj_folder: str):
    """
    这是一个生成器流水线,用于提取输入流水线形状中的所有部件的aabb包围盒,首先使用extract_leaf_components提取出二维数组。
    从每个data.path的obj_folder文件夹下读取每个部件对应的一组名称的形状.
    最后,计算出每个部件的AABB,写入data.parts_aabb

    参数:
    ()
    """
    tree_path = os.path.join(data.path, tree_name)
    with open(tree_path, "r", encoding="utf-8") as tree_file:
        tree_data = json.load(tree_file)
        part_names = extract_leaf_components(tree_data)
