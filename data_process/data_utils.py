"""
一大堆和数据处理有关的小类
大类需要单开一个文件
"""

import os
import json
import random
import numpy as np
from types import SimpleNamespace
from typing import List, Generator, Callable, Any, Dict
import trimesh
from data_process.manifold_plus import ManifoldPlusRunner
from data_process.data_func import (
    extract_parts,
    shape_classification,
    extract_leaf_components,
    populate_objs_in_tree,
    is_valid_shape,
)
from data_process.namespace_hdf5 import NamespaceHDF5
import net.model_func
from shape_process.visualizationManager import VisualizationManager
import net


class DataTraverser:
    """
    这个类的功能是从partnet数据集读取数据地址并以迭代器的形式返回,也就是说这是一个生成器
    为了方便搭建数据处理流水线，使用SimpleNamespace.path的方式返回对应数据的路径,
    支持输入一组类别标签以只读取 例如["Chair","Table"]
    输入参数包含:
    - cfg: 配置对象,其中包含partnet数据的路径 (cfg.data.manifold_plus_path),
    在读取时,如果这个路径下有名为dataset的文件夹,则以partnet_path/dataset
    为真实路径,路径下方是一些以数字命名的文件夹代表每一个数据.
    - categorys: 形如["Chair","Table"]的list,如果没提供则读取每一个数据.
    从每个数据的文件夹下的meta.json确认model_cat属性是否在list中
    - data_ids_path: 包含要处理的数据ID的文件路径，如果提供则只处理这些ID
    """

    def __init__(
        self,
        partnet_path: str,
        output_dir: str,
        categorys: List[str] = None,
        data_ids_path: str = None,
        black=False,
        plus_ones=None,
        **cfg,
    ):
        self.partnet_path = (
            os.path.join(partnet_path, "dataset")
            if os.path.exists(os.path.join(partnet_path, "dataset"))
            else partnet_path
        )
        self.data_path = output_dir
        self.black = black
        self.categorys = categorys
        self.data_ids = self._read_data_ids(data_ids_path) if data_ids_path else None
        self.plus_ones = plus_ones

    def _read_data_ids(self, data_ids_path: str) -> List[str]:
        with open(data_ids_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file.readlines()]

    def _get_category(self, meta_path: str) -> str:
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            meta_data = json.load(meta_file)
            return meta_data.get("model_cat")

    def _is_valid_category(self, category: str) -> bool:
        if self.categorys is None:
            return True
        return (
            category in self.categorys
            if not self.black
            else category not in self.categorys
        )

    def get_filenames_without_extension(self, directory_path):
        filenames = []
        for filename in os.listdir(directory_path):
            # 检查是否是文件（而不是文件夹）
            if os.path.isfile(os.path.join(directory_path, filename)):
                # 分离文件名和扩展名，并只保留文件名部分
                name_without_extension = os.path.splitext(filename)[0]
                filenames.append(name_without_extension)
        return filenames

    def _get_data_paths(self) -> Generator[SimpleNamespace, None, None]:
        folders = self.data_ids if self.data_ids else os.listdir(self.partnet_path)
        exist_data = self.get_filenames_without_extension(self.data_path)
        # folders = folders[folders.index('27399') :]

        folders = list(set(folders) - set(exist_data))
        # folders = [
        #     "1302",
        #     "22702",
        #     "2331",
        #     "24602",
        #     "25158",
        #     "25731",
        #     "27482",
        #     "3074",
        #     "30763",
        #     "32732",
        #     "34469",
        #     "38076",
        #     "40172",
        #     "45955",
        #     "45980",
        #     "46125",
        # ]
        random.shuffle(folders)
        for folder in folders:
            # if int(folder)<=23208:continue###
            folder_path = os.path.join(self.partnet_path, folder)
            if os.path.isdir(folder_path):
                meta_path = os.path.join(folder_path, "meta.json")
                if os.path.exists(meta_path):
                    category = self._get_category(meta_path)
                    if self._is_valid_category(category):
                        if self.plus_ones is not None:
                            for i in self.plus_ones:
                                i()
                        yield SimpleNamespace(
                            path=folder_path,
                            category=category,
                            id=os.path.basename(folder_path),
                        )
        []

    def __iter__(self):
        return self._get_data_paths()

class ObjFilePipeline:
    def __init__(self, add_data_path,**cfg,):
        self.folder_path = add_data_path
        
    def _get_obj_files(self):
        """获取文件夹中的所有.obj文件"""
        for file in os.listdir(self.folder_path):
            if file.endswith('.obj'):
                yield os.path.join(self.folder_path, file)

    def __iter__(self):
        """迭代器，读取.obj文件并返回SimpleNamespace包含文件名和内容"""
        for obj_file in self._get_obj_files():
            mesh = trimesh.load(obj_file, file_type='obj')
            data = SimpleNamespace()
            data.id = os.path.splitext(os.path.basename(obj_file))[0]  # 保存去掉后缀的文件名
            data.mesh = mesh  # 保存.obj文件内容为trimesh格式
            yield data

class GmmFilePipeline:
    def __init__(self, input_generator, add_data_path, **cfg):
        self.input_generator = input_generator
        self.folder_path = add_data_path
    
    def _get_gmm_data(self, gmm_file_path):
        """从指定路径读取JSON文件并解析数据"""
        parsed = []
        with open(gmm_file_path, 'r') as f:
            lines = [line.strip() for line in f]
        for i, line in enumerate(lines):
            line = line.split(" ")
            arr = [float(item) for item in line]
            arr = np.array(arr)
            if i == 0:
                pass  # 假定第一行是标量或特殊处理
            elif 1 <= i <= 2:
                arr = arr.reshape((-1, 3))
            elif i == 3:
                arr = arr.reshape((-1, 3, 3))
            elif i == 4:  # 假定有布尔值处理
                arr = arr.astype(np.bool_)
            parsed.append(arr)
        parsed[2]=np.sqrt(parsed[2])*1.5
        return parsed
    
    def _filter_gmm_data(self,gmm_data):
        # 筛除 gmm_data[-1] == 0 的 GM
        mask = gmm_data[-1] != 0  # 获取布尔掩码
        gmm_data = [arr[mask] if arr.shape[0] == mask.shape[0] else arr for arr in gmm_data]
        gmm_data = [arr[None,None,:] for arr in gmm_data]
        gmm_data = [gmm_data[1],gmm_data[3],gmm_data[0],gmm_data[2]]
        return gmm_data

    def __iter__(self):
        """迭代器，从输入流水线中获取data，并读取对应JSON文件的数据"""
        for data in self.input_generator:
            gmm_file_path = os.path.join(self.folder_path, f"{data.id}.txt")
            if not os.path.exists(gmm_file_path):
                continue
            
            gmm_data_old = self._get_gmm_data(gmm_file_path)

            gmm_data = self._filter_gmm_data(gmm_data_old)

            # 使用 net.model_func.split_mesh_by_gmm 进行分割
            splited_meshes = net.model_func.split_mesh_by_gmm(data.mesh, gmm_data)

            obb_transforms = []
            obb_extents = []
            flag=False

            # from shape_process.visualizationManager import VisualizationManager
            # vm=VisualizationManager()
        # for i in range(phi.shape[-1]):
        #     vm.add_gmm((mu[0,0,i],p[0,0,i],eigen[0,0,i]))
        # vm.show()
            for i,sp_mesh in enumerate(splited_meshes.values()):
                # 用 trimesh 计算每个 sp 的 OBB（定向包围盒）
                sp_mash=trimesh.Trimesh(vertices=data.mesh.vertices,faces=sp_mesh)
                # vm.plotter.add_mesh(sp_mash,color=np.random.rand(3),opacity=0.5)
                
                obb = sp_mash.bounding_box_oriented
                
                if np.prod(obb.primitive.extents)>0.1:
                    continue
                if np.prod(obb.primitive.extents)>0.04:
                    flag=True
                obb_transforms.append(obb.primitive.transform)
                obb_extents.append(obb.primitive.extents)
                
                
                # vm.add_obb(obb.primitive)
                # vm.add_gmm(np.concatenate([g[i].flatten() for g in gmm_data_old[1:-1]],axis=-1),color="yellow" if not flag else "red")
            # vm.show()

            # 将所有 OBB 数据整理成数组
            obb_transforms_np = np.stack(obb_transforms, axis=0) if obb_transforms else None
            obb_extents_np = np.stack(obb_extents, axis=0) if obb_extents else None

            obb = SimpleNamespace(
                transform=obb_transforms_np,
                extents=obb_extents_np,
            )
            data.parts = SimpleNamespace(obb=obb)

            # 返回包含 OBB 数据的对象
            if flag:
                yield data


class JsonFilePipeline:
    def __init__(self, input_generator, add_data_path, skip_if_after_missing=False,**cfg):
        self.input_generator = input_generator
        self.folder_path = add_data_path
        self.skip_if_after_missing = skip_if_after_missing
    
    def _get_json_data(self, json_file_path):
        """从指定路径读取JSON文件并解析数据"""
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        return json_data

    def _calculate_parts_bounding_boxes(self, json_data):
        """根据JSON数据计算每个部分的OBB并返回SimpleNamespace对象"""
        obb_transforms = []
        obb_extents = []

        for part in json_data:
            # 如果缺少'eigenvectors', 'center', 'lengths'，则使用全0占位
            if 'eigenvectors' not in part or 'center' not in part or 'lengths' not in part:
                rotation_matrix = np.zeros((3, 3))
                center = np.zeros((3, 1))
                lengths = np.zeros(3)
            else:
                rotation_matrix = np.array(part['eigenvectors']).T
                center = np.array(part['center']).reshape(3, 1)
                lengths = part['lengths']
            
            transform = np.hstack((rotation_matrix, center))  # Combine rotation and translation
            transform = np.vstack((transform, [0, 0, 0, 1]))  # Add homogeneous coordinate row
            
            obb_transform = transform
            obb_extent = lengths
            obb_transforms.append(obb_transform)
            obb_extents.append(obb_extent)

        obb_transforms_np = np.stack(obb_transforms, axis=0) if obb_transforms else None
        obb_extents_np = np.stack(obb_extents, axis=0) if obb_extents else None

        obb = SimpleNamespace(
            transform=obb_transforms_np,
            extents=obb_extents_np,
        )

        return obb

    def __iter__(self):
        """迭代器，从输入流水线中获取data，并读取对应JSON文件的数据"""
        for data in self.input_generator:
            json_file_path = os.path.join(self.folder_path, f"{data.id}.json")
            if not os.path.exists(json_file_path):
                continue

            json_data = self._get_json_data(json_file_path)
            obb = self._calculate_parts_bounding_boxes(json_data)
            data.parts = SimpleNamespace(obb=obb)

            # 查找附加的_after.json文件
            json_after_file_path = os.path.join(self.folder_path, f"{data.id}_after.json")
            if os.path.exists(json_after_file_path):
                json_after_data = self._get_json_data(json_after_file_path)
                obb_after = self._calculate_parts_bounding_boxes(json_after_data)
                data.parts_after = SimpleNamespace(obb=obb_after)
            elif self.skip_if_after_missing:
                continue

            yield data


class PathPipeline:
    """
    这个类的功能是从input_generator中读取每一项数据,
    读取对应的id作为folder,并将path和category信息写入数据中。
    支持以生成器的形式返回处理后的数据。

    输入参数包含:
    - input_generator: 输入的数据生成器,每一项包含一个id字段.
    - partnet_path: 数据所在的根路径,将与id组合形成folder路径.
    """

    def __init__(self, input_generator: Generator[Any, None, None], partnet_path: str):
        self.input_generator = input_generator
        self.partnet_path = partnet_path

    def _get_category(self, meta_path: str) -> str:
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            meta_data = json.load(meta_file)
            return meta_data.get("model_cat")

    def _process_item(self, item: SimpleNamespace) -> SimpleNamespace:
        folder = str(item.id)  # 将id转换为字符串作为folder
        folder_path = os.path.join(self.partnet_path, folder)

        if os.path.isdir(folder_path):
            meta_path = os.path.join(folder_path, "meta.json")
            if os.path.exists(meta_path):
                category = self._get_category(meta_path)
                # 将path和category添加到item中
                item.path = folder_path
                item.category = category
        return item

    def _process_items(self) -> Generator[SimpleNamespace, None, None]:
        for item in self.input_generator:
            processed_item = self._process_item(item)
            yield processed_item

    def __iter__(self):
        return self._process_items()


class DataLoader:
    """
    这个类的功能是读取储存好的训练数据，并yield。

    初始化参数:
    - data_path: 包含训练数据的文件夹路径

    从输入参数的data_path中读取每一个文件的名字，然后用NamespaceHDF5读取
    """

    def __init__(self, data_path: str, data_list=None, shuffle=False):
        self.data_path = data_path
        self.data_list = data_list
        self.shuffle = shuffle

    def __iter__(self) -> Generator[SimpleNamespace, None, None]:
        file_name_list = os.listdir(self.data_path)
        if self.shuffle:
            random.shuffle(file_name_list)
        for file_name in file_name_list:
            if file_name.endswith(".hdf5"):
                if (
                    self.data_list is not None
                    and file_name.split(".")[0] not in self.data_list
                ):
                    continue
                file_path = os.path.join(self.data_path, file_name)
                ns_hdf5 = NamespaceHDF5(file_path)
                loaded_ns = ns_hdf5.load_namespace()
                yield loaded_ns


class ManifoldPlusPipeline:
    """
    这是一个生成器,用于处理从输入生成器获取的数据,并调用ManifoldPlus程序。

    初始化参数:
    - cfg: 配置对象,其中包含ManifoldPlus程序的路径 (cfg.data.manifold_plus_path)
    - input_generator: 一个生成器,生成包含data.path的SimpleNamespace对象

    调用方法:
    - __iter__(): 迭代输入生成器,处理数据并调用ManifoldPlus程序,yield处理后的SimpleNamespace对象
    """

    def __init__(
        self,
        manifold_plus_path: str,
        input_generator: Generator[SimpleNamespace, None, None],
        obj_folder="obj",
        **cfg,
    ):
        self.manifold_plus_path = manifold_plus_path
        self.input_generator = input_generator
        self.obj_folder = obj_folder
        self.runner = ManifoldPlusRunner(self.manifold_plus_path)

    def __len__(self):
        """
        返回input_generator的长度,如果没有__len__方法,则返回-1
        """
        try:
            return len(self.input_generator)
        except TypeError:
            return 0

    def __iter__(self):
        for data in self.input_generator:
            data_path = data.path
            textured_objs_path = os.path.join(data_path, self.obj_folder)
            complete_obj_path = os.path.join(data_path, self.obj_folder, "complete.obj")
            manifold_obj_path = os.path.join(data_path, self.obj_folder, "manifold.obj")

            if not os.path.exists(manifold_obj_path):
                self._combine_shapes(textured_objs_path, complete_obj_path)
                self.runner.run(complete_obj_path, manifold_obj_path)
            yield data

    def _combine_shapes(self, textured_objs_path: str, complete_obj_path: str):
        """
        将textured_objs_path中的所有形状组合成一个complete.obj文件

        参数:
        - textured_objs_path: 包含形状的文件夹路径
        - complete_obj_path: 完整形状的输出文件路径
        """
        meshes = []
        for obj_file in os.listdir(textured_objs_path):
            obj_file_path = os.path.join(textured_objs_path, obj_file)
            if os.path.isfile(obj_file_path) and obj_file_path.endswith(".obj"):
                mesh = trimesh.load(obj_file_path)
                meshes.append(mesh)

        if meshes:
            combined_mesh = trimesh.util.concatenate(meshes)
            combined_mesh.export(complete_obj_path)


class FilterPipeline:
    """
    这是一个过滤器生成器,用于过滤输入生成器中的数据。

    初始化参数:
    - input_generator: 一个生成器,生成包含数据的对象
    - filter_func: 一个判断函数,对于每个数据,传入并获取bool返回值,如果为False则跳过这个数据
    """

    def __init__(
        self,
        input_generator: Generator[Any, None, None],
        filter_func: Callable[[Any], bool],
        **cfg,
    ):
        self.input_generator = input_generator
        self.filter_func = filter_func
        self.cfg=cfg

    def __iter__(self):
        for data in self.input_generator:
            if self.filter_func(data,**self.cfg):
                yield data
            else:
                []
                #print(data.id,"Filterfail",self.filter_func)

    @staticmethod
    def filter_low_volume(input_generator: Generator[Any, None, None]):
        return FilterPipeline(input_generator, shape_classification)
    
    @staticmethod
    def filter_no_valid_shape(input_generator: Generator[Any, None, None],partnet_path="D:\data\data_v0", required_categories=None, black=False):
        return FilterPipeline(input_generator, is_valid_shape,partnet_path=partnet_path, required_categories=required_categories, black=black)


class ExtractPipeline:
    """
    这是一个用于提取信息的流水线,用于向流水线的data添加字段

    初始化参数:
    - input_generator: 一个生成器,生成包含数据的对象
    - extract_func: 一个提取信息函数,对于每个数据,传入并获取返回值,如果为None则跳过这个数据
    - name: 指明要加入data中的数据叫什么名字
    """

    def __init__(
        self,
        input_generator: Generator[Any, None, None],
        extract_func: Callable[[Any], Any],
        name: str,
    ):
        self.input_generator = input_generator
        self.extract_func = extract_func
        self.name = name

    def __iter__(self):
        for data in self.input_generator:
            extracted_value = self.extract_func(data)
            if extracted_value is not None:
                setattr(data, self.name, extracted_value)
                yield data

    @staticmethod
    def extract_tree(input_generator: Generator[Any, None, None]):
        return ExtractPipeline(input_generator, extract_parts, "parts")


class TreeExtractPipeline:
    """
    这是一个生成器流水线,用于提取输入流水线形状中的所有部件的AABB和OBB包围盒。

    初始化参数:
    - input_generator: 一个生成器,生成包含数据的对象
    - obj_folder: obj文件所在的文件夹名称
    - tree_name: 树结构文件的名称
    - max_parts_length: 最大部件长度
    - min_volume: 最小体积要求
    """

    def __init__(
        self,
        input_generator: Generator[SimpleNamespace, None, None],
        obj_folder: str,
        tree_name: str,
        max_parts_length: int,
        min_volume: float,
        min_parts_length: int,
        min_part_rate: int,
        part_resolution: int,
        **cfg,
    ):
        self.input_generator = input_generator
        self.obj_folder = obj_folder
        self.tree_name = tree_name
        self.max_parts_length = max_parts_length
        self.min_parts_length = min_parts_length
        self.min_volume = min_volume
        self.min_part_rate = min_part_rate
        self.part_resolution = part_resolution

    def __iter__(self):
        for data in self.input_generator:
            tree_path = os.path.join(data.path, self.tree_name)
            with open(tree_path, "r", encoding="utf-8") as tree_file:
                tree_data = json.load(tree_file)
                populate_objs_in_tree(tree_data[0])

                # 计算所有叶子节点的 OBB 体积并存储
                self.leaf_meshes = self._load_meshes(tree_data, data.path)
                self.leaf_volumes = self._compute_volumes()

                parts_names = self._extract_components_with_volume_check(
                    tree_data,
                )

            if (
                len(parts_names) > self.max_parts_length
                or len(parts_names) < self.min_parts_length
            ):
                print(data.id,f"len(parts_names)={len(parts_names)}")
                continue

            if not hasattr(data, "parts"):
                data.parts = SimpleNamespace()

            _, data.parts.obb = self._calculate_parts_bounding_boxes(
                data.path, parts_names
            )
            if (np.prod(data.parts.obb.extents, axis=-1) > self.min_volume).all():
                yield data
            else:
                print(data.id,f"size",*np.round(np.prod(data.parts.obb.extents, axis=-1),4))

    def _load_meshes(self, tree_data, data_path: str) -> Dict[str, trimesh.Trimesh]:
        """
        加载所有叶子节点的mesh，并存储在字典中。
        返回的字典格式为：{obj_name: mesh}
        """
        leaf_meshes = {}

        def recurse(node):
            if "children" in node:
                for child in node["children"]:
                    recurse(child)
            else:
                for obj in node["objs"]:
                    obj_path = os.path.join(data_path, self.obj_folder, f"{obj}.obj")
                    if os.path.exists(obj_path):
                        mesh = trimesh.load(obj_path)
                        leaf_meshes[obj] = mesh

        recurse(tree_data[0])
        return leaf_meshes

    def _compute_volumes(self) -> Dict[str, float]:
        """
        根据加载的mesh计算每个叶子节点的OBB体积，并存储在字典中。
        返回的字典格式为：{obj_name: volume}
        """
        leaf_volumes = {}

        for obj_name, mesh in self.leaf_meshes.items():
            if mesh:
                try:
                    obb_extents = mesh.bounding_box_oriented.primitive.extents
                    obb_volume = np.prod(obb_extents)
                    leaf_volumes[obj_name] = obb_volume
                except Exception as e:
                    print(
                        f"Failed to calculate bounding boxes for obj: {obj_name}. Error: {e}"
                    )
                    leaf_volumes[obj_name] = 0.0
        return leaf_volumes

    def _extract_components_with_volume_check(
        self,
        tree_data,
    ) -> List[List[str]]:
        """
        从树结构中提取部件名称，确保只有在子树所有叶子节点OBB体积总和大于最小体积时才展开节点。
        """
        parts_names = []

        def calculate_subtree_volume(node) -> float:
            if "children" in node:
                total_volume = 0
                for child in node["children"]:
                    total_volume += calculate_subtree_volume(child)
                return total_volume
            else:
                return sum(self.leaf_volumes[obj] for obj in node["objs"])

        def recurse(node):
            _, node_volume = self._calculate_bounding_boxes_and_volume(node["objs"])
            subtree_volume = calculate_subtree_volume(node)
            if node_volume == 0:
                parts_names.append(node["objs"])
                return
            if "children" in node:
                if subtree_volume / node_volume < self.min_part_rate:
                    for child in node["children"]:
                        recurse(child)
                else:
                    for child in node["children"]:
                        _, child_volume = self._calculate_bounding_boxes_and_volume(
                            child["objs"]
                        )
                        subtree_volume = calculate_subtree_volume(child)
                        if subtree_volume < self.min_volume:
                            parts_names.append(node["objs"])
                            break
                    else:
                        for child in node["children"]:
                            recurse(child)

            else:
                parts_names.append(node["objs"])

        recurse(tree_data[0])
        return parts_names

    def _calculate_bounding_boxes_and_volume(self, part_names: List[str]):
        """
        计算给定部件的AABB和OBB，并返回包围盒及其OBB体积。
        现在从self.leaf_data中获取mesh而不是重新加载.obj文件。
        """
        part_meshes = []
        for obj_name in part_names:
            if obj_name in self.leaf_meshes:
                mesh = self.leaf_meshes[obj_name]
                if mesh:
                    part_meshes.append(mesh)

        if part_meshes:
            combined_mesh = trimesh.util.concatenate(part_meshes)
            try:
                # 获取AABB的transform和extents
                aabb_extents = combined_mesh.bounding_box.primitive.extents

                # 获取OBB的transform和extents
                obb_transform = combined_mesh.bounding_box_oriented.primitive.transform
                obb_extents = combined_mesh.bounding_box_oriented.primitive.extents

                # 计算OBB体积
                obb_volume = np.prod(obb_extents)

                # 使用SimpleNamespace保存transform和extents
                aabb_namespace = SimpleNamespace(extents=aabb_extents)
                obb_namespace = SimpleNamespace(
                    transform=obb_transform, extents=obb_extents
                )

                return obb_namespace, obb_volume

            except Exception as e:
                print(
                    f"Failed to calculate bounding boxes for part: {part_names}. Error: {e}"
                )
                return None, 0.0
        return None, 0.0

    def _calculate_parts_bounding_boxes(
        self, data_path: str, parts_names: List[List[str]]
    ) -> SimpleNamespace:
        """
        计算每个部件的AABB和OBB，并将结果拼接成单个SimpleNamespace返回。
        """

        # aabb_extents = []
        obb_transforms = []
        obb_extents = []

        for part in parts_names:
            obb_namespace, _ = self._calculate_bounding_boxes_and_volume(part)
            if obb_namespace:
                # aabb_extents.append(obb_namespace.extents)
                obb_transforms.append(obb_namespace.transform)
                obb_extents.append(obb_namespace.extents)

        # 使用 numpy 在新的维度上拼接
        # aabb_extents_np = np.stack(aabb_extents, axis=0) if aabb_extents else None
        obb_transforms_np = np.stack(obb_transforms, axis=0) if obb_transforms else None
        obb_extents_np = np.stack(obb_extents, axis=0) if obb_extents else None

        # 创建一个SimpleNamespace来存储拼接后的结果
        obb = SimpleNamespace(
            transform=obb_transforms_np,
            extents=obb_extents_np,
        )
        # aabb = SimpleNamespace(
        #     extents=aabb_extents_np,
        # )

        return None, obb


class ShapeReaderPipeline:
    """
    这是一个形状读取流水线，用于读取输入data.path对应的面片数据，
    并将其写入data.mesh后返回。
    从os.path.join(data_path, self.obj_folder)读取manifold.obj
    """

    def __init__(
        self,
        input_generator: Generator[SimpleNamespace, None, None],
        obj_folder: str,
        **cfg,
    ):
        self.input_generator = input_generator
        self.obj_folder = obj_folder

    def __iter__(self):
        for data in self.input_generator:
            data_path = data.path
            manifold_obj_path = os.path.join(data_path, self.obj_folder, "manifold.obj")
            if os.path.exists(manifold_obj_path):
                data.mesh = self._read_mesh(manifold_obj_path)
                yield data

    def _read_mesh(self, obj_path: str) -> trimesh.Trimesh:
        """
        读取obj文件并返回trimesh对象
        """
        return trimesh.load(obj_path)


class DataCleanPipeline:
    """
    这是一个数据清理流水线。

    初始化参数：
    - input_generator: 一个生成器，用于生成包含数据的对象。
    - template_data: 一个模板SimpleNamespace对象，仅用于指示应保留哪些属性。模板data数据仅仅是嵌套的SimpleNamespace，所有数据都是None。

    从input_generator读取数据，仅保留模板data中存在的属性对应的数据，然后将清理后的数据yield出去。
    """

    def __init__(
        self,
        input_generator: Generator[SimpleNamespace, None, None],
        template_data: SimpleNamespace,
    ):
        self.input_generator = input_generator
        self.template_data = template_data

    def __iter__(self):
        for data in self.input_generator:
            cleaned_data = self._clean_data(data, self.template_data)
            yield cleaned_data

    def _clean_data(
        self, data: SimpleNamespace, template: SimpleNamespace
    ) -> SimpleNamespace:
        cleaned_data = SimpleNamespace()
        for key, value in template.__dict__.items():
            if hasattr(data, key):
                attr = getattr(data, key)
                if isinstance(value, SimpleNamespace):
                    setattr(cleaned_data, key, self._clean_data(attr, value))
                else:
                    setattr(cleaned_data, key, attr)
            else:
                print("clean_error")
                []
        return cleaned_data

    @staticmethod
    def default(input_generator):
        data = SimpleNamespace(
            id=None,
            parts=SimpleNamespace(
                center=None,
                obb_voxel=None,
                voxel_indices=None,
                voxel_labels=None,
                patch_size=None,
                obb=SimpleNamespace(extents=None, transform=None),
                #aabb=SimpleNamespace(extents=None),
            ),
            sample=SimpleNamespace(
                global_sample=SimpleNamespace(points=None, labels=None, sdf=None),
                near_a=SimpleNamespace(points=None, labels=None, sdf=None),
                near_b=SimpleNamespace(points=None, labels=None, sdf=None),
            ),
        )
        return DataCleanPipeline(input_generator, data)
