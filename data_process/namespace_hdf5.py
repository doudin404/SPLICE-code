import h5py
import numpy as np
from types import SimpleNamespace


class NamespaceHDF5:
    def __init__(self, filepath):
        self.filepath = filepath

    def namespace_to_dict(self, namespace):
        if isinstance(namespace, SimpleNamespace):
            return {k: self.namespace_to_dict(v) for k, v in namespace.__dict__.items()}
        elif isinstance(namespace, list):
            return [self.namespace_to_dict(item) for item in namespace]
        else:
            return namespace

    def save_dict_to_hdf5(self, group, dict_data):
        for key, value in dict_data.items():
            try:
                if isinstance(value, dict):
                    subgroup = group.create_group(key)
                    subgroup.attrs["type"] = "dict"
                    self.save_dict_to_hdf5(subgroup, value)
                elif isinstance(value, list):
                    list_group = group.create_group(key)
                    list_group.attrs["type"] = "list"
                    for i, item in enumerate(value):
                        try:
                            if isinstance(item, np.ndarray):
                                sublist_group = list_group.create_dataset(str(i), data=item)
                                sublist_group.attrs["type"] = "np.ndarray"
                            else:
                                sublist_group = list_group.create_group(str(i))
                                self.save_dict_to_hdf5(sublist_group, item)
                        except TypeError:
                            print(f"Skipping unsupported list item type: {type(item)}")
                else:
                    dataset = group.create_dataset(key, data=value)
                    dataset.attrs["type"] = type(value).__name__
            except TypeError as e:
                print(
                    f"Skipping key '{key}' with unsupported data type: {type(value)}. Error: {e}"
                )

    def save_namespace(self, namespace):
        data_dict = self.namespace_to_dict(namespace)
        with h5py.File(self.filepath, "w") as f:
            self.save_dict_to_hdf5(f, data_dict)
        #print("Data has been successfully saved to the HDF5 file.")

    def dict_to_namespace(self, dict_data):
        if isinstance(dict_data, dict):
            return SimpleNamespace(
                **{k: self.dict_to_namespace(v) for k, v in dict_data.items()}
            )
        elif isinstance(dict_data, list):
            return [self.dict_to_namespace(item) for item in dict_data]
        else:
            return dict_data

    def load_dict_from_hdf5(self, group):
        result = {}
        for key, item in group.items():
            item_type = item.attrs.get("type")
            if isinstance(item, h5py.Group):
                if item_type == "list":
                    sublist = [
                        self.load_dict_from_hdf5(subitem) if isinstance(subitem, h5py.Group) else subitem[()]
                        for subitem in sorted(item.values(), key=lambda x: int(x.name.split('/')[-1]))
                    ]
                    result[key] = sublist
                else:
                    result[key] = self.load_dict_from_hdf5(item)
            else:
                if item_type == "np.ndarray":
                    result[key] = item[()]
                elif item_type ==  "str":
                    result[key] = item[()].decode('utf-8')
                else:
                    result[key] = item[()]

        return result

    def load_namespace(self):
        with h5py.File(self.filepath, "r") as f:
            data_dict = self.load_dict_from_hdf5(f)
        return self.dict_to_namespace(data_dict)


# 示例用法
if __name__ == "__main__":
    ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2, d=[3, 4, SimpleNamespace(e=5)]))
    ns_hdf5 = NamespaceHDF5("example.h5")
    ns_hdf5.save_namespace(ns)
    loaded_ns = ns_hdf5.load_namespace()
    print(loaded_ns)


# # 使用示例
# data = SimpleNamespace(
#     path='C:/code/noodels/data/',
#     category='Table',
#     id='19179',
#     parts=SimpleNamespace(
#         aabb=[
#             SimpleNamespace(
#                 transform=np.array([
#                     [1.0, 0.0, 0.0, -0.0071835],
#                     [0.0, 1.0, 0.0, -0.263335],
#                     [0.0, 0.0, 1.0, 0.073062],
#                     [0.0, 0.0, 0.0, 1.0]
#                 ]),
#                 extents=np.array([1.125453, 0.488728, 1.081574])
#             )
#             # 可以继续添加更多SimpleNamespace
#         ]
#     )
# )

# # 保存namespace到HDF5文件
# hdf5_manager = NamespaceHDF5('data.hdf5')
# hdf5_manager.save_namespace(data)

# # 从HDF5文件加载namespace
# loaded_data = hdf5_manager.load_namespace()
# print(loaded_data)
