import os
import pickle
import torch
from shutil import copyfile, move
from PIL import Image
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import (
    Tuple,
    List,
    Union,
    Callable,
    Type,
    Iterator,
    Dict,
    Set,
    Optional,
    Any,
    Sized,
    Iterable,
)

def image_to_display(img) -> np.ndarray:
    if type(img) is str:
        img = Image.open(str(img))
    if type(img) is not np.array:
        img = np.array(img)
    return img


def imshow(img, title: Optional[str] = None):
    img = image_to_display(img)
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close('all')


def load_image(path: str, color_type: str = 'RGB') -> np.ndarray:
    for suffix in ('.png', '.jpg'):
        path_ = add_suffix(path, suffix)
        if os.path.isfile(path_):
            path = path_
            break
    image = Image.open(path).convert(color_type)
    return np.array(image)


def save_image(image: Union[np.ndarray, Image.Image], path: str):
    if type(image) is np.ndarray:
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        image = Image.fromarray(image)
    init_folders(path)
    image.save(path)


def split_path(path: str) -> List[str]:
    extension = os.path.splitext(path)[1]
    dir_name, name = os.path.split(path)
    name = name[: len(name) - len(extension)]
    return [dir_name, name, extension]


def init_folders(*folders):
    # if const.DEBUG:
    #     return
    for f in folders:
        dir_name = os.path.dirname(f)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)


def is_file(path: str):
    return os.path.isfile(path)


def add_suffix(path: str, suffix: str) -> str:
    if len(path) < len(suffix) or path[-len(suffix):] != suffix:
        path = f'{path}{suffix}'
    return path




def path_init(suffix: str, path_arg_ind: int, is_save: bool):

    def wrapper(func):

        def do(*args, **kwargs):
            path = add_suffix(args[path_arg_ind], suffix)
            if is_save:
                init_folders(path)
            args = [args[i] if i != path_arg_ind else path for i in range(len(args))]
            return func(*args, **kwargs)

        return do

    return wrapper

@path_init('.pkl', 1, True)
def save_pickle(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

@path_init('.pkl', 0, False)
def load_pickle(path: str):
    data = None
    if os.path.isfile(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except ValueError:
            with open(path, 'rb') as f:
                data = pickle5.load(f)
    return data


# @path_init('.txt', 0, False)
# def load_txt(path: str) -> List[str]:
#     data = []
#     if os.path.isfile(path):
#         with open(path, 'r') as f:
#             for line in f:
#                 data.append(line.strip())
#     return data


# def load_points(path: str) -> torch.Tensor:
#     path = add_suffix(path, '.pts')
#     points = [int_b(num) for num in load_txt(path)]
#     return torch.tensor(points, dtype=torch.int64)


# def save_txt(array, path: str):
#     if const.DEBUG:
#         return
#     path_ = add_suffix(path, '.txt')
#     with open(path_, 'w') as f:
#         for i, num in enumerate(array):
#             f.write(f'{num}{" " if i < len(array) - 1 else ""}')


# def move_file(src: str, dest: str):
#     if const.DEBUG:
#         return
#     if os.path.isfile(src):
#         move(src, dest)
#         return True
#     return False


# @path_init('.json', 1, True)
# def save_json(obj, path: str):
#     with open(path, 'w') as f:
#         json.dump(obj, f, indent=4)


def collect(root: str, *suffix, prefix='') -> List[List[str]]:
    if os.path.isfile(root):
        folder = os.path.split(root)[0] + '/'
        extension = os.path.splitext(root)[-1]
        name = root[len(folder): -len(extension)]
        paths = [[folder, name, extension]]
    else:
        paths = []
        root = add_suffix(root, '/')
        if not os.path.isdir(root):
            print(f'Warning: trying to collect from {root} but dir isn\'t exist')
        else:
            p_len = len(prefix)
            for path, _, files in os.walk(root):
                for file in files:
                    file_name, file_extension = os.path.splitext(file)
                    p_len_ = min(p_len, len(file_name))
                    if file_extension in suffix and file_name[:p_len_] == prefix:
                        paths.append((f'{add_suffix(path, "/")}', file_name, file_extension))
            paths.sort(key=lambda x: os.path.join(x[1], x[2]))
    return paths


# def delete_all(root:str, *suffix: str):
#     if const.DEBUG:
#         return
#     paths = collect(root, *suffix)
#     for path in paths:
#         os.remove(''.join(path))


# def delete_single(path: str) -> bool:
#     if os.path.isfile(path):
#         os.remove(path)
#         return True
#     return False


# def colors_to_colors(colors: COLORS, mesh: T_Mesh) -> torch.Tensor:
#     if type(colors) is not torch.Tensor:
#         if type(colors) is V:
#             colors = torch.from_numpy(colors).long()
#         else:
#             colors = torch.tensor(colors, dtype=torch.int64)
#     if colors.max() > 1:
#         colors = colors.float() / 255
#     if colors.dim() == 1:
#         colors = colors.unsqueeze(int(colors.shape[0] != 3)).expand_as(mesh[0])
#     return colors


def load_mesh(file_name: str, dtype: Union[torch.Tensor, np.array] = torch.Tensor,
              device = "cpu") -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[np.ndarray, np.ndarray], torch.Tensor, Tuple[torch.Tensor, List[List[int]]]]:

    def off_parser():
        header = None

        def parser_(clean_line: list):
            nonlocal header
            if not clean_line:
                return False
            if len(clean_line) == 3 and not header:
                header = True
            elif len(clean_line) == 3:
                return 0, 0, float
            elif len(clean_line) > 3:
                return 1, -int(clean_line[0]), int

        return parser_

    def obj_parser(clean_line: list):
        nonlocal is_quad
        if not clean_line:
            return False
        elif clean_line[0] == 'v':
            return 0, 1, float
        elif clean_line[0] == 'f':
            is_quad = is_quad or len(clean_line) != 4
            return 1, 1, int
        return False

    def fetch(lst: list, idx: int, dtype: type):
        uv_vs_ids = None
        if '/' in lst[idx]:
            lst = [item.split('/') for item in lst[idx:]]
            lst = [item[0] for item in lst]
            idx = 0
        face_vs_ids = [dtype(c.split('/')[0]) for c in lst[idx:]]
        if dtype is float and len(face_vs_ids) > 3:
            face_vs_ids = face_vs_ids[:3]
        return face_vs_ids, uv_vs_ids

    def load_from_txt(parser) -> Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]:
        mesh_ = [[], []]
        with open(file_name, 'r') as f:
            for line in f:
                clean_line = line.strip().split()
                info = parser(clean_line)
                if not info:
                    continue
                data = fetch(clean_line, info[1], info[2])
                mesh_[info[0]].append(data[0])
        if is_quad:
            faces = mesh_[1]
            for face in faces:
                for i in range(len(face)):
                    face[i] -= 1
        else:
            faces = torch.tensor(mesh_[1], dtype=torch.int64)
            if len(faces) > 0 and faces.min() != 0:
                faces -= 1
        mesh_ = torch.tensor(mesh_[0], dtype=torch.float32), faces
        return mesh_

    for suffix in ['.obj', '.off', '.ply']:
        file_name_tmp = add_suffix(file_name, suffix)
        if os.path.isfile(file_name_tmp):
            file_name = file_name_tmp
            break

    is_quad = False
    name, extension = os.path.splitext(file_name)
    if extension == '.obj':
        mesh = load_from_txt(obj_parser)
    elif extension == '.off':
        mesh = load_from_txt(off_parser())
    # elif extension == '.ply':
    #     mesh = load_ply(file_name)
    else:
        raise ValueError(f'mesh file {file_name} is not exist or not supported')
    if type(mesh[1]) is torch.Tensor and not ((mesh[1] >= 0) * (mesh[1] < mesh[0].shape[0])).all():
        print(f"err: {file_name}")
    assert type(mesh[1]) is not torch.Tensor or ((mesh[1] >= 0) * (mesh[1] < mesh[0].shape[0])).all()
    if dtype is np.array:
        mesh = mesh[0].numpy(), mesh[1].numpy()
    elif device != "cpu":
        mesh = mesh[0].to(device), mesh[1].to(device)
    if len(mesh[1]) == 0 and len(mesh[0]) > 0:
        return mesh[0]
    return mesh

@path_init('.obj', 1, True)
def export_mesh(mesh, file_name: str,
                colors = None, normals = None, edges=None, spheres=None):
    # return
    if type(mesh) is not tuple and type(mesh) is not list:
        mesh = mesh, None
    vs, faces = mesh
    if vs.shape[1] < 3:
        vs = torch.cat((vs, torch.zeros(len(vs), 3 - vs.shape[1], device=vs.device)), dim=1)
    if colors is not None:
        colors = colors_to_colors(colors, mesh)
    if not os.path.isdir(os.path.dirname(file_name)):
        return
    if faces is not None:
        if type(faces) is torch.Tensor:
            faces: torch.Tensor = faces + 1
            faces_lst = faces.tolist()
        else:
            faces_lst_: List[List[int]] = faces
            faces_lst = []
            for face in faces_lst_:
                faces_lst.append([face[i] + 1 for i in range(len(face))])
    with open(file_name, 'w') as f:
        for vi, v in enumerate(vs):
            if colors is None or colors[vi, 0] < 0:
                v_color = ''
            else:
                v_color = ' %f %f %f' % (colors[vi, 0].item(), colors[vi, 1].item(), colors[vi, 2].item())
            f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], v_color))
        if normals is not None:
            for n in normals:
                f.write("vn %f %f %f\n" % (n[0], n[1], n[2]))
        if faces is not None:
            for face in faces_lst:
                face = [str(f) for f in face]
                f.write(f'f {" ".join(face)}\n')
        if edges is not None:
            for edges_id in range(edges.shape[0]):
                f.write(f'\ne {edges[edges_id][0].item():d} {edges[edges_id][1].item():d}')
        if spheres is not None:
            for sphere_id in range(spheres.shape[0]):
                f.write(f'\nsp {spheres[sphere_id].item():d}')

@path_init('.txt', 1, True)
def export_gmm(gmm, file_name: str, included: Optional[List[int]] = None):
    if included is None:
        included = [1] * gmm[0].shape[2]
    mu,p, phi, eigen = [tensor.flatten().cpu() for tensor in gmm]
    # phi = phi.softmax(0)
    with open(file_name, 'w') as f:
        for tensor in (mu,eigen, p):
            tensor_str = [f'{number:.5f}' for number in tensor.tolist()]
            f.write(f"{' '.join(tensor_str)}\n")
        list_str = [f'{number:d}' for number in included]
        f.write(f"{' '.join(list_str)}\n")


# @path_init('.xyz', 1, True)
# def export_xyz(pc: torch.Tensor, path: str, normals: Optional[torch.Tensor] = None):
#     pc = pc.tolist()
#     if normals is not None:
#         normals = normals.tolist()
#     with open(path, 'w') as f:
#         for i in range(len(pc)):
#             x, y, z = pc[i]
#             f.write(f'{x} {y} {z}')
#             if normals is not None:
#                 x, y, z = normals[i]
#                 f.write(f' {x} {y} {z}')
#             if i < len(pc) - 1:
#                 f.write('\n')


@path_init('.txt', 0, False)
def load_gmm(path, as_np: bool = False, device  = "cpu"):
    parsed = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f]
    for i, line in enumerate(lines):
        line = line.split(" ")
        arr = [float(item) for item in line]
        if as_np:
            arr = np.array(arr)
        else:
            arr = torch.tensor(arr, device=device)
        if 0 <= i <= 1:
            arr = arr.reshape((-1, 3))
            # swap = arr[:, 2].copy()
            # arr[:, 2] = arr[:, 1]
            # arr[:, 1] = swap
        elif i == 2:
            arr = arr.reshape((-1, 3, 3))
            # arr = arr.transpose(0, 2, 1)
        elif i == 3:
            if as_np:
                arr = arr.astype(np.bool_)
            else:
                arr = arr.bool()
        parsed.append(arr)
    phi=np.ones_like(np.prod(parsed[1],axis=-1))
    phi/=phi.sum()
    parsed=[phi]+parsed
    return parsed

def save_all(z, mesh, gmm, root_path: str, name_prefix: str):
    """
    使用时间戳生成文件名，并保存 z、mesh 和 gmm。
    - obj: 要保存的对象，将以 `.pkl` 格式保存
    - mesh: 要保存的 mesh 对象，将以 `.z` 格式保存
    - gmm: 要保存的 GMM 对象，将以 `.txt` 格式保存
    - root_path: 文件保存的根路径
    - name_prefix: 文件名前缀
    """
    # 生成时间戳后的文件名
    timestamp = time.strftime("%m_%d-%H_%M")
    file_name = f"{name_prefix}_{timestamp}"

    # 保存 obj, mesh, gmm
    save_pickle(z, f"{root_path}/{file_name}")
    export_mesh(mesh, f"{root_path}/{file_name}")
    export_gmm(gmm, f"{root_path}/{file_name}")

# @path_init('.txt', 1, True)
# def export_list(lst: List[Any], path: str):
#     with open(path, "w") as f:
#         for i in range(len(lst)):
#             f.write(f'{lst[i]}\n')


# @path_init('.obj', 1, True)
# def export_mesh(mesh: Union[V_Mesh, T_Mesh, torch.Tensor, Tuple[torch.Tensor, List[List[int]]]], file_name: str,
#                 colors: Optional[COLORS] = None, normals: TN = None, edges=None, spheres=None):
#     # return
#     if type(mesh) is not tuple and type(mesh) is not list:
#         mesh = mesh, None
#     vs, faces = mesh
#     if vs.shape[1] < 3:
#         vs = torch.cat((vs, torch.zeros(len(vs), 3 - vs.shape[1], device=vs.device)), dim=1)
#     if colors is not None:
#         colors = colors_to_colors(colors, mesh)
#     if not os.path.isdir(os.path.dirname(file_name)):
#         return
#     if faces is not None:
#         if type(faces) is torch.Tensor:
#             faces: torch.Tensor = faces + 1
#             faces_lst = faces.tolist()
#         else:
#             faces_lst_: List[List[int]] = faces
#             faces_lst = []
#             for face in faces_lst_:
#                 faces_lst.append([face[i] + 1 for i in range(len(face))])
#     with open(file_name, 'w') as f:
#         for vi, v in enumerate(vs):
#             if colors is None or colors[vi, 0] < 0:
#                 v_color = ''
#             else:
#                 v_color = ' %f %f %f' % (colors[vi, 0].item(), colors[vi, 1].item(), colors[vi, 2].item())
#             f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], v_color))
#         if normals is not None:
#             for n in normals:
#                 f.write("vn %f %f %f\n" % (n[0], n[1], n[2]))
#         if faces is not None:
#             for face in faces_lst:
#                 face = [str(f) for f in face]
#                 f.write(f'f {" ".join(face)}\n')
#         if edges is not None:
#             for edges_id in range(edges.shape[0]):
#                 f.write(f'\ne {edges[edges_id][0].item():d} {edges[edges_id][1].item():d}')
#         if spheres is not None:
#             for sphere_id in range(spheres.shape[0]):
#                 f.write(f'\nsp {spheres[sphere_id].item():d}')

# @path_init('.ply', 1, True)
# def export_ply(mesh: T_Mesh, path: str, colors: torch.Tensor):
#     colors = colors_to_colors(colors, mesh)
#     colors = (colors * 255).long()
#     vs, faces = mesh
#     vs = vs.clone()
#     swap = vs[:, 1].clone()
#     vs[:, 1] = vs[:, 2]
#     vs[:, 2] = swap
#     min_cor, max_cor= vs.min(0)[0], vs.max(0)[0]
#     vs = vs - ((min_cor + max_cor) / 2)[None, :]
#     vs = vs / vs.max()
#     vs[:, 2] = vs[:, 2] - vs[:, 2].min()
#     num_vs = vs.shape[0]
#     num_faces = faces.shape[0]
#     with open(path, 'w') as f:
#         f.write(f'ply\nformat ascii 1.0\n'
#                 f'element vertex {num_vs:d}\nproperty float x\nproperty float y\nproperty float z\n'
#                 f'property uchar red\nproperty uchar green\nproperty uchar blue\n'
#                 f'element face {num_faces:d}\nproperty list uchar int vertex_indices\nend_header\n')
#         for vi, v in enumerate(vs):
#             color = f'{colors[vi, 0].item():d} {colors[vi, 1].item():d} {colors[vi, 2].item():d}'
#             f.write(f'{v[0].item():f} {v[1].item():f} {v[2].item():f} {color}\n')
#         for face in faces:
#             f.write(f'3 {face[0].item():d} {face[1].item():d} {face[2].item():d}\n')


# @path_init('.ply', 0, False)
# def load_ply(path: str):
#     import plyfile
#     plydata = plyfile.PlyData.read(path)
#     vertices = plydata.elements[0].data
#     vertices = [[float(item[0]), float(item[1]), float(item[2])] for item in vertices]
#     vertices = torch.tensor(vertices)
#     faces = plydata.elements[1].data
#     faces = [[int(item[0][0]), int(item[0][1]), int(item[0][2])] for item in faces]
#     faces = torch.tensor(faces)
#     return vertices, faces


# @path_init('', 1, True)
# def save_model(model: Union[Optimizer, nn.Module], model_path: str):
#     if const.DEBUG:
#         return
#     init_folders(model_path)
#     torch.save(model.state_dict(), model_path)


# def load_model(model: Union[Optimizer, nn.Module], model_path: str, device: D, verbose: bool = False):
#     if os.path.isfile(model_path):
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         if verbose:
#             print(f'loading {type(model).__name__} from {model_path}')
#     elif verbose:
#         print(f'init {type(model).__name__}')
#     return model


# def measure_time(func, num_iters: int, *args):
#     start_time = time.time()
#     for i in range(num_iters):
#         func(*args)
#     total_time = time.time() - start_time
#     avg_time = total_time / num_iters
#     print(f"{str(func).split()[1].split('.')[-1]} total time: {total_time}, average time: {avg_time}")


# def get_time_name(name: str, format_="%m_%d-%H_%M") -> str:
#     return f'{name}_{time.strftime(format_)}'


# @path_init('.txt', 0, False)
# def load_shapenet_seg(path: str) -> TS:
#     labels, vs = [], []
#     with open(path, 'r') as f:
#         for line in f:
#             data = line.strip().split()
#             vs.append([float(item) for item in data[:3]])
#             labels.append(int(data[-1].split('.')[0]))
#     return torch.tensor(vs, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


# @path_init('.json', 0, False)
# def load_json(path: str):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     return data
