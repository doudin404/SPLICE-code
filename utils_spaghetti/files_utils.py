import os
import time
import json
from shutil import copyfile, move
from typing import Tuple, List, Union, Optional, Any

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from PIL import Image
import matplotlib.pyplot as plt

import constants as const

def image_to_display(img) -> np.ndarray:
    if type(img) is str:
        img = Image.open(str(img))
    if type(img) is not np.ndarray:
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
    if const.DEBUG:
        return
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


def remove_suffix(path: str, suffix: str) -> str:
    if len(path) > len(suffix) and path[-len(suffix):] == suffix:
        path = path[:-len(suffix)]
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


def copy_file(src: str, dest: str, force=False):
    if const.DEBUG:
        return
    if os.path.isfile(src):
        if force or not os.path.isfile(dest):
            copyfile(src, dest)
            return True
        else:
            print("Destination file already exist. To override, set force=True")
    return False


def load_image(path: str, color_type: str = 'RGB') -> np.ndarray:
    for suffix in ('.png', '.jpg'):
        path_ = add_suffix(path, suffix)
        if os.path.isfile(path_):
            path = path_
            break
    image = Image.open(path).convert(color_type)
    return np.array(image)


@path_init('.png', 1, True)
def save_image(image: np.ndarray, path: str):
    if type(image) is np.ndarray:
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        image = Image.fromarray(image)
    image.save(path)


def save_np(arr_or_dict: Union[np.ndarray, torch.Tensor, dict], path: str):
    if const.DEBUG:
        return
    init_folders(path)
    if type(arr_or_dict) is dict:
        path = add_suffix(path, '.npz')
        np.savez_compressed(path, **arr_or_dict)
    else:
        if type(arr_or_dict) is torch.Tensor:
            arr_or_dict = arr_or_dict.detach().cpu().numpy()
            path = remove_suffix(path, '.npy')
        np.save(path, arr_or_dict)


@path_init('.npy', 0, False)
def load_np(path: str):
    return np.load(path)


@path_init('.pkl', 0, False)
def load_pickle(path: str):
    data = None
    if os.path.isfile(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except ValueError:
            print("Pickle5 is not installed, cannot load file with protocol 5.")
            data = None

    return data


@path_init('.pkl', 1, True)
def save_pickle(obj, path: str):
    if const.DEBUG:
        return
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_txt_labels(path: str) -> Optional[np.ndarray]:
    for suffix in ('.txt', '.seg'):
        path_ = add_suffix(path, suffix)
        if os.path.isfile(path_):
            return np.loadtxt(path_, dtype=np.int64) - 1
    return None


@path_init('.txt', 0, False)
def load_txt(path: str) -> List[str]:
    data = []
    if os.path.isfile(path):
        with open(path, 'r') as f:
            for line in f:
                data.append(line.strip())
    return data


# def load_points(path: str) -> torch.Tensor:
#     path = add_suffix(path, '.pts')
#     points = [int_b(num) for num in load_txt(path)]
#     return torch.tensor(points, dtype=torch.int64)


def save_txt(array, path: str):
    if const.DEBUG:
        return
    path_ = add_suffix(path, '.txt')
    with open(path_, 'w') as f:
        for i, num in enumerate(array):
            f.write(f'{num}{" " if i < len(array) - 1 else ""}')


def move_file(src: str, dest: str):
    if const.DEBUG:
        return
    if os.path.isfile(src):
        move(src, dest)
        return True
    return False


@path_init('.json', 1, True)
def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def collect(root: str, *suffix, prefix='') -> List[List[str]]:
    from snowman_logger import logger
    logger.info(f'collecting files from {root} with suffix {suffix} and prefix {prefix}')
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


def filter_file_list(file_list: List[List[str]], name_contain: str) -> List[List[str]]:
    return [item for item in file_list if name_contain in item[1]]

def delete_all(root:str, *suffix: str):
    if const.DEBUG:
        return
    paths = collect(root, *suffix)
    for path in paths:
        os.remove(''.join(path))


def delete_single(path: str) -> bool:
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def colors_to_colors(colors: Union[torch.Tensor, np.ndarray, Tuple[int, int, int]],
                     mesh: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> torch.Tensor:
    if type(colors) is not torch.Tensor:
        if type(colors) is np.ndarray:
            colors = torch.from_numpy(colors).long()
        else:
            colors = torch.tensor(colors, dtype=torch.int64)
    if colors.max() > 1:
        colors = colors.float() / 255
    if colors.dim() == 1:
        colors = colors.unsqueeze(int(colors.shape[0] != 3)).expand_as(mesh[0])
    return colors


def load_mesh(file_name: str, dtype: Union[type(torch.Tensor), type(np.ndarray)] = torch.Tensor,
              device: torch.device = torch.device('cpu')) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray], torch.Tensor, np.ndarray]:

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

    def fetch(lst: list, idx: int, dtype_fetch: type):
        uv_vs_ids = None
        if '/' in lst[idx]:
            lst = [item.split('/') for item in lst[idx:]]
            lst = [item[0] for item in lst]
            idx = 0
        face_vs_ids = [dtype_fetch(c.split('/')[0]) for c in lst[idx:]]
        if dtype_fetch is float and len(face_vs_ids) > 3:
            face_vs_ids = face_vs_ids[:3]
        return face_vs_ids, uv_vs_ids

    def load_from_txt(parser) -> Tuple[torch.Tensor, Union[torch.Tensor, List[List[int]]]]:
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
        mesh_t = load_from_txt(obj_parser)
    elif extension == '.off':
        mesh_t = load_from_txt(off_parser())
    elif extension == '.ply':
        mesh_t = load_ply(file_name)
    else:
        raise ValueError(f'mesh file {file_name} is not exist or not supported')

    if type(mesh_t[1]) is torch.Tensor and len(mesh_t[1]) > 0:
        if not ((mesh_t[1] >= 0) & (mesh_t[1] < mesh_t[0].shape[0])).all():
            print(f"Error: Invalid face indices in {file_name}")
    elif type(mesh_t[1]) is list and len(mesh_t[1]) > 0:
        max_idx = -1
        for face in mesh_t[1]:
             if face:
                 max_idx = max(max_idx, max(face))
        if not (max_idx < mesh_t[0].shape[0]):
            print(f"Error: Invalid face indices in {file_name}")

    vs_out, faces_out = mesh_t
    if dtype is np.ndarray:
        vs_out = vs_out.numpy()
        if type(faces_out) is torch.Tensor:
             faces_out = faces_out.numpy()
    elif device != torch.device('cpu'):
        vs_out = vs_out.to(device)
        if type(faces_out) is torch.Tensor:
             faces_out = faces_out.to(device)

    if (type(faces_out) is list and len(faces_out) == 0) or \
       (type(faces_out) is torch.Tensor and len(faces_out) == 0) or \
       (type(faces_out) is np.ndarray and len(faces_out) == 0):
        return vs_out

    return vs_out, faces_out


@path_init('.xyz', 1, True)
def export_xyz(pc: torch.Tensor, path: str, normals: Optional[torch.Tensor] = None):
    pc = pc.tolist()
    if normals is not None:
        normals = normals.tolist()
    with open(path, 'w') as f:
        for i in range(len(pc)):
            x, y, z = pc[i]
            f.write(f'{x} {y} {z}')
            if normals is not None:
                x, y, z = normals[i]
                f.write(f' {x} {y} {z}')
            if i < len(pc) - 1:
                f.write('\n')


@path_init('.txt', 2, True)
def export_gmm(gmm: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], item: int, file_name: str, included: Optional[List[int]] = None):
    if included is None:
        included = [1] * gmm[0].shape[2]
    mu, p, phi, eigen = [tensor[item, 0].flatten().cpu() for tensor in gmm]
    with open(file_name, 'w') as f:
        for tensor in (phi, mu, eigen, p):
            tensor_str = [f'{number:.5f}' for number in tensor.tolist()]
            f.write(f"{' '.join(tensor_str)}\n")
        list_str = [f'{number:d}' for number in included]
        f.write(f"{' '.join(list_str)}\n")


@path_init('.txt', 0, False)
def load_gmm(path, as_np: bool = False, device: torch.device = torch.device('cpu')) -> List[Union[np.ndarray, torch.Tensor]]:
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
        if 0 < i < 3:
            arr = arr.reshape((-1, 3))
        elif i == 3:
            arr = arr.reshape((-1, 3, 3))
        elif i == 4:
            if as_np:
                arr = arr.astype(np.bool_)
            else:
                arr = arr.bool()
        parsed.append(arr)
    return parsed


@path_init(".pt", 0, False)
def load_splice_codes(path: str, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    return torch.load(path, map_location=device, weights_only=True)


@path_init(".pt", 0, False)
def load_splice_spheres(path: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    return torch.load(path, map_location=device, weights_only=True)


@path_init('.txt', 1, True)
def export_list(lst: List[Any], path: str):
    with open(path, "w") as f:
        for i in range(len(lst)):
            f.write(f'{lst[i]}\n')


@path_init('.obj', 1, True)
def export_mesh(mesh: Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, np.ndarray, Tuple[torch.Tensor, List[List[int]]]],
                file_name: str,
                colors: Optional[Union[torch.Tensor, np.ndarray, Tuple[int, int, int]]] = None,
                normals: Optional[torch.Tensor] = None, edges=None, spheres=None):
    if type(mesh) is not tuple and type(mesh) is not list:
        vs = mesh
        faces = None
    else:
        vs, faces = mesh

    vs_is_np = isinstance(vs, np.ndarray)
    faces_is_np = isinstance(faces, np.ndarray) if faces is not None else False
    faces_is_list = isinstance(faces, list) if faces is not None else False

    if vs_is_np:
        vs = torch.from_numpy(vs)
    if faces_is_np:
        faces = torch.from_numpy(faces)

    if vs.shape[1] < 3:
        vs = torch.cat((vs, torch.zeros(len(vs), 3 - vs.shape[1], device=vs.device)), dim=1)

    if colors is not None:
        temp_faces = faces if isinstance(faces, torch.Tensor) else None
        temp_mesh_for_colors = (vs, temp_faces)
        colors = colors_to_colors(colors, temp_mesh_for_colors)

    if not os.path.isdir(os.path.dirname(file_name)):
        return

    faces_lst = []
    if faces is not None:
        if isinstance(faces, torch.Tensor):
            faces_writable: torch.Tensor = faces + 1
            faces_lst = faces_writable.tolist()
        elif faces_is_list:
            faces_lst_ = faces
            faces_lst = []
            for face in faces_lst_:
                faces_lst.append([face[i] + 1 for i in range(len(face))])

    vs_list = vs.tolist()

    with open(file_name, 'w') as f:
        for vi, v in enumerate(vs_list):
            if colors is None or colors[vi, 0] < 0:
                v_color = ''
            else:
                v_color = ' %f %f %f' % (colors[vi, 0].item(), colors[vi, 1].item(), colors[vi, 2].item())
            f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], v_color))

        if normals is not None:
            normals_list = normals.tolist()
            for n in normals_list:
                f.write("vn %f %f %f\n" % (n[0], n[1], n[2]))

        if faces is not None:
            for face in faces_lst:
                face_str = [str(f) for f in face]
                f.write(f'f {" ".join(face_str)}\n')

        if edges is not None:
            edges_list = edges.tolist()
            for edge in edges_list:
                f.write(f'\ne {edge[0]:d} {edge[1]:d}')

        if spheres is not None:
            spheres_list = spheres.tolist()
            for sphere in spheres_list:
                f.write(f'\nsp {sphere:d}')

@path_init('.ply', 1, True)
def export_ply(mesh: Tuple[torch.Tensor, torch.Tensor], path: str, colors: torch.Tensor):
    colors = colors_to_colors(colors, mesh)
    colors = (colors * 255).long()
    vs, faces = mesh
    vs = vs.clone()
    swap = vs[:, 1].clone()
    vs[:, 1] = vs[:, 2]
    vs[:, 2] = swap
    min_cor, max_cor= vs.min(0)[0], vs.max(0)[0]
    vs = vs - ((min_cor + max_cor) / 2)[None, :]
    vs = vs / vs.max()
    vs[:, 2] = vs[:, 2] - vs[:, 2].min()
    num_vs = vs.shape[0]
    num_faces = faces.shape[0]
    vs_list = vs.tolist()
    colors_list = colors.tolist()
    faces_list = faces.tolist()
    with open(path, 'w') as f:
        f.write(f'ply\nformat ascii 1.0\n'
                f'element vertex {num_vs:d}\nproperty float x\nproperty float y\nproperty float z\n'
                f'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                f'element face {num_faces:d}\nproperty list uchar int vertex_indices\nend_header\n')
        for vi, v in enumerate(vs_list):
            color = f'{colors_list[vi][0]:d} {colors_list[vi][1]:d} {colors_list[vi][2]:d}'
            f.write(f'{v[0]:f} {v[1]:f} {v[2]:f} {color}\n')
        for face in faces_list:
            f.write(f'3 {face[0]:d} {face[1]:d} {face[2]:d}\n')


@path_init('.ply', 0, False)
def load_ply(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    import plyfile
    plydata = plyfile.PlyData.read(path)
    vertices = plydata.elements[0].data
    vertices = [[float(item[0]), float(item[1]), float(item[2])] for item in vertices]
    vertices = torch.tensor(vertices)
    faces = plydata.elements[1].data
    faces = [[int(item[0][0]), int(item[0][1]), int(item[0][2])] for item in faces]
    faces = torch.tensor(faces)
    return vertices, faces


@path_init('', 1, True)
def save_model(model: Union[Optimizer, nn.Module], model_path: str):
    if const.DEBUG:
        return
    init_folders(model_path)
    torch.save(model.state_dict(), model_path)


def load_model(model: Union[Optimizer, nn.Module], model_path: str, device: torch.device, verbose: bool = False):
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        if verbose:
            print(f'loading {type(model).__name__} from {model_path}')
    elif verbose:
        print(f'init {type(model).__name__}')
    return model


def measure_time(func, num_iters: int, *args):
    start_time = time.time()
    for i in range(num_iters):
        func(*args)
    total_time = time.time() - start_time
    avg_time = total_time / num_iters
    print(f"{str(func).split()[1].split('.')[-1]} total time: {total_time}, average time: {avg_time}")


def get_time_name(name: str, format_="%m_%d-%H_%M") -> str:
    return f'{name}_{time.strftime(format_)}'


@path_init('.txt', 0, False)
def load_shapenet_seg(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    labels, vs = [], []
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split()
            vs.append([float(item) for item in data[:3]])
            labels.append(int(data[-1].split('.')[0]))
    return torch.tensor(vs, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


@path_init('.json', 0, False)
def load_json(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
