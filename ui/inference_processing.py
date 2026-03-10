import os
import enum
import ctypes
from typing import Tuple, List, Union, Callable, Optional
import multiprocessing as mp
from multiprocessing import synchronize
from multiprocessing.sharedctypes import SynchronizedArray

import numpy as np
import torch

import options
import constants
from ui.occ_inference import Inference
from utils_spaghetti import files_utils
if constants.IS_WINDOWS or 'DISPLAY' in os.environ:
    from pynput.keyboard import Key, Controller
else:
    from ui.mock_keyboard import Key, Controller
from snowman_logger import logger

class UiStatus(enum.Enum):
    Waiting = 0
    GetMesh = 1
    SetGMM = 2
    SetMesh = 3
    ReplaceMesh = 4
    Exit = 5


def value_eq(value: SynchronizedArray, status: UiStatus) -> bool:
    return value.value == status.value


def value_neq(value: SynchronizedArray, status: UiStatus) -> bool:
    return value.value != status.value


def set_value(value: SynchronizedArray, status: UiStatus):
    with value.get_lock():
        value.value = status.value
    logger.debug({0: 'Waiting', 1: 'GetMesh', 2: 'SetGMM', 3: 'SetMesh', 4: 'ReplaceMesh', 5: 'Exit'}[status.value],
                 extra={"tag": "UI STATUS"})


def set_value_if_eq(value: SynchronizedArray, status: UiStatus, check: UiStatus):
    if value_eq(value, check):
        set_value(value, status)


def set_value_if_neq(value: SynchronizedArray, status: UiStatus, check: UiStatus):
    if value_neq(value, check):
        set_value(value, status)


def store_mesh(mesh: Tuple[torch.Tensor, Optional[torch.Tensor]], shared_meta: SynchronizedArray, shared_vs: SynchronizedArray, shared_faces: SynchronizedArray):

    def store_tensor(tensor: torch.Tensor, s_array: SynchronizedArray, dtype, meta_index):
        nonlocal shared_meta_
        s_array_ = to_np_arr(s_array, dtype)
        array_ = tensor.detach().cpu().flatten().numpy()
        arr_size = array_.shape[0]
        s_array_[:array_.shape[0]] = array_
        shared_meta_[meta_index] = arr_size

    if mesh is not None:
        shared_meta_ = to_np_arr(shared_meta, np.int32)
        vs, faces = mesh
        store_tensor(vs, shared_vs, np.float32, 0)
        store_tensor(faces, shared_faces, np.int32, 1)


def load_mesh(shared_meta: SynchronizedArray, shared_vs: SynchronizedArray, shared_faces: SynchronizedArray) -> Tuple[np.ndarray, np.ndarray]:

    def load_array(s_array: SynchronizedArray, dtype, meta_index) -> np.ndarray:
        nonlocal shared_meta_
        s_array_ = to_np_arr(s_array, dtype)
        array_ = s_array_[: shared_meta_[meta_index]].copy()
        array_ = array_.reshape((-1, 3))
        return array_

    shared_meta_ = to_np_arr(shared_meta, np.int32)
    vs = load_array(shared_vs, np.float32, 0)
    faces = load_array(shared_faces, np.int32, 1)
    return vs, faces


def store_gmm(shared_gmm: SynchronizedArray, gmm: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], included: torch.Tensor, res: int):
    shared_arr = to_np_arr(shared_gmm, np.float32)
    mu, p, phi, eigen = gmm
    num_gaussians = included.shape[0]
    ptr = 0
    for i, (item, skip) in enumerate(zip((included, mu, p, phi, eigen), InferenceProcess.skips)):
        item = item.flatten().detach().cpu().numpy()
        if item.dtype != np.float32:
            item = item.astype(np.float32)
        shared_arr[ptr: ptr + skip * num_gaussians] = item
        if i == 0:
            shared_arr[ptr + skip * num_gaussians: ptr + skip * constants.MAX_GAUSSIANS] = -1
        ptr += skip * constants.MAX_GAUSSIANS
    shared_arr[-1] = float(res)


def load_gmm(shared_gmm: SynchronizedArray) -> Tuple[Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], torch.Tensor, int]:
    shared_arr = to_np_arr(shared_gmm, np.float32)
    parsed_arr = []
    num_gaussians = 0
    ptr = 0
    shape = {1: (1, 1, -1), 2: (-1, 2), 3: (1, 1, -1, 3), 9: (1, 1, -1, 3, 3)}
    for i, skip in enumerate(InferenceProcess.skips):
        raw_arr = shared_arr[ptr: ptr + skip * constants.MAX_GAUSSIANS]
        if i == 0:
            arr = torch.tensor([int(item) for item in raw_arr if item >= 0], dtype=torch.int64)
            num_gaussians = arr.shape[0] // 2
        else:
            arr = torch.from_numpy(raw_arr[: skip * num_gaussians]).float()
        arr = arr.view(*shape[skip])
        parsed_arr.append(arr)
        ptr += skip * constants.MAX_GAUSSIANS
    return parsed_arr[1:], parsed_arr[0], int(shared_arr[-1])


def inference_process(opt: options.Options, wake_condition: synchronize.Condition,
                      sleep__condition: synchronize.Condition, status: SynchronizedArray, samples_root: str,
                      shared_gmm: SynchronizedArray, shared_meta: SynchronizedArray, shared_vs: SynchronizedArray, shared_faces: SynchronizedArray):
    model = Inference(opt)
    if opt.model_name == "spaghetti":
        items = files_utils.collect(samples_root, '.pkl')
        items = [files_utils.load_pickle(''.join(item)) for item in items]  # List[tensor.shape = (16, 512)]
        # 为了适配 SPLCE 变长的小球数量，此处修改为不使用 stack 堆叠，而是使用 list 存储每一个小球的 tensor
        # 因为 stack 会要求所有的小球数量相同，而 SPLICE 的小球数量是变长的，SPAGHETTI 的椭球数量固定为 16
        # items = torch.stack(items, dim=0)  # tensor.shape = (num_shape, 16, 512)
        # items = [int(item[1]) for item in items]
    elif opt.model_name == "splice-v1.5":
        items = files_utils.collect(samples_root, '.pkl')
        items = [files_utils.load_pickle(''.join(item)) for item in items]  # List[tensor.shape = (16, 256)]
        # v1.5 版本的 SPLICE，输入格式适配为 SPAGHETTI 的格式，但是 latent code 的 tensor shape 为 (16, 256)
    elif opt.model_name == "splice":
        items = files_utils.collect(samples_root, '.pt')
        items = files_utils.filter_file_list(items, 'codes')
        # items_sphere = files_utils.filter_file_list(items, 'spheres')
        items = [files_utils.load_splice_codes(''.join(item)) for item in items]
    else:
        raise ValueError("Unsupported model name")
    model.set_items(items)
    keyboard = Controller()
    while value_neq(status, UiStatus.Exit):
        while value_eq(status, UiStatus.Waiting):
            with sleep__condition:
                sleep__condition.wait()
        if value_eq(status, UiStatus.GetMesh):
            set_value(status, UiStatus.SetGMM)
            gmm_info = load_gmm(shared_gmm)
            set_value_if_eq(status, UiStatus.SetMesh, UiStatus.SetGMM)
            mesh = model.get_mesh_from_mid(*gmm_info)
            if mesh is not None:
                store_mesh(mesh, shared_meta, shared_vs, shared_faces)
                keyboard.press(Key.ctrl_l)
                keyboard.release(Key.ctrl_l)
            set_value_if_eq(status, UiStatus.ReplaceMesh, UiStatus.SetMesh)
            # with wake_condition:
            #     wake_condition.notify_all()
    with wake_condition:
        wake_condition.notify_all()
    return 0


def to_np_arr(shared_arr: SynchronizedArray, dtype) -> np.ndarray:
    return np.frombuffer(shared_arr.get_obj(), dtype=dtype)


class InferenceProcess:

    skips = (2, 3, 9, 1, 3)

    def exit(self):
        set_value(self.status, UiStatus.Exit)
        with self.wake_condition:
            self.wake_condition.notify_all()
        self.model_process.join()

    def replace_mesh(self):
        mesh = load_mesh(self.shared_meta, self.shared_vs, self.shared_faces)
        self.fill_ui_mesh(mesh)
        set_value_if_eq(self.status, UiStatus.Waiting, UiStatus.ReplaceMesh)

    def get_mesh(self, res: int):
        if value_neq(self.status, UiStatus.SetGMM):
            gmms, included = self.request_gmm()
            if gmms is None or included is None:
                logger.error("GMM request returned None, exiting inference process.")
                set_value(self.status, UiStatus.Exit)
                return
            store_gmm(self.shared_gmm, gmms, included, res)
            set_value_if_neq(self.status, UiStatus.GetMesh, UiStatus.GetMesh)
            # if value_eq(self.status, UiStatus.Waiting):
            with self.wake_condition:
                self.wake_condition.notify_all()
        return

    def __init__(self, opt, fill_ui_mesh: Callable[[Tuple[np.ndarray, np.ndarray]], None], 
                 request_gmm: Callable[[], Tuple[Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], torch.Tensor]],
                 samples_root: List[List[str]]):
        self.status = mp.Value('i', UiStatus.Waiting.value)
        self.request_gmm = request_gmm
        self.sleep_condition = mp.Condition()
        self.wake_condition = mp.Condition()
        self.shared_gmm = mp.Array(ctypes.c_float, constants.MAX_GAUSSIANS * sum(self.skips) + 1)
        self.shared_vs = mp.Array(ctypes.c_float, constants.MAX_VS * 3)
        self.shared_faces = mp.Array(ctypes.c_int, constants.MAX_VS * 8)
        self.shared_meta = mp.Array(ctypes.c_int, 2)
        self.model_process = mp.Process(target=inference_process,
                                        args=(opt, self.sleep_condition, self.wake_condition, self.status, samples_root,
                                              self.shared_gmm, self.shared_meta, self.shared_vs, self.shared_faces))
        self.fill_ui_mesh = fill_ui_mesh
        self.model_process.start()
