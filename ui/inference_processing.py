import os
import multiprocessing as mp
from multiprocessing import synchronize
from ui.occ_inference import Inference
from ui import files_utils
import ctypes
from enum import Enum
import numpy as np
import torch
from pynput.keyboard import Key, Controller
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
#save_data = [None,None]

class UiStatus(Enum):
    Waiting = 0
    GetMesh = 1
    SetGMM = 2
    SetMesh = 3
    ReplaceMesh = 4
    Exit = 5
    Adjustment=6
max_parts_length=20

def value_eq(value: mp.Value, status: UiStatus) -> bool:
    return value.value == status.value


def value_neq(value: mp.Value, status: UiStatus) -> bool:
    return value.value != status.value


def set_value(value: mp.Value, status: UiStatus):
    with value.get_lock():
        value.value = status.value
    print({0: 'Waiting', 1: 'GetMesh', 2: 'SetGMM', 3: 'SetMesh', 4: 'ReplaceMesh', 5: 'Exit',6:"Adjustment"}[status.value])


def set_value_if_eq(value: mp.Value, status: UiStatus, check: UiStatus):
    if value_eq(value, check):
        set_value(value, status)


def set_value_if_neq(value: mp.Value, status: UiStatus, check: UiStatus):
    if value_neq(value, check):
        set_value(value, status)


def store_mesh(
    mesh, shared_meta: mp.Array, shared_vs: mp.Array, shared_faces: mp.Array
):
    # 修改后的 store_tensor 函数，处理 numpy 数组而不是 tensor
    def store_array(array, s_array: mp.Array, dtype, meta_index):
        nonlocal shared_meta_
        s_array_ = to_np_arr(s_array, dtype)  # 将共享数组转换为 numpy 数组
        array_ = array.flatten()  # 将输入数组展平
        arr_size = array_.shape[0]
        s_array_[: arr_size] = array_  # 将展平的数组存储到共享内存中
        shared_meta_[meta_index] = arr_size  # 存储数组的大小信息

    if mesh is not None:
        shared_meta_ = to_np_arr(shared_meta, np.int32)  # 将共享元数据数组转换为 numpy 数组
        vs = mesh.vertices  # 获取顶点数据
        faces = mesh.faces  # 获取面片数据
        
        # 存储顶点和面片数据
        store_array(vs, shared_vs, np.float32, 0)
        store_array(faces, shared_faces, np.int32, 1)



def load_mesh(
    shared_meta: mp.Array, shared_vs: mp.Array, shared_faces: mp.Array
):

    def load_array(s_array: mp.Array, dtype, meta_index):
        nonlocal shared_meta_
        s_array_ = to_np_arr(s_array, dtype)
        array_ = s_array_[: shared_meta_[meta_index]].copy()
        array_ = array_.reshape((-1, 3))
        return array_

    shared_meta_ = to_np_arr(shared_meta, np.int32)
    vs = load_array(shared_vs, np.float32, 0)
    faces = load_array(shared_faces, np.int32, 1)
    return vs, faces


def store_gmm(shared_gmm: mp.Array, gmm, included,selected, res: int=None):
    shared_arr = to_np_arr(shared_gmm, np.float32)
    mu, p, phi, eigen = gmm
    num_gaussians = included.shape[0]
    ptr = 0
    for i, (item, skip) in enumerate(zip((included,selected, mu, p, phi, eigen), InferenceProcess.skips)):
        item = item.flatten().detach().cpu().numpy()
        if item.dtype != np.float32:
            item = item.astype(np.float32)
        shared_arr[ptr: ptr + skip * num_gaussians] = item
        if i == 0:
            shared_arr[ptr + skip * num_gaussians: ptr + skip * max_parts_length] = -2
        ptr += skip * max_parts_length
    if res is not None:
        shared_arr[-1] = float(res)


def load_gmm(shared_gmm: mp.Array):
    shared_arr = to_np_arr(shared_gmm, np.float32)
    parsed_arr = []
    num_gaussians = 0
    ptr = 0
    shape = {1: (1, 1, -1), 2: (-1, 2), 3: (1, 1, -1, 3), 9: (1, 1, -1, 3, 3)}
    for i, skip in enumerate(InferenceProcess.skips):
        raw_arr = shared_arr[ptr : ptr + skip * max_parts_length]
        if i == 0:
            arr = torch.tensor(
                [int(item) for item in raw_arr if item >= -1], dtype=torch.int64
            )
            num_gaussians = arr.shape[0] // 2
        else:
            arr = torch.from_numpy(raw_arr[: skip * num_gaussians]).float()
        arr = arr.view(*shape[skip])
        parsed_arr.append(arr)
        ptr += skip * max_parts_length
    return parsed_arr[2:], parsed_arr[0],parsed_arr[1], int(shared_arr[-1])

def store_cond(shared_cond, lock_pose, lock_shape, lock_empty, generate_mode):

    # Convert boolean values to floats (1.0 for True, 0.0 for False)
    bool_vector = [int(lock_pose), int(lock_shape), int(lock_empty), int(generate_mode)]
    
    # Convert the shared memory array to a numpy array
    shared_arr = to_np_arr(shared_cond, np.int32)
    
    # Store the boolean vector in the shared array
    for i, value in enumerate(bool_vector):
        shared_arr[i] = value



def store_mid(shared_mid: mp.Array, mid: torch.Tensor, max_parts_length: int):
    """
    将形状为 [x, 512] 的 mid Tensor 存储到共享内存 shared_mid 中。
    - shared_mid: mp.Array 类型共享内存对象
    - mid: torch.Tensor，形状为 [x, 512]
    - max_parts_length: 允许的最大 x 值
    """
    shared_arr = to_np_arr(shared_mid, np.float32)
    
    # 展平 mid，并确保数据类型为 float32
    flat_mid = mid.flatten().detach().cpu().numpy()
    if flat_mid.dtype != np.float32:
        flat_mid = flat_mid.astype(np.float32)
    
    # 将 flat_mid 存入 shared_arr 中，并填充剩余位置为 -1
    shared_arr[: flat_mid.size] = flat_mid
    if flat_mid.size < max_parts_length * 512:
        shared_arr[flat_mid.size : max_parts_length * 512] = -1  # 填充多余空间为 -1

def load_mid(shared_mid: mp.Array):
    """
    从共享内存 shared_mid 中加载形状为 [x, 512] 的 Tensor。
    - shared_mid: mp.Array 类型共享内存对象
    - max_parts_length: 允许的最大 x 值
    - 返回值: 加载后的 torch.Tensor，形状为 [x, 512]
    """
    shared_arr = to_np_arr(shared_mid, np.float32)
    
    # 找到有效数据部分（即不等于 -1 的部分）
    valid_size = np.count_nonzero(shared_arr != -1)
    
    # 提取有效数据并将其还原为 [x, 512] 形状
    valid_data = shared_arr[:valid_size]
    mid_tensor = torch.from_numpy(valid_data).view(-1, 512)
    
    return mid_tensor

def pad_to_max_length(tensor_list, padding_value=0):
    """
    将 tensor_list 中的每个张量填充到最大长度，使第一维长度一致。
    - tensor_list: List[torch.Tensor]，每个张量的形状为 [x, ...]
    - padding_value: 用于填充的值，默认填充为 0
    - 返回: 填充后的张量列表，每个张量形状为 [max_length, ...]
    """
    # 获取最大长度
    max_length = max(tensor.size(0) for tensor in tensor_list)
    
    # 填充每个张量
    padded_tensors = []
    for tensor in tensor_list:
        padding_size = (max_length - tensor.size(0),) + tensor.size()[1:]  # 计算填充形状
        padded_tensor = torch.cat([tensor, torch.full(padding_size, padding_value, dtype=tensor.dtype)])
        padded_tensors.append(padded_tensor)
    
    return padded_tensors

def inference_process(
    cfg,
    wake__condition: synchronize.Condition,
    sleep__condition: synchronize.Condition,
    status: mp.Value,
    samples_root: str,
    shared_gmm: mp.Array,
    shared_meta: mp.Array,
    shared_vs: mp.Array,
    shared_faces: mp.Array,
    shared_mid,
    cond
):
    model = Inference(cfg)  # 初始化 model

    items = files_utils.collect(samples_root, ".pkl")
    items = [files_utils.load_pickle("".join(item)) for item in items]
    items = pad_to_max_length(items)  # 填充到最大长度
    items = torch.stack(items, dim=0)
    # items = [int(item[1]) for item in items]
    model.set_items(items)
    keyboard = Controller()
    while value_neq(status, UiStatus.Exit):
        while value_eq(status, UiStatus.Waiting):
            with sleep__condition:
                sleep__condition.wait()
                # 增加新的状态 UiStatus.ExtractFeatures 来调用新模型
        
        if value_eq(status, UiStatus.Adjustment):
            set_value(status, UiStatus.SetGMM)
            gmm_info = load_gmm(shared_gmm)
            set_value_if_eq(status, UiStatus.SetMesh, UiStatus.SetGMM)
            print("do_adjustment,do_adjustment,do_adjustment")
            gmm_info=model.do_adjustment(cond,*gmm_info)
            store_gmm(shared_gmm,*gmm_info)
            set_value(status, UiStatus.GetMesh)

        if value_eq(status, UiStatus.GetMesh):
            set_value(status, UiStatus.SetGMM)
            gmm_info = load_gmm(shared_gmm)
            set_value_if_eq(status, UiStatus.SetMesh, UiStatus.SetGMM)
            mesh,mid = model.get_mesh_from_mid(*gmm_info)
            store_mid(shared_mid,mid,max_parts_length)
            #print(mesh is not None)
            if mesh is not None:
                store_mesh(mesh, shared_meta, shared_vs, shared_faces)
                keyboard.press(Key.ctrl_l)
                keyboard.release(Key.ctrl_l)
            set_value_if_eq(status, UiStatus.ReplaceMesh, UiStatus.SetMesh)
            with wake__condition:
                wake__condition.notify_all()
    return 0


def to_np_arr(shared_arr: mp.Array, dtype):
    return np.frombuffer(shared_arr.get_obj(), dtype=dtype)


class InferenceProcess:

    skips = (2,1, 3, 9, 1, 3)
    #skips = (2, 3, 3, 9)

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
            gmms, included,selected = self.request_gmm()
            store_gmm(self.shared_gmm, gmms, included,selected, res)
            set_value_if_neq(self.status, UiStatus.GetMesh, UiStatus.GetMesh)
            # if value_eq(self.status, UiStatus.Waiting):
            with self.wake_condition:
                self.wake_condition.notify_all()
        return
    def do_adjust(self,lock_pose,lock_shape,lock_empty,generate_mode):
        keyboard=Controller()
        print(lock_pose,lock_shape,lock_empty,generate_mode)
        if value_neq(self.status, UiStatus.SetGMM):
            gmms, included,selected = self.request_gmm()
            store_gmm(self.shared_gmm, gmms, included,selected)
            store_cond(self.shared_cond,lock_pose,lock_shape,lock_empty,generate_mode)
            set_value_if_neq(self.status,UiStatus.Adjustment,UiStatus.GetMesh )
            # if value_eq(self.status, UiStatus.Waiting):
            with self.wake_condition:
                print("adj start")
                self.wake_condition.notify_all()

                    # 等待模型进程完成工作
            with self.sleep_condition:  
                self.sleep_condition.wait()  # 等待模型进程完成任务并通知
                print("adj over")

            if self.send_gmm is not None:
                gmms, included,selected,_=load_gmm(self.shared_gmm)
                self.send_gmm(gmms, included,selected)

        return

    def __init__(self, cfg1, fill_ui_mesh, request_gmm, samples_root,send_gmm=None):
        global cfg 
        cfg = cfg1
    
        self.status = mp.Value("i", UiStatus.Waiting.value)
        self.request_gmm = request_gmm
        self.send_gmm = send_gmm
        self.sleep_condition = mp.Condition()
        self.wake_condition = mp.Condition()
        self.shared_gmm = mp.Array(
            ctypes.c_float, max_parts_length * sum(self.skips) + 1
        )
        self.shared_vs = mp.Array(ctypes.c_float, 400000 * 3)
        self.shared_faces = mp.Array(ctypes.c_int, 400000 * 8)
        self.shared_meta = mp.Array(ctypes.c_int, 2)
        self.shared_mid = mp.Array(ctypes.c_float, max_parts_length * 512)
        self.shared_cond = mp.Array(ctypes.c_int, 4)
        self.model_process = mp.Process(
            target=inference_process,
            args=(
                cfg,
                self.sleep_condition,
                self.wake_condition,
                self.status,
                samples_root,
                self.shared_gmm,
                self.shared_meta,
                self.shared_vs,
                self.shared_faces,
                self.shared_mid,
                self.shared_cond,
            ),
        )
        self.fill_ui_mesh = fill_ui_mesh
        self.model_process.start()
