import os
from pathlib import Path
from typing import Optional

import torch

# ===== 根据你的工程实际路径修改这些 import =====
# 假设下面这三个类与当前脚本在同一工程中：
# - DiffDataModule  (包装 DiffDataset)
# - SPLICE_Diffusion (扩散模型)
# - SPLICE_Module    (解码模型)
# - ShapeSampler     (已在“Shape Sampler Mask Denoise”中实现)
from data_process.diff_data import DiffDataModule  # 替换为实际模块路径
from model.SPLICE_model import SPLICE_Module  # 替换为实际模块路径
from model.DIFF_model import SPLICE_Diffusion  # 替换为实际模块路径
from model.SHAPE_sampler import ShapeSampler      # 替换为实际模块路径
def transform_data(data):
    #scale = torch.rand(1).item() * 0.4 + 0.8
    data[:, 1:4] *=4
    data[:, 13:16] *=4
    data[:, 16:]*=0.1
    data[:, 4:13]*=1
    return data

def get_one_batch(dm: DiffDataModule, which: str = "predict") -> torch.Tensor:
    """
    从 DataModule 的指定 dataloader 中取出一个 batch。
    which: "train" | "val" | "test" | "predict"
    返回张量形状 [B, L, D]，与训练时一致（flag+15+feat）。
    """
    if which == "train":
        loader = dm.train_dataloader()
    elif which == "val":
        loader = dm.val_dataloader()
    elif which == "test":
        loader = dm.test_dataloader()
    else:
        loader = dm.predict_dataloader()

    batch = next(iter(loader))  # DiffDataset __getitem__ 直接返回 [L, D]
    if isinstance(batch, torch.Tensor) and batch.ndim == 3:
        return batch  # [B, L, D]
    # 若你的 DataModule 返回字典/其他结构，这里适配一下
    if isinstance(batch, dict) and "x" in batch:
        return batch["x"]
    raise TypeError("无法从 dataloader 解析到形状为 [B, L, D] 的张量，请检查 DataModule 返回格式。")


def build_fixed_mask_from_flags(x: torch.Tensor) -> torch.Tensor:
    """根据 batch 的 flag 列自动构建固定掩码：flag>0 的位置设为 True。
    x: [B, L, D]，第 0 列为 flag。
    返回 [B, L] 的 bool 掩码。
    """
    if x.ndim != 3 or x.shape[-1] < 1:
        raise ValueError("期望 x 形状为 [B, L, D] 且 D>=1")
    return (x[:, :, 0] > 0)


def main() -> None:
    torch.set_float32_matmul_precision("high")

    # 可改动的采样配置（按需增减）
    sample_configs = {
        # --- 数据与输出 ---
        "data_dir": "data/chair_feats_3",   # 你的 feats/params 所在目录
        "save_dir": "export/gen_overfit_3",    # OBJ 输出目录
        # --- DataModule ---
        "batch_size": 100,
        "num_workers": 5,
        # --- 使用已有数据做去噪优化? ---
        "use_init_data": True,              # True: 从 DataModule 取一批作为起点；False: 纯采样（忽略 steps，严格全步）
        "init_loader": "predict",          # 从哪个 dataloader 取：train/val/test/predict
        # --- 部分固定（补全/约束）---
        "use_fixed_mask": False,            # True: 根据 flag>0 自动构建掩码；False: 不使用掩码
        # --- 去噪步数（仅在 use_init_data=True 时生效）---
        "noise_t": 1000,                        # 表示在 T(如1000) 步中子采样多少步进行去噪优化
        # --- 其他 ---
        "name_prefix": "gen",              # 输出文件前缀
        "device": "cuda:0",                     # 手动指定 'cuda'/'cpu'；None 自动
        "diffusion_ckpt": "checkpoints_splice_diff/overfit_3.ckpt",  # 扩散模型存档
        "decoder_ckpt": "./checkpoints_splice/to_diff_1e-3.ckpt"#splice-1.5-03001627_chair.ckpt"
    }

    # ===== 实例化 DataModule =====
    dm = DiffDataModule(
        data_dir=sample_configs["data_dir"],
        batch_size=sample_configs["batch_size"],
        num_workers=sample_configs["num_workers"],
        transform=transform_data,  # 可选数据变换
    )
    # 为了能拿到 train/val/test，通常先 setup("fit")；predict 用 setup("predict") 即可
    dm.setup(stage="fit")
    dm.setup(stage="predict")

    # ===== 实例化模型（使用默认超参）=====

    # 从存档加载初始参数
    diffusion = SPLICE_Diffusion.load_from_checkpoint(sample_configs["diffusion_ckpt"])
    decoder = SPLICE_Module.load_from_checkpoint(sample_configs["decoder_ckpt"])

    # 可选：手动设定设备
    if sample_configs["device"] is not None:
        device = torch.device(sample_configs["device"])
        diffusion.to(device)
        decoder.to(device)

    # ===== 组装采样器 =====
    sampler = ShapeSampler(diffusion_model=diffusion, decoder_model=decoder, save_dir=sample_configs["save_dir"])

    # ===== 准备 init_data / fixed_mask =====
    if sample_configs["use_init_data"]:
        batch_x = get_one_batch(dm, which=sample_configs["init_loader"])  # [B, L, D]
        fixed_mask = build_fixed_mask_from_flags(batch_x) if sample_configs["use_fixed_mask"] else None
        paths = sampler.generate(
            init_data=batch_x,
            fixed_mask=fixed_mask,
            noise_t=sample_configs["noise_t"],
            
            name_prefix=sample_configs["name_prefix"],
        )
    else:
        # 纯采样：严格全步去噪，忽略 noise_t
        paths = sampler.generate(
            init_data=None,
            fixed_mask=None,
            batch_size=sample_configs["batch_size"],
            max_len=20,
            noise_t=1,  # 无效，占位
            name_prefix=sample_configs["name_prefix"],
        )

    print("Saved:")
    for p in paths:
        print(p)
    []


if __name__ == "__main__":
    main()
