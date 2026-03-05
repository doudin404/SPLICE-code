import os
import torch
import socket
import torch.nn
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
import data_process
import data_process.data_module
from omegaconf import DictConfig
import model
import model.fried_noodles
import model.noodles
import torch.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint
import typing_extensions
import trick_mian
from model.noodles import model_load
from model.fried_noodles import adjust_load, pose_load

# import wandb


@hydra.main(config_path="config", config_name="config_fry", version_base="1.2")
def main(cfg: DictConfig = None):
    # os.environ["USE_LIBUV"] = "0"
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )

    torch.set_float32_matmul_precision("medium")

    ae_model = model_load(cfg).eval()
    adjust_model = adjust_load(cfg).eval()
    pose_model = pose_load(cfg).eval()

    z_gmm = pose_model.pose.model.mask_sample().float()

    code = torch.zeros([1, 20, 1024], device=z_gmm.device)

    mask = torch.zeros_like(code).bool()

    # Mask the GMM part (first 512 dimensions)
    mask[:, : z_gmm.shape[0], :512] = True
    # Unmask the mid part (last 512 dimensions)
    mask[:, : z_gmm.shape[0], 512:] = False

    gmm_encoded = ae_model.noodles.encode_gm(
        z_gmm.unsqueeze(0), torch.zeros_like(z_gmm[None, :, 0]).bool()
    )  # z_gmm.unsqueeze(0): [1, x', 15]

    # Prepare mid features
    # mid shape: [1, 20, 512]
    mid = torch.zeros([1, 20, 512], device=z_gmm.device)

    # Concatenate gmm_encoded and mid to form code
    # code[0, : x', :] shape: [x', 1024]
    code[0, : gmm_encoded.shape[1], :] = torch.cat(
        [gmm_encoded[0], mid[0, : gmm_encoded.shape[1]]], dim=-1
    )  # code: [1, 20, 1024]

    sample = adjust_model.model.mask_sample(code, ~mask).float()
    sample = sample[..., 512:]

    # Remove invalid entries (norm less than 1)
    # sample_mask shape: [1, 20]
    sample_mask = sample.norm(2, -1) < 1

    # Update GMM with valid entries from code
    # gmm shape: [x'', 15], where x'' ≤ 20
    sample = sample[0, ~sample_mask[0]]
    gmm = code[0, ~sample_mask[0], :15]

    #还有最后的解码部分



if __name__ == "__main__":
    main()