import os
import torch
import socket
import torch.nn
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
import data_process
import data_process.data_module
from omegaconf import DictConfig
import model
import model.noodles
import torch.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint
import typing_extensions

# import wandb


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig = None):
    # os.environ["USE_LIBUV"] = "0"
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )

    torch.set_float32_matmul_precision("medium")

    # 创建数据模块
    data_module = data_process.data_module.NoodlesDataModule(**cfg.train)

    # 创建模型
    noodles = model.noodles.NoodlesModel(**cfg.train)

    # # 确保模型和数据都支持 CUDA
    # if torch.cuda.is_available():
    #     noodles = noodles.cuda()

    # 配置 TensorBoard Logger
    tensorboard_logger = TensorBoardLogger("tb_logs", name="NoodlesProject_vvv")

    # 配置 ModelCheckpoint 回调来保存最好的模型
    best_model_checkpoint_callback = ModelCheckpoint(
        monitor="occ_loss_epoch",  # 监控验证损失
        mode="min",  # 选择最小的验证损失作为最佳模型
        save_top_k=1,  # 只保留最好的一个模型
        dirpath="checkpoints/",  # 保存的路径
        filename="best_model",  # 保存的文件名
    )

    # 配置 ModelCheckpoint 回调来保存最新的模型
    latest_model_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,  # 只保留最新的一个模型
        dirpath="checkpoints/",  # 保存的路径
        filename="latest_model",  # 保存的文件名
        every_n_epochs=1,  # 每个 epoch 都保存一次
    )

    # 创建 Trainer
    trainer = pl.Trainer(
        max_epochs=20000,
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,
        log_every_n_steps=50,
        callbacks=[best_model_checkpoint_callback, latest_model_checkpoint_callback],
        gradient_clip_val=1.0,
    )

    # 开始训练
    trainer.fit(noodles, data_module)#, ckpt_path="checkpoints/开源测试.ckpt")


if __name__ == "__main__":
    main()
