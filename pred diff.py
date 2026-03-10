import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data_process.part_vox_data import VoxelRayDataModule
from model.SPLICE_model import SPLICE_Module
import multiprocessing as mp
import math

# ----------------- 运行示例 -----------------
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    # 数据模块
    files = 'data/03001627_chair_test'  # 替换为实际路径
    dm = VoxelRayDataModule(root_dir=files, batch_size=1, num_workers=5,data_name="partnet")
    # 定义模型
    model = SPLICE_Module(lr=1e-4,export_mode="DIFF",export_shape=False) #load_from_checkpoint("checkpoints_splice/best-v1.ckpt",lr=1e-3)


    # Trainer：添加回调
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[7],
    )

    # 如果存在已保存的 checkpoint，自动恢复训练
    last_ckpt = "./checkpoints_splice/splice-1.5-v1.ckpt"
    if last_ckpt:
        print(f"加载已有 checkpoint: {last_ckpt}")
        trainer.predict(model,  datamodule=dm, ckpt_path=last_ckpt)
    else:
        trainer.predict(model, datamodule=dm)
