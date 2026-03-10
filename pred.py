import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data_process.part_vox_data import VoxelRayDataModule
from model.SPLICE_model import SPLICE_Module
import multiprocessing as mp
import os

# ShapeNet: airplane
# DATA_DIR = 'data/02691156_airplane' 
# SAVE_DIR = "export/02691156_airplane_g" 
# CKPT_DIR = "./checkpoints_splice/splice-1.5-02691156_airplane.ckpt"
# DATA_NAME = "dae_net"

# PartNet: cabinet
# DATA_DIR = 'data/data_StorageFurniture' 
# SAVE_DIR = "export/02933112_cabinet_test_g" 
# CKPT_DIR = "./checkpoints_splice/splice-1.5-02933112_cabinet.ckpt"
# DATA_NAME = "partnet"

# # PartNet: table
# DATA_DIR = 'data/data_Table' 
# SAVE_DIR = "export/04379243_table_g" 
# CKPT_DIR = "./checkpoints_splice/splice-1.5-04379243_table.ckpt"
# DATA_NAME = "partnet"

# PartNet: chair
DATA_DIR = 'data/data_Chair'
SAVE_DIR = "export/03001627_chair_diff"
CKPT_DIR = "./checkpoints_splice/splice-1.5-03001627_chair.ckpt"
DATA_NAME = "partnet"

# # PartNet: plane 消融
# DATA_DIR = 'data/02691156_airplane'
# SAVE_DIR = "export/plane_v4"
# CKPT_DIR = "checkpoints_splice/last-v12.ckpt"
# DATA_NAME = "dae_net"

# ----------------- 运行示例 -----------------
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    # 数据模块
    files = DATA_DIR
    dm = VoxelRayDataModule(root_dir=files, batch_size=1, num_workers=5,data_name=DATA_NAME)
    
    # 定义模型
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    model = SPLICE_Module(save_dir=SAVE_DIR,export_shape=False,export_mode="DIFF")

    # Trainer：添加回调
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[7],
    )

    last_ckpt = "./checkpoints_splice/to_diff_1e-3.ckpt"# CKPT_DIR
    if last_ckpt:
        print(f"加载已有 checkpoint: {last_ckpt}")
        trainer.predict(model,  datamodule=dm, ckpt_path=last_ckpt)
    else:
        trainer.predict(model, datamodule=dm)
