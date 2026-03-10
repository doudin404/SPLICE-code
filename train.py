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
    # 日志记录器
    tb_logger = TensorBoardLogger(save_dir="logs/splice_log", name="partnet")

    # 数据模块
    files = 'data/data_Chair'  # 替换为实际路径
    dm = VoxelRayDataModule(root_dir=files, batch_size=10, num_workers=5,data_name="partnet")
    # 定义模型
    model = SPLICE_Module() #load_from_checkpoint("checkpoints_splice/best-v1.ckpt",lr=1e-3)

    # Checkpoint 回调：保存最佳模型和最后一次训练模型
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_splice",          # 保存文件夹
        filename="best",                # 最佳模型文件名前缀
        save_top_k=1,                     # 只保留最优模型
        monitor="total",             # 监控指标，可根据需改
        mode="min",                     # 指标越小越好
        save_last=True                    # 额外保存最后一次训练的模型
    )

    # Trainer：添加回调
    trainer = pl.Trainer(
        max_epochs=100000,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=7,#[7],
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    # 如果存在已保存的 checkpoint，自动恢复训练
    last_ckpt = "checkpoints_splice/last-v8.ckpt"#splice-1.5-03001627_chair.ckpt"
    if last_ckpt:
        print(f"加载已有 checkpoint: {last_ckpt}")
        trainer.fit(model,  datamodule=dm, ckpt_path=last_ckpt)
    else:
        trainer.fit(model, datamodule=dm)

    # 训练完成后，最佳和最后模型路径
    print("最佳模型路径：", checkpoint_callback.best_model_path)
    print("最后一次模型路径：", checkpoint_callback.last_model_path)