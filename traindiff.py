import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data_process.diff_data import DiffDataModule
from model.DIFF_model import SPLICE_Diffusion
import multiprocessing as mp
import math

def transform_data(data):
    scale = torch.rand(1).item() * 0.4 + 0.8
    data[:, 1:4] *= scale*5
    data[:, 13:16] *= scale*5
    data[:, 16:]*=0.1
    data[:, 4:13]*=1
    return data
# ----------------- 运行示例 -----------------
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    # 日志记录器
    tb_logger = TensorBoardLogger(save_dir="logs/splice_log", name="partnet")

    # 数据模块
    files = 'data/chair_feats_5'  # 替换为实际路径
    dm = DiffDataModule(data_dir=files, batch_size=256, num_workers=5,transform=transform_data)
    # 定义模型
    model = SPLICE_Diffusion()#.load_from_checkpoint("checkpoints_splice_diff/overfit_3.ckpt")#14 15

    # Checkpoint 回调：保存最佳模型和最后一次训练模型
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_splice_diff",          # 保存文件夹
        filename="best_overfit_5",                # 最佳模型文件名前缀
        save_top_k=1,                     # 只保留最优模型
        monitor="val_loss",             # 监控指标，可根据需改
        mode="min",                     # 指标越小越好
        save_last=True                    # 额外保存最后一次训练的模型
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"overfit_5"########################

    # Trainer：添加回调
    trainer = pl.Trainer(
        max_epochs=-1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[2,3],#[7],
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=100
    )

    # 如果存在已保存的 checkpoint，自动恢复训练
    last_ckpt = "checkpoints_splice_diff/overfit_5.ckpt"
    if last_ckpt:
        print(f"加载已有 checkpoint: {last_ckpt}")
        trainer.fit(model,  datamodule=dm, ckpt_path=last_ckpt)
    else:
        trainer.fit(model, datamodule=dm)

    # 训练完成后，最佳和最后模型路径
    print("最佳模型路径：", checkpoint_callback.best_model_path)
    print("最后一次模型路径：", checkpoint_callback.last_model_path)