import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data_process.part_vox_data import VoxelRayDataModule
from model.SPLICE_model import SPLICE_Module
import multiprocessing as mp
import math

# ----------------- Execution Entry Point -----------------
if __name__ == '__main__':
    # Set matrix multiplication precision for better performance on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')

    # Logger setup
    tb_logger = TensorBoardLogger(save_dir="logs/splice_log", name="partnet")

    # Data module initialization
    files = 'data/data_Chair'  # Replace with actual data path
    dm = VoxelRayDataModule(
        root_dir=files, 
        batch_size=10, 
        num_workers=5, 
        data_name="partnet"
    )

    # Model definition
    # model = SPLICE_Module.load_from_checkpoint("checkpoints_splice/best-v1.ckpt", lr=1e-3)
    model = SPLICE_Module()

    # Checkpoint Callback: Save the best model based on a metric and the latest state
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_splice",       # Directory to save checkpoints
        filename="best",                    # Filename prefix for the best model
        save_top_k=1,                       # Keep only the best model
        monitor="total",                    # Metric to monitor (change as needed)
        mode="min",                         # Optimization mode (minimize metric)
        save_last=True                      # Additionally save the latest model state
    )

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=100000,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=7,                          # Number of GPUs to use (or list of indices)
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    # Resume training if a checkpoint exists
    last_ckpt = "checkpoints_splice/last-v8.ckpt" # Example: "splice-1.5-03001627_chair.ckpt"
    
    if last_ckpt:
        print(f"Resuming training from checkpoint: {last_ckpt}")
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        print("Starting training from scratch...")
        trainer.fit(model, datamodule=dm)

    # Output paths after training completion
    print("Best model path:", checkpoint_callback.best_model_path)
    print("Latest model path:", checkpoint_callback.last_model_path)
