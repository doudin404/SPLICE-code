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
    # Set matrix multiplication precision for better performance on modern GPUs
    torch.set_float32_matmul_precision('high')

    # Data module configuration
    files = 'data/03001627_chair_test'  # Replace with the actual data path
    dm = VoxelRayDataModule(
        root_dir=files, 
        batch_size=1, 
        num_workers=5, 
        data_name="partnet"
    )
    
    # Model definition
    # Initializing with specific learning rate and export settings
    model = SPLICE_Module(
        lr=1e-4, 
        export_mode="DIFF", 
        export_shape=False
    )

    # Trainer configuration for inference/prediction
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[7],  # Target GPU index
    )

    # Load from checkpoint and run prediction
    last_ckpt = "./checkpoints_splice/splice-1.5-v1.ckpt"
    if last_ckpt:
        print(f"Loading existing checkpoint: {last_ckpt}")
        trainer.predict(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        print("No checkpoint found, proceeding with default weights.")
        trainer.predict(model, datamodule=dm)
