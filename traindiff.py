import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data_process.diff_data import DiffDataModule
from model.DIFF_model import SPLICE_Diffusion
import multiprocessing as mp
import math

def transform_data(data):
    """
    Apply random scaling and feature normalization to the input data.
    """
    # Generate a random scale factor between 0.8 and 1.2
    scale = torch.rand(1).item() * 0.4 + 0.8
    
    # Scale specific coordinate or spatial features
    data[:, 1:4] *= scale * 5
    data[:, 13:16] *= scale * 5
    
    # Apply weight adjustments to other feature dimensions
    data[:, 16:] *= 0.1
    data[:, 4:13] *= 1
    return data

# ----------------- Execution Entry Point -----------------
if __name__ == '__main__':
    # Optimize matrix multiplication precision for Ampere/Hopper GPUs
    torch.set_float32_matmul_precision('high')

    # Logger configuration
    tb_logger = TensorBoardLogger(save_dir="logs/splice_log", name="partnet")

    # Data Module initialization
    files = 'data/chair_feats_5'  # Replace with the actual data path
    dm = DiffDataModule(
        data_dir=files, 
        batch_size=256, 
        num_workers=5, 
        transform=transform_data
    )

    # Model definition
    # model = SPLICE_Diffusion.load_from_checkpoint("checkpoints_splice_diff/overfit_3.ckpt")
    model = SPLICE_Diffusion()

    # Checkpoint Callback: Save the best model and the latest training state
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_splice_diff",  # Directory for saving checkpoints
        filename="best_overfit_5",           # Filename prefix for the best model
        save_top_k=1,                        # Keep only the single best model
        monitor="val_loss",                  # Metric to monitor
        mode="min",                          # Optimization direction (minimize loss)
        save_last=True                       # Save a 'last.ckpt' automatically
    )
    
    # Manually setting the 'last' checkpoint filename pattern
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"overfit_5"

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=-1,                       # Train indefinitely until manual stop
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[2, 3],                      # Indices of GPUs to use
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=100          # Frequency of validation checks
    )

    # Resume training if a checkpoint path is provided
    last_ckpt = "checkpoints_splice_diff/overfit_5.ckpt"
    if last_ckpt:
        print(f"Loading existing checkpoint: {last_ckpt}")
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        print("Starting training from scratch...")
        trainer.fit(model, datamodule=dm)

    # Summary of saved model paths
    print("Best model path:", checkpoint_callback.best_model_path)
    print("Latest model path:", checkpoint_callback.last_model_path)
