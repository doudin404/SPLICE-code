import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.optim import Adam
import data_process.data_module
import hydra
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Step 1: Define the MLP Model using PyTorch Lightning
class MLPModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch[...,15:512],batch[...,:15]
        y_hat = self(x+torch.randn_like(x)*0.06)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss,prog_bar=True,on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[...,15:512],batch[...,:15]
        y_hat = self(x+torch.randn_like(x)*0.06)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss,prog_bar=True,on_epoch=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

# Step 4: Train the model
@hydra.main(config_path="config", config_name="config_fry", version_base="1.2")
def main(cfg):
    # Create the dataset
    data_module = data_process.data_module.FryDataModule(**cfg.fry)

    # Define the model
    input_size = 512-15#1024-15
    hidden_size = 512
    output_size = 15
    model = MLPModel(input_size, hidden_size, output_size)
    latest_model_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,  # 只保留最新的一个模型
        dirpath="checkpoints/trick/",  # 保存的路径
        filename="latest_model",  # 保存的文件名
        every_n_epochs=1,  # 每个 epoch 都保存一次
    )
    wandb_logger = WandbLogger(project="NoodlesProject_vvv")
    # Setup the trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=wandb_logger,
        log_every_n_steps=50,
        callbacks=[latest_model_checkpoint_callback],
    )  # Set gpus=1 if GPU is available

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Step 5: Inference
    model.eval()
    test_input = torch.tensor([[0.5]])
    prediction = model(test_input)
    print(f"Prediction for input 0.5: {prediction.item()}")

if __name__ == "__main__":
    main()

def translate_load(cfg):
    model = MLPModel.load_from_checkpoint(
        cfg.ui.translate_path,
        input_size = 512-15,#1024-15
        hidden_size = 512,
        output_size = 15,
        map_location=lambda storage, loc: storage.cuda(0),
    )
    return model