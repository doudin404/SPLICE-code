import torch
from torch import nn
from net.latent_array_transformer import LatentArrayTransformer
from net.fried_util import edm_sampler
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
import math
import random
import matplotlib.pyplot as plt
import numpy as np

from model.noodles import NoodlesModel
import shape_process
import shape_process.visualizationManager


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        n_latents=20,
        channels=8,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
        n_heads=8,
        d_head=64,
        depth=12,
        inner_dim=1024,
        gms_factor=1,
        shape_z_factor=1,
        # depth = 6,

    ):
        super().__init__()
        self.n_latents = n_latents
        #self.n_latents = 20
        self.channels = channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.gms_factor = gms_factor
        self.shape_z_factor = shape_z_factor


        self.model = LatentArrayTransformer(
            in_channels=channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            inner_dim=inner_dim
        )

        #self.category_emb = nn.Embedding(55, n_heads * d_head)

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):

        #self.gms_factor 
        #self.shape_z_factor 
        #x =torch.cat([x[...,:512]/self.gms_factor,x[...,512:]/self.shape_z_factor ],dim=-1)

        if class_labels is None:
            cond_emb = None
        else:
            if class_labels.dtype == torch.float32:
                cond_emb = class_labels
            else:
                cond_emb = self.category_emb(class_labels).unsqueeze(1)

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(
            (c_in * x).to(dtype), c_noise.flatten(), cond=cond_emb, **model_kwargs
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)#.to(dtype=torch.float32,device="cuda:0")

    @torch.no_grad()
    def sample(self, cond, batch_seeds=None):
        # print(batch_seeds)
        if cond is not None:
            batch_size, device = *cond.shape, cond.device
            if batch_seeds is None:
                batch_seeds = torch.arange(batch_size)
        else:
            device = "cuda:1"#batch_seeds.device
            batch_size = 1#batch_seeds.shape[0]

        # batch_size, device = *cond.shape, cond.device
        # batch_seeds = torch.arange(batch_size)

        # rnd = StackedRandomGenerator(device, batch_seeds)
        latents = torch.randn(
            [batch_size, self.n_latents, self.channels], device=device
        )
        x=edm_sampler(self, latents, cond)
        return torch.cat([x[...,:-512]*self.gms_factor,x[...,-512:]*self.shape_z_factor ],dim=-1)
    
        
    @torch.no_grad()
    def mask_sample(self, cond,mask,is_gen=True,batch_seeds=None):
        # print(batch_seeds)
        if cond is not None:
            batch_size, device = cond.shape, cond.device
            # if batch_seeds is None:
            #     batch_seeds = torch.arange(batch_size)

        # batch_size, device = *cond.shape, cond.device
        # batch_seeds = torch.arange(batch_size)

        # rnd = StackedRandomGenerator(device, batch_seeds)
        latents = torch.randn(
            batch_size, device=device
        )

        return edm_sampler(self, latents, None,sigma_max=80 if is_gen else 0.01,cond=cond,mask=mask,num_steps=18*5 if is_gen else 18*2)


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.2179,gms_factor=1,shape_z_factor=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_history = []
        self.loss_history = []
        self.gms_factor = gms_factor
        self.shape_z_factor = shape_z_factor

    def scale_tensor(self,tensor, lower_bound, upper_bound):
        min_val = tensor.min()
        max_val = tensor.max()
        # 避免除以零的情况
        if max_val == min_val:
            return torch.full_like(tensor, lower_bound + (upper_bound - lower_bound) / 2)
        
        scaled_tensor = lower_bound + (tensor - min_val) * (upper_bound - lower_bound) / (max_val - min_val)
        return scaled_tensor

    def __call__(self, net, inputs, labels=None, augment_pipe=None, weight_mask=None):

        inputs =torch.cat([inputs[...,:-512]/self.gms_factor,inputs[...,-512:]/self.shape_z_factor ],dim=-1)


        rnd_normal = torch.randn([inputs.shape[0], 1, 1], device=inputs.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # # 生成随机的 step_indices，不需要是整数，范围在 [0, num_steps)
        # rho=7;sigma_min=0.002;sigma_max=80;num_steps=180
        # step_indices = torch.rand([inputs.shape[0], 1, 1], dtype=torch.float64, device=inputs.device) * (180)

        # # 计算 t_steps
        # sigma = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho



        # sigma=self.scale_tensor(sigma,0.002, 80)

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(inputs) if augment_pipe is not None else (inputs, None)
        )

        n = torch.randn_like(y) * sigma

        D_yn = net(y + n, sigma, labels)
        # if weight_mask is None:
        #     weight_mask = torch.ones_like(D_yn[0,0], device=inputs.device)
        #     weight_mask[:15]==2
        #     weight_mask[15:]=0.5
    
        loss = weight * ((D_yn - y) ** 2)#*(1+(y==0).all(-1,keepdim=True)*-0.9)
        
        # # 记录 rnd_normal 和 loss
        # self.sigma_history.extend(sigma.view(-1).cpu().detach().numpy())
        # self.loss_history.extend((loss ** 2).mean(-1).mean(-1).view(-1).cpu().detach().numpy())

        return loss.mean()

    def plot_history(self):
        if len(self.sigma_history) == 0 or len(self.loss_history) == 0:
            print("No data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.scatter(self.sigma_history[:2000], self.loss_history[:2000], alpha=0.5)
        plt.xlabel('sigma')
        plt.ylabel('Loss')
        plt.title('sigma vs Loss')
        # 设置原点为 (0, 0)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.show()
        self.sigma_history = []
        self.loss_history = []


def kl_d512_m512_l8_d24_edm(sigma_data,data_len,inner_dim,gms_factor=1,shape_z_factor=1):
    model = EDMPrecond(n_latents=20, channels=data_len, depth=4,n_heads=8,sigma_data=sigma_data,inner_dim=inner_dim,gms_factor=gms_factor,shape_z_factor=shape_z_factor)#512+512
    return model


class FryModel(pl.LightningModule):
    def __init__(self, lr, reg_weight, sigma_data=1,ae=None,gf=None,data_len=527,inner_dim=1024,gms_factor=1,shape_z_factor=1, **args):
        super(FryModel, self).__init__()
        self.model = kl_d512_m512_l8_d24_edm(sigma_data=sigma_data, data_len=data_len,inner_dim=inner_dim,gms_factor=gms_factor,shape_z_factor=shape_z_factor)
        self.ae = [ae,gf]
        self.criterion = EDMLoss(sigma_data=sigma_data,gms_factor=gms_factor,shape_z_factor=shape_z_factor)
        self.args = args
        self.lr = lr
        self.reg_weight = reg_weight
        self.gms_factor = gms_factor
        self.shape_z_factor = shape_z_factor

        # 冻结 ae 的参数
        if self.ae[0]!=None:
            for param in self.ae[0].parameters():
                param.requires_grad = False
                # 冻结 ae 的参数
        if self.ae[1]!=None:
            for param in self.ae[1].parameters():
                param.requires_grad = False

    def split_gms_shapez(self, data):
        """
        将连接后的数据重新拆分回 gms 和 shape_z.
        输入数据格式：[sample_num, len, 15+512]
        返回两个部分，分别是 gms 和 shape_z
        """
        pre_gms = data[:, :, :-512]
        shape_z = data[:, :,-512:]
        mask = shape_z.norm(2,-1)<1
        #ms =  self.ae[0].noodles.make_gm_with_center(pre_gms,mask)
        return pre_gms, shape_z, mask
    
    @torch.no_grad()
    def sample(self, cond=None, batch_seeds=None, id=None):
        if id is None:
            id = random.random()
        result = self.model.sample(cond, batch_seeds).to(torch.float32)
        gm=result[...,:15]
        for i in range(gm.shape[0]):

            vm = shape_process.visualizationManager.VisualizationManager(
                off_screen=False
            )
            vm.add_axes_at_origin(subplot=(0, 0))
            for j, gmm in enumerate(gm[i]):
                if not (gmm==0).all():
                    vm.add_gmm(gmm, subplot=(0, 0))
            vm.show()
        gms_z, shape_z, mask = self.split_gms_shapez(result)  
        #gm = self.ae[1](gms_z[...,15:512])
        return self.ae[0].decode(gms_z[...,:], shape_z, mask)#[0],self.ae[0].decode(gms_z, shape_z, mask)[0]

    def forward(self, x, categories):
        return self.model(x, categories)

    def training_step(self, batch, batch_idx):
        x = batch

        # for i in range(x.shape[0]):

        #     vm = shape_process.visualizationManager.VisualizationManager(
        #         off_screen=False
        #     )
        #     vm.add_axes_at_origin(subplot=(0, 0))
        #     for j, gmm in enumerate(x[...,:15][i]):
        #         if not (gmm==0).all():
        #             vm.add_gmm(gmm, subplot=(0, 0))
        #     vm.show()
        loss = self.criterion(self.model, x)
        loss_value = loss.item()

        # Log loss to TensorBoard
        self.log(
            "train_loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss
        
    # def on_train_epoch_end(self):
    #     self.criterion.plot_history()

    def validation_step(self, batch, batch_idx):
        x = batch

        #x =torch.cat([x[...,:512]*self.gms_factor,x[...,512:]*self.shape_z_factor],dim=-1)

        loss = self.criterion(self.model, x)
        loss_value = loss.item()

        # Log validation loss to TensorBoard
        self.log(
            "val_loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.reg_weight
        )
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):



        non_zero_mask = torch.any(batch != 0, dim=-1)  # [batch, len]

        # Step 2: 生成遮蔽前512维的布尔张量 [1024]
        mask_512 = torch.zeros(1024, dtype=torch.bool, device=batch.device)
        mask_512[:512] = 1  # 前512个位置设为True

        # Step 3: 使用广播乘法 [batch, len, 1024]
        # 使用unsqueeze将mask_512扩展到[1, 1, 1024]，以便进行广播
        mask = non_zero_mask.unsqueeze(-1) * mask_512

        
        result = self.model.mask_sample(cond=batch,mask=mask).to(torch.float32)
        gms_z, shape_z, mask = self.split_gms_shapez(result)
        self.ae[0].decode(gms_z, shape_z, mask, id=f"{batch_idx}_result")
        gms_z, shape_z, mask = self.split_gms_shapez(batch)
        self.ae[0].decode(gms_z, shape_z, mask, id=f"{batch_idx}_gt")

        return
    

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_scheduler)
    #     return [optimizer], [scheduler]

    # def lr_scheduler(self, epoch):
    #     """Custom learning rate scheduler"""
    #     warmup_epochs=10
    #     epochs=100
    #     if epoch < warmup_epochs:
    #         lr_scale = epoch / warmup_epochs
    #     else:
    #         lr_scale = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    #     return lr_scale * self.lr

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #     optimizer.step(closure=optimizer_closure)
    #     self.log("lr", optimizer.param_groups[0]["lr"], prog_bar=True, logger=True)
def adjust_load(cfg):
    model = FryModel.load_from_checkpoint(
        cfg.ui.adjust_path,
        data_len=1024,
        inner_dim=1024,
        sigma_data=cfg.ui.adjust_sigma,
        **cfg.fry,
        map_location=lambda storage, loc: storage.cuda(0),
    )
    return model

def pose_load(cfg):
    model = FryModel.load_from_checkpoint(
        cfg.ui.pose_path,
        data_len=15,
        inner_dim=512,
        sigma_data=cfg.ui.pose_sigma,
        **cfg.fry,
        map_location=lambda storage, loc: storage.cuda(0),
    )
    return model