import random
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import pyvista as pv
import net
import net.PTV3
import net.encoder
import net.model_func
import net.noodles_utils
import net.spareconev
import net.utils
import shape_process
import shape_process.RotationGenerator
import shape_process.gm_transformer
import wandb
import h5py
import os
import pickle

import shape_process.visualizationManager
import model

# wandb.require("core")


class Noodles(nn.Module):
    def __init__(
        self,
        d_model,
        d_point,
        shape_input_channels,
        shape_enc_features,
        shape_spatial_shape,
        mix_nhead,
        mix_nlayer,
        occ_nhead,
        occ_nlayer,
        occ_mlp_size,
        max_parts_length,
        use_ptv3,
        enc_depths,
        enc_num_head,
        stride,
        enc_patch_size,
        freeze_shape_enc=False,
        **cfg,
    ):
        super().__init__()
        self.d_model = d_model
        self.point_enc = net.encoder.PointEncoder(d_point)
        self.gm_enc = net.encoder.GMEncoder(d_model)
        self.pad = net.utils.DataPadder(max_parts_length)
        self.KL_layer = net.utils.KLRegularizationLayer()
        self.voxel_rotator = shape_process.RotationGenerator.VoxelRotator(
            shape_spatial_shape[0]
        )

        if not use_ptv3:
            self.shape_enc = net.spareconev.PartsToVoxelEncoder(
                input_channels=shape_input_channels,
                num_layers=len(shape_enc_features),
                features=shape_enc_features,
                spatial_shape=shape_spatial_shape,
                output_channels=d_model * 2 + 7,
            )
        else:
            # shape_enc_features[-1]=d_model * 2 + 7
            # enc_depths[-1]=0
            self.shape_enc = net.PTV3.PartsToPointTransformerV3Encoder(
                in_channels=shape_input_channels + 3,
                enc_depths=[0, 0] + enc_depths,
                enc_channels=[16, 32] + shape_enc_features,
                enc_num_head=[0, 0] + enc_num_head,
                stride=[2, 2] + stride,
                enc_patch_size=[16, 16] + enc_patch_size,
                out_channels=d_model * 2 + 7,
            )

        # 冻结 shape_enc 模块
        if freeze_shape_enc:
            for param in self.shape_enc.parameters():
                param.requires_grad = False

        self.mix = net.noodles_utils.MixNet(d_model, mix_nhead, mix_nlayer)

        self.occ = net.noodles_utils.OccupancyNet(
            d_model, d_point, occ_nhead, occ_nlayer, occ_mlp_size
        )

    def make_gm(self, pre_gm, center, size, mask):
        basic = shape_process.gm_transformer.quaternion_to_basic(pre_gm[..., -4:])
        gm = torch.cat(
            (center, torch.exp(pre_gm[..., :3]) * size[..., None], basic), dim=-1
        )
        gm[mask] = 0
        quat = torch.cat((center, pre_gm), dim=-1)
        return gm, quat

    def make_gm_with_center(self, pre_gm, mask):
        basic = shape_process.gm_transformer.quaternion_to_basic(pre_gm[..., -4:])
        gm = torch.cat((pre_gm[..., :3], torch.exp(pre_gm[..., 3:6]), basic), dim=-1)
        gm[mask] = 0
        return gm

    def encode_shape(self, voxel, label, center, size, apply_part_rotation=True):
        # Apply random rotation to the input voxel
        rotation_matrices, rotated_voxel = self.voxel_rotator.apply_rotation_batch(
            voxel, apply_rotation=apply_part_rotation
        )
        rotated_voxel = rotated_voxel

        # Pass through shape encoder
        output0 = self.shape_enc(rotated_voxel, label)
        output, mask = self.pad.pad_and_stack(output0)
        center, _ = self.pad.pad_and_stack(center)
        size, _ = self.pad.pad_and_stack(size)
        shape_z = output[..., : self.d_model * 2]
        shape_z, mean, log_var = self.KL_layer(shape_z)
        to_gmm = output[..., self.d_model * 2 :]
        gms, quat = self.make_gm(to_gmm, center, size, mask)

        # Pad and stack rotation matrices
        rotation_matrices_padded, _ = self.pad.pad_and_stack(rotation_matrices)

        return shape_z, gms, mask, mean, log_var, quat, rotation_matrices_padded

    def encode_gm(self, gm, mask):
        gm_z = self.gm_enc(gm, mask)
        return gm_z

    def encode_mix(self, shape_z, gm_z, mask=None):
        z = shape_z + gm_z
        z = self.mix(z, mask)
        z[mask] = 0
        return z

    def decode_occ(self, z0, x, mask):
        x_point = self.point_enc(x)
        x_output, z, attns = self.occ.forward(x_point, z0, mask)
        x_ass = x_output[..., 1:]
        x_occ = x_output[..., 0]
        x_sdf = x_output[..., 1]

        return x_occ, x_sdf, x_ass, z

    def forward(self, voxel, label, center, size, x, transform=True):
        shape_z, gms_grad, mask, mean, log_var, quat, part_rotation = self.encode_shape(
            voxel, label, center, size, apply_part_rotation=transform
        )
        if transform:
            gms_grad = self.voxel_rotator.inverse_transform_gmm(
                gms_grad, part_rotation, mask=mask
            )
        gms = gms_grad.detach()
        if transform:
            gms, x = net.model_func.random_transform(gms, mask, x)
        test_points = x.clone()
        gm_z = self.encode_gm(gms, mask)
        z = self.encode_mix(shape_z, gm_z, mask)
        x_occ, x_sdf, x_ass, x_code = self.decode_occ(z, x, mask)
        attn = torch.einsum("bid,bjd->bij", x_code[..., -100:], z[..., -100:])
        maskinf = torch.zeros_like(mask, dtype=torch.float32)
        maskinf[mask] = -torch.inf
        attn += maskinf[:, None, :]
        return {
            "gms_grad": gms_grad,
            "x_occ": x_occ,
            "x_sdf": x_sdf,
            "x": x_occ,
            "test_points": test_points,
            "x_ass": x_ass,
            "mask": mask,
            "shape_z": shape_z,
            "gm_z": gm_z,
            "gms": gms,
            "z": z,
            "attn": attn,
            "mean": mean,
            "log_var": log_var,
            "quat": quat,
        }


class NoodlesModel(pl.LightningModule):
    def __init__(
        self, batch_size, colors, lr, reg_weight, extract=None, render=False, **cfg
    ):
        super().__init__()
        self.noodles = Noodles(**cfg)
        self.data_to_view = []
        self.batch_size = batch_size
        self.colors = shape_process.visualizationManager.VisualizationManager.colors
        random.shuffle(colors)
        self.random_color = colors[:20]
        self.lr = lr
        self.reg_weight = reg_weight
        self.start_epoch = None
        self.extract = extract
        self.render = render
        self.data_to_extract = []

    def forward(self, voxel, label, center, size, x, transform=False):
        return self.noodles(voxel, label, center, size, x, transform=transform)

    def gms_loss(self, gms, parts_points):

        batch_lose = []
        for i in range(len(parts_points)):
            part_loss = []
            for j in range(len(parts_points[i])):
                part_loss.append(
                    net.model_func.gm_loss(
                        gms[i : i + 1, j], parts_points[i][j].unsqueeze(0)
                    )
                )
            batch_lose.append(torch.sum(torch.stack(part_loss)) / len(parts_points[i]))
        gms_loss = torch.sum(torch.stack(batch_lose)) / len(parts_points)

        return gms_loss

    def occ_loss(self, y_sdf, y_sdf_gt):
        return net.model_func.occ_loss(y_sdf, y_sdf_gt), net.model_func.sdf_accuracy(
            y_sdf, y_sdf_gt
        )

    def sdf_loss(self, y_sdf, y_sdf_gt):
        return net.model_func.sdf_loss(y_sdf, y_sdf_gt), net.model_func.sdf_accuracy(
            y_sdf, y_sdf_gt
        )

    def ass_loss(self, y_ass, gms, train_points, mask):
        gms = gms.detach()
        assist_points = net.model_func.make_assist_points(
            gms, train_points, mask
        )
        assist_points = assist_points.detach()
        y_ass_flat = y_ass[..., :3].view(-1, 3)
        assist_points_flat = assist_points.view(-1, 3)

        non_zero_mask = torch.any(assist_points_flat != 0, dim=1)
        valid_y_ass = y_ass_flat[non_zero_mask]
        valid_assist_point = assist_points_flat[non_zero_mask]

        if valid_assist_point.numel() > 0:
            mark_loss = net.model_func.min_mse_loss(valid_y_ass, valid_assist_point)
            mark_loss = torch.mean(mark_loss)
        else:
            mark_loss = torch.tensor(0.0)
        return mark_loss

    def attn_loss(
        self,
        attn,
        gms,
        train_points,
        mask,
        y_sdf_gt,
        positive_weight=0.1,
        negative_weight=1.0,
    ):
        gms = gms.detach()

        # Create attention labels
        attn_labels = net.model_func.make_attn_labels(gms, train_points, mask)

        mark_loss_sum = torch.tensor(0.0, device=attn.device)
        accuracy = torch.tensor(0.0, device=attn.device)

        # Flatten attention and labels
        attn_flat = attn.reshape(-1, attn.shape[-1])
        attn_labels_flat = attn_labels.view(-1)

        valid_mask = attn_labels_flat != -1

        valid_attn = attn_flat[valid_mask]
        valid_labels = attn_labels_flat[valid_mask]
        valid_y_sdf_gt = y_sdf_gt.view(-1)[valid_mask]

        if valid_labels.numel() > 0:
            # Get mask information to determine non-masked elements
            positive_sdf_mask = valid_y_sdf_gt > 0

            # target = target.softmax(dim=-1)
            np_valid_attn = valid_attn[~positive_sdf_mask]
            np_attn_labels = valid_labels[~positive_sdf_mask]
            np_loss = F.cross_entropy(
                np_valid_attn, np_attn_labels, reduction="none", ignore_index=-1
            )
            np_acc = np_valid_attn.argmax(-1) == np_attn_labels
            p_valid_attn = valid_attn[positive_sdf_mask]

            inf_mask=p_valid_attn == float("-inf")
            p_label = 1-inf_mask.to(torch.float)
            p_attn_labels=p_label/p_label.sum(-1,keepdim=True)
            p_valid_attn[inf_mask] = -1e10
            p_loss = F.cross_entropy(
                p_valid_attn, p_attn_labels, reduction="none"
            )
            p_acc=(p_loss<0.5)

            mark_loss = (
                p_loss.sum() * positive_weight + np_loss.sum() * negative_weight
            ) / valid_attn.shape[0]
            mark_loss_sum += mark_loss

        return mark_loss_sum, p_acc.float().mean(), np_acc.float().mean()

    def kl_loss(self, mean, log_var):
        # 计算KL散度
        kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())

        # 计算每行中KL散度小于0.5的维度的个数
        kl_under_threshold = (kl_div < 0.5).float().sum(dim=-1)

        # 计算平均每行中散度小于0.5的维度的个数
        kl_act = kl_under_threshold.mean()

        # 返回KL散度的均值和每行散度小于0.5的维度的平均个数
        return kl_div.mean(), kl_act

    def compute_loss(
        self,
        outputs,
        batch,
        detail=True,
    ):
        gms_loss = self.gms_loss(outputs["gms_grad"], batch["parts_obb_voxel"])
        occ_loss, occ_acc = self.occ_loss(outputs["x_occ"], batch["sample_train_sdfs"])
        # sdf_loss, sdf_acc = self.sdf_loss(outputs["x_sdf"], batch["sample_train_sdfs"])

        attn_loss, p_acc, np_acc = self.attn_loss(
            outputs["attn"],
            outputs["gms"],
            outputs["test_points"],
            outputs["mask"],
            batch["sample_train_sdfs"],
        )
        # attn_labels[batch["sample_train_sdfs"]>0]=-1
        # ass_loss = self.ass_loss(
        #     outputs["x_ass"], outputs["gms"], outputs["test_points"], outputs["mask"]
        # )
        kl_loss, kl_act = self.kl_loss(outputs["mean"], outputs["log_var"])

        loss = (
            gms_loss * 100
            + occ_loss * 1
            # + sdf_loss * 0
            #+ ass_loss*5
            + attn_loss * 0#0.1
            + kl_loss * 0.001
        )
        return {
            "loss": loss,
            "gms_loss": gms_loss,
            "occ_loss": occ_loss,
            "occ_acc": occ_acc,
            # "sdf_loss": sdf_loss,
            # "sdf_acc": sdf_acc,
            #"ass_loss": ass_loss,
            "attn_loss": attn_loss,
            "kl_loss": kl_loss,
            "kl_act": kl_act,
            "p_acc":p_acc,
            "np_acc": np_acc,
        }

    def training_step(self, batch, batch_idx):
        outputs = self(
            batch["parts_voxel_indices"],
            batch["parts_voxel_labels"],
            batch["parts_center"],
            batch["parts_patch_size"],
            batch["sample_train_points"],
            transform=True,
        )
        net.model_func.save_low_accuracy_shapes()
        losses = self.compute_loss(outputs, batch)

        loss_keys = [
            "loss",
            "gms_loss",
            "occ_loss",
            "occ_acc",
            # "sdf_loss",
            # "sdf_acc",
            # "ass_loss",
            "attn_loss",
            "kl_act",
            "kl_loss",
            "p_acc",
            "np_acc",
        ]
        prog_bar_keys = ["occ_acc","occ_loss"]#, "ass_loss"]
        on_step_keys = ["occ_acc","occ_loss"]#, "ass_loss"]

        for key in loss_keys:
            self.log(
                key,
                losses[key],
                prog_bar=key in prog_bar_keys,
                batch_size=self.batch_size,
                on_epoch=True,
                on_step=key in on_step_keys,
                sync_dist=True,
            )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self(
            batch["parts_voxel_indices"],
            batch["parts_voxel_labels"],
            batch["parts_center"],
            batch["parts_patch_size"],
            batch["sample_train_points"],
            transform=True,
        )

        losses = self.compute_loss(outputs, batch)
        if batch_idx == 0:
            for i in range(min(4, len(batch["id"]))):
                self.data_to_view.append(
                    {
                        "test_points": outputs["test_points"][i],
                        "gt_labels": batch["sample_train_sdfs"][i],
                        "pred_gms": outputs["gms_grad"][i],
                        "pred_labels": outputs["x"][i],
                        "parts_obb_voxel": batch["parts_obb_voxel"][i],
                        "trans_gms": outputs["gms"][i],
                    }
                )

        # self.log(
        #     "val_occ_loss",
        #     losses["occ_loss"],
        #     batch_size=self.batch_size,
        #     on_epoch=True,
        #     on_step=False,
        # )

    def on_validation_epoch_end(self):
        if self.start_epoch is None:
            self.start_epoch = self.current_epoch
        if self.current_epoch % 1 == 0 or self.current_epoch - self.start_epoch < 10:
            random.shuffle(self.colors)
            self.random_color = self.colors[:20]
            # 使用保存的数据进行可视化
            for i, sample in enumerate(self.data_to_view):
                plotter = self.visualize_points_and_gmms(**self.data_to_view[i])
                try:
                    image_path = f"picture/visualization_epoch{self.current_epoch}_rank{self.trainer.local_rank}_sample{i}.png"
                    plotter.screenshot(image_path)
                    plotter.close()
                    wandb.log({f"visualization_sample{i}": wandb.Image(image_path)})
                except Exception as e:
                    print(e)

        # 清空数据以备下一次验证
        self.data_to_view = []

    def configure_optimizers(self):
        # 定义优化器
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.reg_weight
        )

        # 定义学习率的 warm-up 调度器
        def lr_lambda(step):
            if step < 100:
                # 在前100步进行 warm-up，学习率从 0 增长到 self.lr
                return step / 100
            # 100步之后学习率保持在 self.lr
            return 1.0

        # 使用 LambdaLR 调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 返回优化器和调度器
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def test_step(self, batch, batch_idx):
        shape_z_, gms_grad, mask, shape_z, log_var, quat,_ = self.noodles.encode_shape(
            batch["parts_voxel_indices"],
            batch["parts_voxel_labels"],
            batch["parts_center"],
            batch["parts_patch_size"],
        )
        gms = gms_grad.detach()
        gm_z = self.noodles.encode_gm(gms, mask)

        if self.extract is not None:
            # for i in range(gms.shape[0]):
            #     vm = shape_process.visualizationManager.VisualizationManager(
            #         off_screen=False,shape=[1,2]
            #     )
            #     vm.add_axes_at_origin(subplot=(0, 0))
            #     for j, gmm in enumerate(gms[i]):
            #         if not (gmm==0).all():
            #             vm.add_gmm(gmm, subplot=(0, 0))
            #     vm.add_axes_at_origin(subplot=(0, 1))
            #     for j, points in enumerate(batch['parts_obb_voxel'][i]):
            #         vm.add_points(points, subplot=(0, 1))
            #     vm.show()
            # # 如果设置为提取模式，将数据保存到data_to_extract
            self.data_to_extract.append((batch["id"][0], gm_z[0], shape_z[0]))
        if self.render:
            meshs = self.decode(gm_z, shape_z, mask)
            for i, mesh in enumerate(meshs):
                if len(meshs) == 1:
                    mesh.export(f"mesh/{batch['id'][0]}.obj")
                else:
                    mesh.export(f"mesh/{batch['id'][0]}-{i}.obj")

    def decode(self, gm, shape_z, mask, res=128):
        if gm.shape[-1] != shape_z.shape[-1]:
            gm_z = self.noodles.encode_gm(gm, mask)
        else:
            gm_z=gm
        z = self.noodles.encode_mix(shape_z, gm_z, mask)

        meshs = []

        for i in range(z.shape[0]):

            # vm = shape_process.visualizationManager.VisualizationManager(
            #     off_screen=False
            # )
            # vm.add_axes_at_origin(subplot=(0, 0))
            # for j, gmm in enumerate(gm[i]):
            #     if not mask[i][j]:
            #         vm.add_gmm(gmm, subplot=(0, 0), color=self.random_color[i])
            # vm.show()

            def occ_func(x):
                y, _, _, _ = self.noodles.decode_occ(z[i : i + 1], x, mask[i : i + 1])
                return F.sigmoid(y)

            meshlizer = net.utils.OccMeshlizer(occ_func)

            # 生成方阵点云
            grid_points = meshlizer.generate_grid_points(size=res).to(device=z.device)################################

            # 使用模型进行扫描
            meshs.append(meshlizer.get_mesh_from_prediction(grid_points, size=res))

        return meshs

    def on_test_epoch_end(self):
        self.extract_data()

    def visualize_points_and_gmms(
        self,
        test_points,
        gt_labels,
        pred_gms,
        pred_labels,
        parts_obb_voxel,
        trans_gms,
    ):

        vm = shape_process.visualizationManager.VisualizationManager(
            shape=(2, 2), off_screen=True
        )

        # Visualize real data
        vm.add_axes_at_origin(subplot=(0, 0))
        vm.add_points(
            test_points,
            gt_labels,
            subplot=(0, 0),
            point_size=1,
            colors_to_show=["red", "green"],
        )
        # for gmm in gmms:
        #     if not (gmm == 0).all():
        #         vm.add_gmm_axes(gmm, subplot=(0, 0))

        # Visualize prediction data
        vm.add_axes_at_origin(subplot=(0, 1))
        if pred_labels is not None:
            []
            # pred_labels = F.sigmoid(pred_labels)
        else:
            pred_labels = gt_labels == True  # Using true labels as default predictions
        vm.add_points(
            test_points,
            pred_labels,
            subplot=(0, 1),
            point_size=1,
            colors_to_show=["red", "green"],
        )

        vm.add_axes_at_origin(subplot=(0, 1))
        for i, gmm in enumerate(trans_gms):
            if not (gmm == 0).all():
                vm.add_gmm_axes(gmm, subplot=(0, 1), color=self.random_color[i])

        vm.add_axes_at_origin(subplot=(1, 0))
        for i, parts in enumerate(parts_obb_voxel):
            vm.add_points(parts, subplot=(1, 0), color=self.random_color[i])

        vm.add_axes_at_origin(subplot=(1, 1))
        for i, gmm in enumerate(pred_gms):
            #gmm[3:6]*=2
            if not (gmm == 0).all():
                vm.add_gmm(gmm, subplot=(1, 1), color=self.random_color[i])

        # vm.add_axes_at_origin(subplot=(1, 1))
        # for i, gmm in enumerate(pred_gms):
        #     if not (gmm == 0).all():
        #         vm.add_points(gmm[None,:3], subplot=(1, 1), color=self.random_color[i],point_size=10)
        vm.show()
        return vm.plotter

    def extract_data(self):
        if not self.extract or not self.data_to_extract:
            return

        extract_dir = "extract"
        os.makedirs(extract_dir, exist_ok=True)

        if "h5" in self.extract:
            net.model_func.save_h5(self.data_to_extract, extract_dir)

        if "txt_pkl" in self.extract:
            net.model_func.save_txt_pkl(self.data_to_extract, extract_dir)

        if "id_to_code" in self.extract:
            net.model_func.save_ids_txt(self.data_to_extract, extract_dir)

        self.data_to_extract.clear()



def model_load(cfg):
    model = NoodlesModel.load_from_checkpoint(
        cfg.ui.model_path,
        **cfg.train,
        map_location=lambda storage, loc: storage.cuda(0),
    )
    return model
