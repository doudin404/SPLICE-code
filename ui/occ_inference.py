import torch

# from utils import train_utils, mcubes_meshing, files_utils, mesh_utils

# from models.occ_gmm import Spaghetti
# from models import models_utils
# todo 连接模型
from model.noodles import model_load
from model.fried_noodles import adjust_load, pose_load
from trick_mian import translate_load


class Inference:

    # def get_occ_fun(self, z):

    #     def forward(x):
    #         nonlocal z
    #         x = x.unsqueeze(0)
    #         out = self.model.occupancy_network(x, z)[0, :]
    #         out = 2 * out.sigmoid_() - 1
    #         return out

    #     if z.dim() == 2:
    #         z = z.unsqueeze(0)
    #     return forward

    # @torch_no_grad
    # def get_mesh(self, z, res: int):
    #     mesh = self.meshing.occ_meshing(self.get_occ_fun(z), res=res)
    #     return mesh

    # def plot_occ(
    #     self, z, z_base, gmms, fixed_items, folder_name: str, res=200, verbose=False
    # ):
    #     for i in range(len(z)):
    #         mesh = self.get_mesh(z[i], res)
    #         name = f"{fixed_items[i]:04d}"
    #         if mesh is not None:
    #             files_utils.export_mesh(
    #                 mesh, f"{self.opt.cp_folder}/{folder_name}/occ/{name}"
    #             )
    #             files_utils.save_pickle(
    #                 z_base[i].detach().cpu(),
    #                 f"{self.opt.cp_folder}/{folder_name}/occ/{name}",
    #             )
    #             if gmms is not None:
    #                 files_utils.export_gmm(
    #                     gmms, i, f"{self.opt.cp_folder}/{folder_name}/occ/{name}"
    #                 )
    #         if verbose:
    #             print(f"done {i + 1:d}/{len(z):d}")

    # def load_file(self, info_path, disclude=None):
    #     info = files_utils.load_pickle("".join(info_path))
    #     keys = list(info["ids"].keys())
    #     items = map(lambda x: int(x.split("_")[1]) if type(x) is str else x, keys)
    #     items = torch.tensor(list(items), dtype=torch.int64, device=self.device)
    #     zh, _, gmms_sanity, _ = self.model.get_embeddings(items)
    #     gmms = [item for item in info["gmm"]]
    #     zh_ = []
    #     split = []
    #     gmm_mask = torch.ones(gmms[0].shape[2], dtype=torch.bool)
    #     counter = 0
    #     # gmms_ = [[] for _ in range(len(gmms))]
    #     for i, key in enumerate(keys):
    #         gaussian_inds = info["ids"][key]
    #         if disclude is not None:
    #             for j in range(len(gaussian_inds)):
    #                 gmm_mask[j + counter] = gaussian_inds[j] not in disclude
    #             counter += len(gaussian_inds)
    #             gaussian_inds = [ind for ind in gaussian_inds if ind not in disclude]
    #             info["ids"][key] = gaussian_inds
    #         gaussian_inds = torch.tensor(gaussian_inds, dtype=torch.int64)
    #         zh_.append(zh[i, gaussian_inds])
    #         split.append(
    #             len(split)
    #             + torch.ones(
    #                 len(info["ids"][key]), dtype=torch.int64, device=self.device
    #             )
    #         )
    #     zh_ = torch.cat(zh_, dim=0).unsqueeze(0).to(self.device)
    #     gmms = [item[:, :, gmm_mask].to(self.device) for item in info["gmm"]]
    #     return zh_, gmms, split, info["ids"]

    # @torch_no_grad
    # def get_z_from_file(self, info_path):
    #     zh_, gmms, split, _ = self.load_file(info_path)
    #     zh_ = self.model.merge_zh_step_a(zh_, [gmms])
    #     zh, _ = self.model.affine_transformer.forward_with_attention(zh_)
    #     # gmms_ = [torch.cat(item, dim=1).unsqueeze(0) for item in gmms_]
    #     # zh, _ = self.model.merge_zh(zh_, [gmms])
    #     return zh, zh_, gmms, torch.cat(split)

    # def plot_from_info(self, info_path, res):
    #     zh, zh_, gmms, split = self.get_z_from_file(info_path)
    #     mesh = self.get_mesh(zh[0], res, gmms)
    #     if mesh is not None:
    #         attention = self.get_attention_faces(mesh, zh, fixed_z=split)
    #     else:
    #         attention = None
    #     return mesh, attention

    # @staticmethod
    # def combine_and_pad(zh_a, zh_b):
    #     if zh_a.shape[1] == zh_b.shape[1]:
    #         mask = None
    #     else:
    #         pad_length = max(zh_a.shape[1], zh_b.shape[1])
    #         mask = torch.zeros(2, pad_length, device=zh_a.device, dtype=torch.bool)
    #         padding = torch.zeros(
    #             1,
    #             abs(zh_a.shape[1] - zh_b.shape[1]),
    #             zh_a.shape[-1],
    #             device=zh_a.device,
    #         )
    #         if zh_a.shape[1] > zh_b.shape[1]:
    #             mask[1, zh_b.shape[1] :] = True
    #             zh_b = torch.cat((zh_b, padding), dim=1)
    #         else:
    #             mask[0, zh_a.shape[1] :] = True
    #             zh_a = torch.cat((zh_a, padding), dim=1)
    #     return torch.cat((zh_a, zh_b), dim=0), mask

    # @staticmethod
    # def get_intersection_z(z_a, z_b):
    #     diff = (z_a[0, :, None, :] - z_b[0, None]).abs().sum(-1)
    #     diff_a = diff.min(1)[0].lt(0.1)
    #     diff_b = diff.min(0)[0].lt(0.1)
    #     if diff_a.shape[0] != diff_b.shape[0]:
    #         padding = torch.zeros(
    #             abs(diff_a.shape[0] - diff_b.shape[0]),
    #             device=z_a.device,
    #             dtype=torch.bool,
    #         )
    #         if diff_a.shape[0] > diff_b.shape[0]:
    #             diff_b = torch.cat((diff_b, padding))
    #         else:
    #             diff_a = torch.cat((diff_a, padding))
    #     return torch.cat((diff_a, diff_b))

    # def get_attention_points(self, vs, zh, mask=None, alpha=None):
    #     vs = vs.unsqueeze(0)
    #     attention = self.model.occupancy_network.forward_attention(
    #         vs, zh, mask=mask, alpha=alpha
    #     )
    #     attention = torch.stack(attention, 0).mean(0).mean(-1)
    #     attention = attention.permute(1, 0, 2).reshape(attention.shape[1], -1)
    #     attention_max = attention.argmax(-1)
    #     return attention_max

    # @torch_no_grad
    # def get_attention_faces(self, mesh, zh, mask=None, fixed_z=None, alpha=None):
    #     coords = mesh[0][mesh[1]].mean(1).to(zh.device)

    #     attention_max = self.get_attention_points(coords, zh, mask, alpha)
    #     if fixed_z is not None:
    #         attention_select = fixed_z[attention_max].cpu()
    #     else:
    #         attention_select = attention_max
    #     return attention_select

    # @torch_no_grad
    # def plot_folder(self, *folders, res: int = 256):
    #     logger = train_utils.Logger()
    #     for folder in folders:
    #         paths = files_utils.collect(folder, ".pkl")
    #         logger.start(len(paths))
    #         for path in paths:
    #             name = path[1]
    #             out_path = f"{self.opt.cp_folder}/from_ui/{name}"
    #             mesh, colors = self.plot_from_info(path, res)
    #             if mesh is not None:
    #                 files_utils.export_mesh(mesh, out_path)
    #                 files_utils.export_list(colors.tolist(), f"{out_path}_faces")
    #             logger.reset_iter()
    #         logger.stop()

    # def get_zh_from_idx(self, items):
    #     zh, _, gmms, __ = self.model.get_embeddings(items.to(self.device))
    #     zh, attn_b = self.model.merge_zh(zh, gmms)
    #     return zh, gmms

    @property
    def device(self):
        return self.model.device

    # def get_new_ids(self, folder_name, nums_sample):
    #     names = [
    #         int(path[1])
    #         for path in files_utils.collect(
    #             f"{self.opt.cp_folder}/{folder_name}/occ/", ".obj"
    #         )
    #     ]
    #     ids = torch.arange(nums_sample)
    #     if len(names) == 0:
    #         return ids + self.opt.dataset_size
    #     return ids + max(max(names) + 1, self.opt.dataset_size)

    # @torch_no_grad
    # def random_plot(self, folder_name: str, nums_sample, res=200, verbose=False):
    #     zh_base, gmms = self.model.random_samples(nums_sample)
    #     zh, attn_b = self.model.merge_zh(zh_base, gmms)
    #     numbers = self.get_new_ids(folder_name, nums_sample)
    #     self.plot_occ(zh, zh_base, gmms, numbers, folder_name, verbose=verbose, res=res)

    # @torch_no_grad
    # def plot(self, folder_name: str, nums_sample: int, verbose=False, res: int = 200):
    #     if self.model.opt.dataset_size < nums_sample:
    #         fixed_items = torch.arange(self.model.opt.dataset_size)
    #     else:
    #         fixed_items = torch.randint(
    #             low=0, high=self.opt.dataset_size, size=(nums_sample,)
    #         )
    #     zh_base, _, gmms = self.model.get_embeddings(fixed_items.to(self.device))
    #     zh, attn_b = self.model.merge_zh(zh_base, gmms)
    #     self.plot_occ(
    #         zh, zh_base, gmms, fixed_items, folder_name, verbose=verbose, res=res
    #     )
    def make_z_gmm_from_gmm(self, gmms):
        b, gp, g, _ = gmms[0].shape
        mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmms]
        p = p.reshape(*p.shape[:2], -1)
        z_gmm = torch.cat((mu, eigen, p), dim=2).detach()

        return z_gmm

    def split_z_gmm_to_gmm(self, z_gmm):

        mu = z_gmm[None, None, ..., :3]
        eigen = z_gmm[None, None, ..., 3:6]
        p = z_gmm[None, None, ..., 6:].view(1, 1, z_gmm.shape[-2], 3, 3)

        phi = torch.ones_like(z_gmm[None, None, ..., :1]) / z_gmm.shape[-2]

        return mu, p, phi, eigen

    def get_mesh_from_mid(self, gmm, included, selected, res: int, load_mid=None):
        if self.mid is None:
            return None
        gmm = [elem.to(self.device) for elem in gmm]
        z_gmm = self.make_z_gmm_from_gmm(gmm)
        included = included.to(device=self.device)
        mid_ = self.mid[included[:, 0], included[:, 1]].unsqueeze(0)
        if load_mid is not None:
            load_mid(mid_)
        mask = mid_.norm(2, -1) < 1
        mesh = self.model.decode(z_gmm, mid_, mask, res)[0]
        return mesh, mid_

    def do_adjustment(self, cond, gmm, included, selected, res: int):
        # gmm = [elem.to(self.device) for elem in gmm]
        # z_gmm = self.make_z_gmm_from_gmm(gmm)
        # included = included.to(device=self.device)
        # mid_ = self.mid[included[:, 0], included[:, 1]].unsqueeze(0)
        # # z_gmm=self.model.noodles.encode_gm(z_gmm,torch.zeros_like(z_gmm[:,:,0]).to(bool))
        # # code=torch.zeros([1,20,z_gmm.shape[-1]+mid_.shape[-1]]).to(z_gmm.device)
        # # code[0,:z_gmm.shape[1]]=torch.concat([z_gmm,mid_],-1)
        # # mask=torch.zeros_like(code).to(bool)
        # is_gen = cond[3] == 1
        # if not cond[0]:
        #     zz_gmm = torch.zeros([1, 20, 15]).to(z_gmm.device)
        #     zz_gmm[0, : z_gmm.shape[1], :] = z_gmm
        #     mask = (
        #         torch.ones_like(zz_gmm).to(bool)
        #         if cond[2]
        #         else torch.zeros_like(zz_gmm).to(bool)
        #     )
        #     mask[:, : z_gmm.shape[1], :] = 0
        #     mask[:, : z_gmm.shape[1]][selected.squeeze(0) == 1] = 1
        #     zz_gmm = self.pose.model.mask_sample(zz_gmm, ~mask, is_gen).to(
        #         torch.float32
        #     )
        #     zz_gmm_mask = zz_gmm.norm(2, -1) < 1
        #     z_gmm = zz_gmm[~zz_gmm_mask][None, :]
        # else:
        #     #gmm = torch.zeros([1, 20, 15]).to(z_gmm.device)
        #     #gmm[0, : z_gmm.shape[1]] = z_gmm
        #     z_gmm=z_gmm[0]

        # #model.mask_sample 的输入形状分别是[1,20,15] [1,20,1024] 输出也是这个形状
        
        # #这里产生的z_gmm应该形如[x, 15] x小于20

        # if not cond[1]:

        #     code = torch.zeros([1, 20, 1024]).to(z_gmm.device)
        #     mask = (
        #         torch.ones_like(code).to(bool)
        #         if cond[2]
        #         else torch.zeros_like(code).to(bool)
        #     )
        #     mask[:, : z_gmm.shape[0], :512] = 1
        #     mask[:, : z_gmm.shape[0], 512:] = 0
        #     mask[:, : selected.shape[2]][selected.squeeze(0) == 1] = 1
        #     gmm = self.model.noodles.encode_gm(
        #         z_gmm, torch.zeros_like(z_gmm[:, 0]).to(bool)
        #     )
        #     mid = torch.zeros([1, 20, 512]).to(z_gmm.device)
        #     mid[0, :mid_.shape[0]] = mid_
        #     code[0, : gmm.shape[1]] = torch.concat(
        #         [gmm[0, :], mid[0, : gmm.shape[1]]], -1
        #     )
        #     sample = self.adjust.model.mask_sample(code, ~mask, is_gen).to(
        #         torch.float32
        #     )[..., 512:]
        #     sample_mask = sample.norm(2, -1) < 1
        #     sample = sample[~sample_mask]
        #     gmm = code[~sample_mask][..., :15]
        # else:
        #     mid = torch.zeros([1, 20, 512]).to(z_gmm.device)
        #     mid[:, :mid_.shape[0]] = mid_
        #     sample = mid[0,:,:z_gmm.shape[0]]
        #     gmm=z_gmm
        # Move gmm to the device
    # Move gmm to the device
        gmm = [elem.to(self.device) for elem in gmm]

        # Convert gmm to latent representation
        # z_gmm shape: [x, 15], where x is the number of elements
        z_gmm = self.make_z_gmm_from_gmm(gmm)[0]  # z_gmm: [x, 15]
        included = included.to(device=self.device)

        # Extract corresponding mid features
        # mid_ shape: [1, x, 512]
        mid_ = self.mid[included[:, 0], included[:, 1]].unsqueeze(0)  # mid_: [1, x, 512]

        # Determine if generation mode is on
        is_gen = cond[3] == 1

        # Process GMM based on cond[0]
        if not cond[0]:  # Adjust GMM
            # Initialize zz_gmm with zeros
            # zz_gmm shape: [1, 20, 15]
            zz_gmm = torch.zeros([1, 20, 15], device=z_gmm.device)

            # Copy z_gmm into zz_gmm
            zz_gmm[0, : z_gmm.shape[0], :] = z_gmm  # z_gmm.shape[0] = x

            # Create mask for GMM adjustment
            # mask shape: [1, 20, 15]
            if cond[2]:  # Allow filling extra parts
                mask = torch.ones_like(zz_gmm).bool()
            else:
                mask = torch.zeros_like(zz_gmm).bool()

            # Unmask existing elements
            mask[:, : z_gmm.shape[0], :] = False

            # Apply selection mask
            # selected shape: should be compatible with mask[:, :selected.shape[-1]]
            mask[:, :selected.shape[-1]][selected[0, :, :] == 1] = True

            # Adjust GMM samples
            # zz_gmm shape after mask_sample: [1, 20, 15]
            zz_gmm = self.pose.model.mask_sample(zz_gmm, ~mask, is_gen).float()

            # Remove invalid entries (norm less than 1)
            # zz_gmm_mask shape: [1, 20]
            zz_gmm_mask = zz_gmm.norm(2, -1) < 1

            # z_gmm shape: [x', 15], where x' ≤ 20
            z_gmm = zz_gmm[0, ~zz_gmm_mask[0]]  # Extract valid entries
        else:  # Generate GMM
            # Use existing z_gmm
            # z_gmm shape remains: [x, 15]
            z_gmm = z_gmm

        # Process code based on cond[1]
        if not cond[1]:  # Adjust code
            # Initialize code with zeros
            # code shape: [1, 20, 1024]
            code = torch.zeros([1, 20, 1024], device=z_gmm.device)

            # Create mask for code adjustment
            # mask shape: [1, 20, 1024]
            if cond[2]:  # Allow filling extra parts
                mask = torch.ones_like(code).bool()
            else:
                mask = torch.zeros_like(code).bool()

            # Mask the GMM part (first 512 dimensions)
            mask[:, : z_gmm.shape[0], :512] = True
            # Unmask the mid part (last 512 dimensions)
            mask[:, : z_gmm.shape[0], 512:] = False

            # Apply selection mask
            mask[:, :selected.shape[-1]][selected[0, :, :] == 1] = True

            # Encode GMM into code
            # gmm_encoded shape: [1, x', 512]
            gmm_encoded = self.model.noodles.encode_gm(
                z_gmm.unsqueeze(0), torch.zeros_like(z_gmm[None, :, 0]).bool()
            )  # z_gmm.unsqueeze(0): [1, x', 15]

            # Prepare mid features
            # mid shape: [1, 20, 512]
            mid = torch.zeros([1, 20, 512], device=z_gmm.device)
            mid[0, : mid_.shape[1], :] = mid_[0]  # mid_: [1, x', 512]

            # Concatenate gmm_encoded and mid to form code
            # code[0, : x', :] shape: [x', 1024]
            code[0, : gmm_encoded.shape[1], :] = torch.cat(
                [gmm_encoded[0], mid[0, : gmm_encoded.shape[1]]], dim=-1
            )  # code: [1, 20, 1024]

            # Adjust code samples
            # sample shape after mask_sample: [1, 20, 1024]
            sample = self.adjust.model.mask_sample(code, ~mask, is_gen).float()

            # Extract mid part from sample (last 512 dimensions)
            # sample shape: [1, 20, 512]
            sample = sample[..., 512:]

            # Remove invalid entries (norm less than 1)
            # sample_mask shape: [1, 20]
            sample_mask = sample.norm(2, -1) < 1

            # Update GMM with valid entries from code
            # gmm shape: [x'', 15], where x'' ≤ 20
            sample = sample[0, ~sample_mask[0]]
            gmm = code[0, ~sample_mask[0], :15]
        else:  # Generate code
            # Prepare mid features
            # mid shape: [1, 20, 512]
            mid = torch.zeros([1, 20, 512], device=z_gmm.device)
            mid[0, : mid_.shape[1], :] = mid_[0]

            # Extract sample from mid features
            # sample shape: [x, 512]
            sample = mid[0, : z_gmm.shape[0], :]

            # GMM remains the same
            # gmm shape: [x, 15]
            gmm = z_gmm

        # At this point, sample and gmm should have the same number of elements

        # Compute norms
        # sample_norm shape: [number of elements]
        sample_norm = sample.norm(2, -1)  # sample_norm: [num_elements]
        gmm_norm = gmm.norm(2, -1)        # gmm_norm: [num_elements]

        # Compute valid mask where both norms are greater than 1
        # valid_mask shape: [number of elements]
        valid_mask = (sample_norm > 1) & (gmm_norm > 1)  # valid_mask: [num_elements]

        # Filter sample and gmm using valid_mask
        # sample and gmm shapes after filtering: [num_valid_elements, ...]
        sample = sample[valid_mask]
        gmm = gmm[valid_mask]

        # Update mid features with valid samples
        # self.mid shape depends on implementation; assuming [batch_size, 20, 512]
        self.mid[-1, : sample.shape[0], :] = sample  # Update with valid samples

        # Update included tensor
        # included shape: [num_valid_elements, 2]
        included = torch.zeros(sample.shape[0], 2, dtype=torch.int64, device=sample.device)
        included[:, 0] = -1
        included[:, 1] = torch.arange(sample.shape[0], device=sample.device)

        # Convert latent representation back to GMM
        # gmm is now a list of elements on the device
        gmm = self.split_z_gmm_to_gmm(gmm)

        # Return updated gmm, included, selected, and res
        selected=torch.zeros_like(included[:,0])
        return gmm, included, selected, res

    def set_items(self, items):
        self.mid = items.to(self.device)

        self.mid = torch.concat([self.mid, torch.zeros_like(self.mid[0:1])], 0)

    def __init__(self, opt):
        self.opt = opt
        self.model = model_load(opt).eval()
        self.adjust = adjust_load(opt).eval()
        self.pose = pose_load(opt).eval()
        # self.translate = translate_load(opt).eval()
        self.mid = None
        self.gmms = None
        # self.meshing = mcubes_meshing.MarchingCubesMeshing(self.device, scale=1.0)
