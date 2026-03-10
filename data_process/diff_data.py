import torch
from pathlib import Path
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import math
from typing import Optional

class DiffDataset(Dataset):
    """
    读取:
      - {data_dir}/feats.pth  -> [valid_num, max_len, feat_dim]
      - {data_dir}/params.pth -> [valid_num, max_len, 15]
    或兼容:
      - diff_feats.pth / diff_params.pth

    输出:
      每个样本: [max_len, 1 + 15 + feat_dim]
      其中最前面 1 维为标志位: 有效行为  1.0, 填充行为 -1.0 (依据 feat 是否全 0)
    """
    def __init__(self, data_dir: str, transform=None, dtype=torch.float32):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.dtype = dtype

        # 兼容两种命名
        feats_path = self.data_dir / 'feats.pth'
        params_path = self.data_dir / 'params.pth'
        if not feats_path.exists() or not params_path.exists():
            alt_feats = self.data_dir / 'diff_feats.pth'
            alt_params = self.data_dir / 'diff_params.pth'
            if alt_feats.exists() and alt_params.exists():
                feats_path, params_path = alt_feats, alt_params

        if not feats_path.exists() or not params_path.exists():
            raise FileNotFoundError(
                f"Data directory {data_dir} 缺少 feats.pth/params.pth "
                f"（或 diff_feats.pth/diff_params.pth）。"
            )

        # 加载为 CPU float 张量
        feats = torch.load(feats_path, map_location='cpu')
        params = torch.load(params_path, map_location='cpu')

        if not isinstance(feats, torch.Tensor) or not isinstance(params, torch.Tensor):
            raise TypeError("feats.pth 和 params.pth 必须保存为 PyTorch 张量。")

        # 形状检查
        if feats.ndim != 3 or params.ndim != 3:
            raise ValueError(
                f"feats/params 形状不正确："
                f"feats.ndim={feats.ndim}, params.ndim={params.ndim}，"
                f"期望都是 [valid_num, max_len, C] 的三维张量。"
            )
        if feats.shape[:2] != params.shape[:2]:
            raise ValueError(
                f"feats 与 params 的前两维不一致："
                f"feats[:2]={feats.shape[:2]}, params[:2]={params.shape[:2]}"
            )
        if params.shape[-1] != 15:
            raise ValueError(f"params 的最后一维必须是 15，当前为 {params.shape[-1]}。")

        # 统一 dtype
        self.feats = feats.to(self.dtype).contiguous()
        self.params = params.to(self.dtype).contiguous()

        self.valid_num, self.max_len, self.feat_dim = self.feats.shape
        # 预先计算标志位: 每行 feats 是否全 0
        # mask: True 表示填充行（全 0）
        with torch.no_grad():
            zero_mask = (self.feats.abs().sum(dim=-1) == 0)    # [valid_num, max_len]
            # 有效  -> 1.0；填充 -> -1.0
            flags = torch.where(zero_mask, torch.tensor(-0.01, dtype=self.dtype),
                                           torch.tensor( 0.01, dtype=self.dtype))
            self.flags = flags.unsqueeze(-1)                   # [valid_num, max_len, 1]

        # 预先拼接，减少 __getitem__ 成本
        self._packed = torch.cat([self.flags, self.params, self.feats], dim=-1).contiguous()
        # 最终每条样本的形状: [max_len, 1 + 15 + feat_dim]

    def __len__(self):
        return self.valid_num

    def __getitem__(self, idx):
        x = self._packed[idx].clone()  # [max_len, 1+15+feat_dim]
        if self.transform is not None:
            x = self.transform(x)
        return x

    @property
    def output_dim(self):
        return 1 + 15 + self.feat_dim

    @property
    def max_length(self):
        return self.max_len
    

class DiffDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers:bool= True,
        drop_last: bool = False,
        dtype: torch.dtype = torch.float32,
        transform=None,
    ):
        """
        data_dir: 目录中需包含 feats.pth 与 params.pth（或 diff_feats.pth 与 diff_params.pth）
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dtype = dtype
        self.transform = transform

        # 如果未显式指定，默认当 num_workers>0 时启用
        self.persistent_workers = (
            bool(num_workers > 0) if persistent_workers is None else persistent_workers
        )

        self.dataset = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        # 不做磁盘写操作；仅用于多进程下载/准备的场景，这里留空
        pass

    def setup(self, stage: Optional[str] = None):
        # 只构建一次底层数据集
        if self.dataset is None:
            self.dataset = DiffDataset(self.data_dir, transform=self.transform, dtype=self.dtype)

        if stage == "fit" or stage is None:
            n_total = len(self.dataset)
            n_val = int(math.floor(n_total * self.val_ratio))
            n_train = n_total - n_val
            # 至少保证训练集有样本
            if n_train <= 0:
                raise ValueError(
                    f"切分后训练集为空：总数={n_total}, val_ratio={self.val_ratio}, test_ratio={self.test_ratio}"
                )
            gen = torch.Generator().manual_seed(self.seed)
            self.train_set, self.val_set= random_split(
                self.dataset, lengths=[n_train, n_val], generator=gen
            )
            self.test_set = self.val_set

        if stage == "validate" and self.val_set is None:
            # 仅验证阶段调用时，保证 val_set 存在
            self.setup(stage="fit")
        if stage == "test" and self.test_set is None:
            # 仅测试阶段调用时，保证 test_set 存在
            self.setup(stage="fit")
        if stage == "predict":
            # 预测阶段直接使用完整数据集
            pass

    # ===== DataLoaders =====
    def train_dataloader(self):
        return DataLoader(
            self.dataset,#self.train_set
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def predict_dataloader(self):
        # 预测时通常希望顺序、不打乱，覆盖全量
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    # ===== 便捷属性（可选）=====
    @property
    def max_len(self):
        return self.dataset.max_length if self.dataset is not None else None

    @property
    def output_dim(self):
        return self.dataset.output_dim if self.dataset is not None else None