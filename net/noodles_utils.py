import torch
from torch import nn
import torch.nn.functional as F


class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(PreNormTransformerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CustomMultiheadAttention(d_model,d_model,d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, src, src_key_padding_mask=None,**cfg):
        src2 = self.norm1(src)
        src2 = self.attention(
            src2, src2, key_padding_mask=src_key_padding_mask
        )
        src = src + src2
        src2 = self.norm2(src)
        src2 = self.feedforward(src2)
        src = src + src2
        return src
    
class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x,src_key_padding_mask=src_key_padding_mask)
        return x
    
class CustomTransformerDecoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
    def forward(self, x,y, memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x,y,memory_key_padding_mask=memory_key_padding_mask)
        return x

class MixNet(nn.Module):
    def __init__(self, d_model, nhead, nlayer, **cfg):
        super().__init__()
        self.layer = PreNormTransformerLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2
        )
        self.transformer = CustomTransformerEncoder(self.layer, nlayer)

    def forward(self, x, mask):
        x = self.transformer(x, src_key_padding_mask=mask)
        return x


class CustomMultiheadAttention(nn.Module):
    def __init__(self, d_tgt, d_memory, d_model, nhead, attn_list_add=None):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.q_proj = nn.Linear(d_tgt, d_model)
        self.k_proj = nn.Linear(d_memory, d_model)
        self.v_proj = nn.Linear(d_memory, d_model)
        self.out_proj = nn.Linear(d_model, d_tgt)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_head]))
        self.attn_list_add = attn_list_add

    def forward(self, tgt, memory, key_padding_mask=None):
        batch_size, tgt_len, _ = tgt.size()
        _, memory_len, _ = memory.size()
        query = self.q_proj(tgt)
        key = self.k_proj(memory)
        value = self.v_proj(memory)
        query = query.view(batch_size, tgt_len, self.nhead, self.d_head).transpose(1, 2)
        # query=F.normalize(query,dim=-1)
        key = key.view(batch_size, memory_len, self.nhead, self.d_head).transpose(1, 2)
        # key=F.normalize(key,dim=-1)
        value = value.view(batch_size, memory_len, self.nhead, self.d_head).transpose(
            1, 2
        )
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale.to(
            query.device
        )
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        if self.attn_list_add is not None:
            self.attn_list_add(attn)
        context = torch.matmul(attn, value)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, tgt_len, self.d_head * self.nhead)
        )
        output = self.out_proj(context)
        return output


class OccupancyLayer(nn.Module):
    def __init__(self, d_model, d_point, nhead, attn_list, **cfg):
        super().__init__()
        self.cross_attn = CustomMultiheadAttention(
            d_point, d_model, d_model, nhead, attn_list
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_point, 2 * d_point), nn.ReLU(), nn.Linear(2 * d_point, d_point)
        )
        self.norm1 = nn.LayerNorm(d_point)
        self.norm2 = nn.LayerNorm(d_point)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory, memory_key_padding_mask, **cfg):
        tgt2 = self.cross_attn(
            self.norm1(tgt), memory, key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)

        tgt2 = self.feed_forward(self.norm2(tgt))
        tgt = tgt + self.dropout(tgt2)
        return tgt


class OccupancyMlP(nn.Module):
    def __init__(self, d_model, d_point, head_occ_size, **cfg):
        super(OccupancyMlP, self).__init__()
        dim_in = d_point + d_point
        self.head_occ_size = head_occ_size
        dims = [dim_in] * self.head_occ_size + [dim_in // 2] + [1 + 3 * 3]
        self.latent_in = self.head_occ_size // 2
        # dims[self.latent_in] += dim_in

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(True)
        layers = []
        for i in range(len(dims) - 1):
            if i == self.latent_in:
                layers.append(nn.Linear(dims[i] + dim_in, dims[i + 1]))
            else:
                layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, z):
        x_ = x = torch.cat((x, z), dim=-1)
        for i, layer in enumerate(self.layers):
            if i == self.latent_in:
                x = torch.cat([x, x_], dim=-1)
            x = layer(x)
            if i < len(self.layers) - 2:
                x = self.dropout(self.relu(x))
        return x


class OccupancyNet(nn.Module):
    def __init__(self, d_model, d_point, nhead, num_layers, occ_mlp_size, **cfg):
        super().__init__()
        self.attns = []

        def attn_list_add(attn):
            self.attns.append(attn)

        self.layer = OccupancyLayer(
            d_model=d_model, d_point=d_point, nhead=nhead, attn_list=None
        )
        self.transformer = CustomTransformerDecoder(self.layer, num_layers)
        self.mlp = OccupancyMlP(d_model, d_point, occ_mlp_size)

    def forward(self, x, src, mask):
        self.attns.clear()
        z = self.transformer(x, src, mask)
        x = self.mlp(x, z)
        return x, z, self.attns
