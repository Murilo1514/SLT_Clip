import torch
import torch.nn as nn
import re
import math
import torch.nn.functional as F


# Credit by https://github1s.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py
class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def build_vision_projector(mm_projector_type='linear', mm_hidden_size=512, hidden_size=768, mlp_depth=1):
    if mm_projector_type == 'linear':
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mm_projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match.group(1).isdigit() else mlp_depth
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if mm_projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {mm_projector_type}')



class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

    def forward(self, x):
        B, T, D = x.shape

        position = torch.arange(
            T, device=x.device
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0, D, 2,
                device=x.device,
                dtype=torch.float32
            ) * (-math.log(10000.0) / D)
        )

        pe = torch.zeros(
            T, D,
            device=x.device,
            dtype=torch.float32
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe.unsqueeze(0).to(dtype=x.dtype)


class LandmarkTransformerProjector(nn.Module):
    def __init__(
        self,
        mm_hidden_size=26160,
        hidden_size=768,
        embed_dim=1024,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        # Cada landmark possui 3 coordenadas (x, y, z)
        self.num_landmarks = mm_hidden_size // 3

        self.input_proj = nn.Linear(mm_hidden_size, embed_dim)

        self.pos_embedding = SinusoidalPositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.input_norm = nn.LayerNorm(mm_hidden_size) 
        self.norm = nn.LayerNorm(embed_dim)

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: [B, 26160]
        # B = x.shape[0]

        # [B, 8720, 3]
        # x = x.view(B, self.num_landmarks, 3)

        # B, T, _ = x.shape
        # x = x.view(B, T, self.num_landmarks, 3)

        # [B, 8720, 256]
        x = self.input_norm(x)
        x = self.input_proj(x)

        x = self.pos_embedding(x)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  

        # Transformer
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)


        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        # [B, 256]
        x = self.norm(x) #.mean(dim=1)

        # [B, 768]
        x = self.projector(x)

        return x


def build_landmarks_projector(
    mm_hidden_size=26160,
    hidden_size = 768,
    linear = False
):
    if not linear:
        return LandmarkTransformerProjector(
            mm_hidden_size=mm_hidden_size,
            hidden_size=hidden_size
        )

    seq = nn.Sequential(
        nn.LayerNorm(mm_hidden_size),
        nn.Linear(mm_hidden_size, hidden_size)
    )
    
    return seq


# https://github1s.com/facebookresearch/jepa/blob/main/src/models/utils/modules.py
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim*2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = (xattn @ v)

        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)
    
        return q


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q
