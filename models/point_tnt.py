import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import argparse


# from helpers import farthest_point_sample, index_points, get_graph_feature

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=3,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Baseline(nn.Module):
    '''
    Baseline Transformer model that applies self-attention globally on all points
    '''

    def __init__(
            self,
            num_classes,
            channels=3,
    ):
        super(Baseline, self).__init__()

        parser = argparse.ArgumentParser(description='Point Cloud Recognition')
        parser.add_argument('--global_attention', type=str2bool, default=True,
                            help='Toggle the use of global attention (True/False)')
        parser.add_argument('--patch_dim', type=int, default=192,  #
                            help='Anchor embedding dimension')
        parser.add_argument('--depth', type=int, default=1,  # 4
                            help='Number of sequential transformers')
        parser.add_argument('--heads', type=int, default=3,  # 3
                            help='Number of attention heads')
        parser.add_argument('--dim_head', type=int, default=64,
                            help='Attention head dimension')
        parser.add_argument('--attn_dropout', type=float, default=0.,
                            help='Attention dropout')
        parser.add_argument('--ff_dropout', type=float, default=0.,
                            help='Feed-forward dropout')
        parser.add_argument('--emb_dims', type=int, default=1024,
                            help='Dimension of embeddings')

        # args = parser.parse_args()
        global_attention = True
        patch_dim = 192
        depth = 1
        heads = 3
        dim_head = 64
        attn_dropout = 0.
        ff_dropout = 0.
        emb_dims = 1024

        self.global_attention = global_attention

        self.to_anchor = nn.Sequential(
            Rearrange('b c n -> b n c'),
            nn.Linear(channels, patch_dim)
        )

        layers = nn.ModuleList([])
        for _ in range(depth):
            layers.append(nn.ModuleList([
                PreNorm(patch_dim, Attention(dim=patch_dim, heads=heads, dim_head=dim_head,
                                                  dropout=attn_dropout)) if self.global_attention else nn.Identity(),
                PreNorm(patch_dim, FeedForward(
                    dim=patch_dim, dropout=ff_dropout)),
            ]))

        self.layers = layers

        self.final_conv = nn.Sequential(
            nn.LayerNorm(patch_dim * depth),
            nn.Linear(patch_dim * depth, emb_dims),
            nn.GELU(),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dims * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        batch_size, _, _ = x.shape

        # all points are anchores
        points = self.to_anchor(x)
        ylist = []

        for patch_attn, patch_ff in self.layers:
            # apply global transformer
            points = patch_attn(points) + points
            points = patch_ff(points) + points

            # store intermediate features
            ylist.append(points)

        # apply final layer to all intermediate patch features
        y = torch.cat(ylist, dim=-1)
        y = self.final_conv(y)

        # pool over all points
        y1 = y.max(dim=1, keepdim=False)[0]
        y2 = y.mean(dim=1, keepdim=False)
        y = torch.cat((y1, y2), dim=1)

        # apply classifier head
        return self.mlp_head(y)
