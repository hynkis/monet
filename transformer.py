"""
Reference
    - ViViT: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vivit.py

"""
import torch
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, embed_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, embed_dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == embed_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias = False)

        if project_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, embed_dim),
                nn.Dropout(dropout)
            )
        elif heads == 1:
            self.to_out = nn.Linear(inner_dim, embed_dim)
        else:
            self.to_out = nn.Identity()

        self.att_map = None

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (batch, head, N, embed_dim)

        attn = self.attend(dots) # (batch, head, N, N)
        attn = self.dropout(attn)

        # save attention map
        with torch.no_grad():
            self.att_map = attn.clone() # (batch, head, N, N)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') # (batch, N, inner_dim(=head*dim_head))
        return self.to_out(out) # (batch, N, embed_dim)

class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.MHA_block = Attention(embed_dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.MLP_block = FeedForward(embed_dim, mlp_dim, dropout = dropout)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embed_dim, self.MHA_block),
                PreNorm(embed_dim, self.MLP_block)
            ]))

        # for logging
        self.att_map = {}

    def forward(self, x):
        i = 0 # for checking depth of transformer layers
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

            # save attention map
            with torch.no_grad():
                self.att_map[i] = self.MHA_block.att_map # (batch, head, N, N)
            i += 1

        return x


class ViViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=224,
        image_patch_size=16,
        frames=4,
        frame_patch_size=2,
        output_dim=64,
        embed_dim=1024,
        spatial_depth=6,
        temporal_depth=6,
        heads=8,
        mlp_dim=2048,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.3,
        emb_dropout = 0.1,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width) # image_height -> patch_height
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (f pf) c (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.Linear(patch_dim, embed_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_feature_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if not self.global_average_pool else None
        self.temporal_feature_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if not self.global_average_pool else None

        self.spatial_transformer = Transformer(embed_dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(embed_dim, temporal_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

        self.spatial_att_map = {}
        self.temporal_att_map = {}

    def forward(self, seq_img):
        x = self.to_patch_embedding(seq_img) # (batch, T, C, H, W) -> (batch, t, (h*w), embed_dim)
        b, f, n, _ = x.shape
        # # - additive position embedding
        # x = x + self.pos_embedding
        # # - concat position embedding
        # pos_embedding = repeat(self.pos_embedding, '() t n d -> batch t n d', batch = b)
        # x = torch.cat([x, pos_embedding], dim=3) # (batch, t, (h*w), embed_dim)

        if exists(self.spatial_feature_token):
            spatial_cls_tokens = repeat(self.spatial_feature_token, '1 1 d -> b f 1 d', b = b, f = f)
            x = torch.cat((spatial_cls_tokens, x), dim = 2) # (batch, t, (h*w)+1, embed_dim)

        x = self.dropout(x)
        x = rearrange(x, 'b f n d -> (b f) n d') # (batch*t, (h*w)+1, embed_dim)

        # attend across space
        x = self.spatial_transformer(x) # (batch*t, (h*w)+1, embed_dim)
        with torch.no_grad():
            for key in self.spatial_transformer.att_map.keys():
                self.spatial_att_map[key] = self.spatial_transformer.att_map[key] # already cloned

        # excise out the spatial cls tokens or average pool for temporal attention
        x = rearrange(x, '(b f) n d -> b f n d', b = b) # (batch, t, (h*w)+1, embed_dim)
        if self.pool == 'cls':
            x = x[:, :, 0]
        elif self.pool == 'mean':
            x = reduce(x, 'b f n d -> b f d', 'mean')
        else:
            raise NotImplementedError
        # x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean') # (batch, t, embed_dim)

        # append temporal CLS tokens
        if exists(self.temporal_feature_token):
            temporal_cls_tokens = repeat(self.temporal_feature_token, '1 1 d-> b 1 d', b = b)
            x = torch.cat((temporal_cls_tokens, x), dim = 1) # (batch, t+1, embed_dim)

        # attend across time
        x = self.temporal_transformer(x) # (batch, t+1, embed_dim)
        with torch.no_grad():
            for key in self.spatial_transformer.att_map.keys():
                self.temporal_att_map[key] = self.temporal_transformer.att_map[key] # already cloned

        # excise out temporal cls token or average pool
        if self.pool == 'cls':
            x = x[:, 0]
        elif self.pool == 'mean':
            x = reduce(x, 'b f d -> b d', 'mean')
        else:
            x = x[:, 0]
        # x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean') # (batch, embed_dim)
        x = self.to_latent(x) # (batch, embed_dim)
        return self.mlp_head(x) # (batch, output_dim)

    def get_spatial_att_map(self):
        return self.spatial_att_map

    def get_temporal_att_map(self):
        return self.temporal_att_map

class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=224,
        image_patch_size=16,
        output_dim=64,
        embed_dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.3,
        emb_dropout = 0.1,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, embed_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )
        self.spatial_att_map = {}

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        with torch.no_grad():
            for key in self.transformer.att_map.keys():
                self.spatial_att_map[key] = self.transformer.att_map[key] # already cloned

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def get_spatial_att_map(self):
        return self.spatial_att_map

class CoT(nn.Module):
    def __init__(
        self,
        *,
        input_size=128,
        input_embed_size=16,
        input_patch_size=4,
        output_dim=16,
        embed_dim=1024,
        contextual_depth=6,
        heads=16,
        mlp_dim=2048,
        pool = 'cls',
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()

        assert input_size % input_patch_size == 0, 'Input data must be divisible by input patch size'
        num_input_patches = (input_size // input_patch_size)

        assert pool in {'cls', 'mean', 'max', 'none'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.input_embedding = nn.Linear(1, input_embed_size) # (batch, input_size, 1) -> (batch, input_size, input_embed_dim)

        patch_dim = input_embed_size * input_patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (f pf) c -> b f (pf c)', pf = input_patch_size),
            nn.Linear(patch_dim, embed_dim),
        ) # (batch, input_size, input_embed_dim) -> (batch, N_ct, embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_input_patches, embed_dim)) # (batch, N_ct, embed_dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.contextual_feature_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if not self.global_average_pool else None

        self.contextual_transformer = Transformer(embed_dim, contextual_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        if pool == "none":
            # mlp_head for feature without pooling
            self.mlp_head = nn.Sequential(
                nn.LayerNorm((num_input_patches+1, embed_dim)), # +1 for cls_token
                nn.Linear(embed_dim, output_dim, bias=False) # for query output
            )
            # # mlp_head for feature without pooling
            # self.mlp_head = nn.Sequential(
            #     nn.LayerNorm(num_input_patches, embed_dim),
            #     nn.Linear(embed_dim, output_dim)
            # )
        else:
            # mlp_head for feature with pooling
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, output_dim)
            )

        self.contextual_att_map = {}

    def forward(self, context):
        # Input embedding
        x = context.unsqueeze(dim=-1) # (batch, input_size) -> (batch, input_size, 1)
        x = self.input_embedding(x) # (batch, input_size, 1) -> (batch, input_size, input_embed_dim)

        # Patch & Positional embedding
        x = self.to_patch_embedding(x) # (batch, input_size, input_embed_dim) -> (batch, N_ct, embed_dim)
        b, f, _ = x.shape

        # - additive position embedding
        x_cont = x + self.pos_embedding # (batch, N_ct, embed_dim)
        # # - concat position embedding
        # pos_embedding = repeat(self.pos_embedding, '() n d -> batch n d', batch = b)
        # x = torch.cat([x, pos_embedding], dim=2) # (batch, N_ct, embed_dim)

        # append contextual CLS tokens
        if exists(self.contextual_feature_token):
            contextual_cls_tokens = repeat(self.contextual_feature_token, '1 1 d-> b 1 d', b = b)
            x_cont = torch.cat((contextual_cls_tokens, x_cont), dim = 1) # (batch, N_ct+1, embed_dim)
        x_cont = self.dropout(x_cont)

        # contextual transformer (CoT)
        x_cont = self.contextual_transformer(x_cont) # (batch, N_ct, embed_dim)
        with torch.no_grad():
            for key in self.contextual_transformer.att_map.keys():
                self.contextual_att_map[key] = self.contextual_transformer.att_map[key] # already cloned

        # excise out contextual cls token or average pool
        if self.pool == 'max':
            x_cont = reduce(x_cont, 'b f d -> b d', 'max') # (batch, N_ct, embed_dim) or (batch, embed_dim)
        elif self.pool == 'cls':
            x_cont = x_cont[:, 0]
        elif self.pool == 'mean':
            x_cont = reduce(x_cont, 'b f d -> b d', 'mean') # (batch, N_ct, embed_dim) or (batch, embed_dim)
        else:
            pass # no pooling
        
        x_cont = self.to_latent(x_cont) # (batch, N_ct, embed_dim) or (batch, embed_dim)
        return self.mlp_head(x_cont) # (batch, N_ct, embed_dim); (batch, N_ct, embed_dim): latent decision query
    
    def get_contextual_att_map(self):
        return self.contextual_att_map




