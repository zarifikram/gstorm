import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from sub_models.attention_blocks import get_vector_mask
from sub_models.attention_blocks import (
    PositionalEncoding1D,
    AttentionBlock,
    AttentionBlockKVCache,
)
from sub_models.dino_transformer_utils import CrossAttention

class StateMixer(nn.Module):
    def __init__(self, z_dim, action_dim, feat_dim, type, num_heads=None):
        super().__init__() 
        self.type_mixer = type
        if self.type_mixer == 'concat':
            input_dim = z_dim + action_dim
            self.mha = lambda z, a: z
        
        elif self.type_mixer == 'concat+attn':
            input_dim = z_dim + action_dim
            self.mha = CrossAttention(num_heads, input_dim, action_dim)
            
        elif self.type_mixer == 'z+attn':
            input_dim = z_dim 
            self.mha = CrossAttention(num_heads, z_dim, action_dim)
        else:
           raise f'Mixer of type {self.type_mixer} not defined, modify config file!'

        self.mixer = nn.Sequential(
                nn.Linear(input_dim, feat_dim, bias=False),
                nn.LayerNorm(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim, bias=False),
                nn.LayerNorm(feat_dim)
                )
        
    def forward(self, z, a):
        input =  torch.cat([z, a], dim=-1) if 'concat' in self.type_mixer else z
        h = self.mha(input, a)
        return self.mixer(h)

class StochasticTransformer(nn.Module):
    def __init__(
        self,
        stoch_dim,
        action_dim,
        feat_dim,
        num_layers,
        num_heads,
        max_length,
        dropout,
        state_mix_type:str,
    ):
        super().__init__()
        self.action_dim = action_dim

        # mix image_embedding and action
        # self.stem = nn.Sequential(
        #     nn.Linear(stoch_dim + action_dim, feat_dim, bias=False),
        #     nn.LayerNorm(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, feat_dim, bias=False),
        #     nn.LayerNorm(feat_dim),
        # )
        num_head_mixer = 2 if 'attn' in state_mix_type else None
        self.stem = StateMixer(
            stoch_dim,
            action_dim,
            feat_dim,
            state_mix_type,
            num_head_mixer
        )
        self.position_encoding = PositionalEncoding1D(
            max_length=max_length, embed_dim=feat_dim
        )
        self.layer_stack = nn.ModuleList(
            [
                AttentionBlock(
                    feat_dim=feat_dim,
                    hidden_dim=feat_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)

        self.head = nn.Linear(feat_dim, stoch_dim)

    def forward(self, samples, action, mask):
        action = F.one_hot(action.long(), self.action_dim).float()
        # feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.stem(samples, action)
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)

        feat = self.head(feats)
        return feat


class StochasticTransformerKVCache(nn.Module):
    def __init__(
        self,
        stoch_dim,
        action_dim,
        feat_dim,
        num_layers,
        num_heads,
        max_length,
        dropout,
        device: torch.device,
        state_mix_type: str,
        continuous_action=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.device = device
        self.continuous_action = continuous_action

        # mix image_embedding and action
        # self.stem = nn.Sequential(
        #     nn.Linear(stoch_dim + action_dim, feat_dim, bias=False),
        #     nn.LayerNorm(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, feat_dim, bias=False),
        #     nn.LayerNorm(feat_dim),
        # )
        num_head_mixer = 2 if 'attn' in state_mix_type else None
        self.stem = StateMixer(
            stoch_dim,
            action_dim,
            feat_dim,
            state_mix_type,
            num_head_mixer
        )
        self.position_encoding = PositionalEncoding1D(
            max_length=max_length, embed_dim=feat_dim
        )
        self.layer_stack = nn.ModuleList(
            [
                AttentionBlockKVCache(
                    feat_dim=feat_dim,
                    hidden_dim=feat_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)

    def forward(self, samples, action, mask):
        """
        Normal forward pass
        """
        action = F.one_hot(action.long(), self.action_dim).float() if not self.continuous_action else action
        # feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.stem(samples, action)
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)

        return feats

    def reset_kv_cache_list(self, batch_size, dtype):
        """
        Reset self.kv_cache_list
        """
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(
                torch.zeros(
                    size=(batch_size, 0, self.feat_dim), dtype=dtype, device=self.device
                )
            )

    def forward_with_kv_cache(self, samples, action):
        """
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        """
        # assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1] + samples.size(1), samples.device)
        action = F.one_hot(action.long(), self.action_dim).float() if not self.continuous_action else action
        # feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.stem(samples, action)
        feats = self.position_encoding.forward_with_position(
            feats, position=self.kv_cache_list[0].shape[1]
        )
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(
                feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask
            )

        return feats
