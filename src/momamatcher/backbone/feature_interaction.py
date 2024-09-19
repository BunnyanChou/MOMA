import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from yacs.config import CfgNode as CN

from src.momamatcher.ada_module.transformer import (
    DecoderLayer,
    EncoderLayer,
    TransformerDecoder,
    LocalFeatureTransformer_Flow
)


def make_head_layer(cnv_dim, curr_dim, out_dim, head_name=None):

    fc = nn.Sequential(
        nn.Conv2d(cnv_dim, curr_dim, kernel_size=3, padding=1, bias=True),
        # nn.BatchNorm2d(curr_dim, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
        nn.Conv2d(curr_dim, out_dim, kernel_size=3, stride=1, padding=1),
    )

    for l in fc.modules():
        if isinstance(l, nn.Conv2d):
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    return fc

class FeatureAttention(nn.Module):
    config = CN()
    config.nhead = 8
    config.layer_names = ["self", "cross"] * 4
    config.attention = "linear"

    def __init__(self, layer_num=2, d_model=256):
        super(FeatureAttention, self).__init__()
        self.config.layer_names = ["self", "cross"] * layer_num
        self.d_model = d_model
        encoder_layer = EncoderLayer(
            d_model=self.d_model,
            nhead=self.config.nhead,
            attention=self.config.attention,
        )
        self.layer_names = self.config.layer_names
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x0, x1, x0_mask=None, x1_mask=None, flag=False):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]      ->   # [N, L, C]
            source (torch.Tensor): [N, C, H1, W1] ->   # [N, S, C]
            x_mask (torch.Tensor): [N, H0, W0]       -> # [N, L] (optional)
            source_mask (torch.Tensor): [N, H1, W1]  -> # [N, S] (optional)
        """
        assert self.d_model == x0.size(
            2
        ), "the feature number of src and transformer must be equal"

        if x0_mask != None and x1_mask != None:
            x0_mask, x1_mask = x0_mask.flatten(-2), x1_mask.flatten(-2)

        if flag is False:
            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == "self":
                    src0, src1 = x0, x1
                    src0_mask, src1_mask = x0_mask, x1_mask
                elif name == "cross":
                    src0, src1 = x1, x0
                    src0_mask, src1_mask = x1_mask, x0_mask
                else:
                    raise KeyError
                x0 = layer(x0, src0, x0_mask, src0_mask)
                x1 = layer(x1, src1, x1_mask, src1_mask)
        elif flag == 1:  # origin
            for layer, name in zip(self.layers, self.layer_names):
                if name == "self":
                    x0 = layer(x0, x0, x0_mask, x0_mask)
                    x1 = layer(x1, x1, x1_mask, x1_mask)
                elif name == "cross":
                    x0 = layer(x0, x1, x0_mask, x1_mask)
                    x1 = layer(x1, x0, x1_mask, x0_mask)
                else:
                    raise KeyError
        elif flag == 2:
            for layer, name in zip(self.layers, self.layer_names):
                if name == "self":
                    x0 = layer(x0, x0, x0_mask, x0_mask)
                    x1 = layer(x1, x1, x1_mask, x1_mask)
                elif name == "cross":
                    x1 = layer(x1, x0, x1_mask, x0_mask)
                    x0 = layer(x0, x1, x0_mask, x1_mask)
                else:
                    raise KeyError

        return x0, x1

class FICAS(nn.Module):
    config = CN()
    config.nhead = 8
    config.attention = "linear"

    def __init__(self, config, d_model=256):
        super(FICAS, self).__init__()
        self.d_model = config['d_model']
        self.num_query = 1
        self.coarsest_level = config['coarsest_level']

        encoder_layer = EncoderLayer(
            d_model=self.d_model,
            nhead=self.config.nhead,
            attention=self.config.attention,
        )

        self.loftr_coarse = LocalFeatureTransformer_Flow(config)

        self.feature_embed = nn.Embedding(self.num_query, self.d_model)
        decoder_layer = DecoderLayer(
            d_model,
            8,
            dropout=0.1,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=2)
        self.layer_names2 = ["cross"]
        self.layers2 = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names2))]
        )

        self.layer_names3 = [
            "self",
            "cross",
        ]
        self.layers3 = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names3))]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def transformer(self, x0, x1, x0_mask, x1_mask, layer_name, layer):
        if layer_name == "self":
            src0, src1 = x0, x1
            src0_mask, src1_mask = x0_mask, x1_mask
        elif layer_name == "cross":
            src0, src1 = x1, x0
            src0_mask, src1_mask = x1_mask, x0_mask
        else:
            raise KeyError
        if (
            x0.shape == x1.shape
            and src0.shape == src1.shape
            and x0_mask is not None
            and x1_mask is not None
            and src0_mask is not None
            and src1_mask is not None
            and not self.training
            and 0
        ):
            temp_x = layer(
                torch.cat([x0, x1], dim=0),
                torch.cat([src0, src1], dim=0),
                torch.cat([x0_mask, x1_mask], dim=0),
                torch.cat([src0_mask, src1_mask], dim=0),
            )
            x0, x1 = temp_x.split(x0.shape[0])
        else:
            x0 = layer(x0, src0, x0_mask, src0_mask)
            x1 = layer(x1, src1, x1_mask, src1_mask)
        return x0, x1

    def feature_interaction(self, data, x0, x1, pos_encoding0, pos_encoding1, x0_mask=None, x1_mask=None):
        """
        x (torch.Tensor): [N, C, H, W] -> [N, L, C] 
        source (torch.Tensor): [N, C, H, W] -> [N, S, C] 
        x_mask (torch.Tensor): [N, H0, W0]       -> # [N, L] (optional) 
        source_mask (torch.Tensor): [N, H1, W1]  -> # [N, S] (optional)
        """
        bs = x0.size(0)  
        if x0_mask != None and x1_mask != None:
            x0_mask, x1_mask = x0_mask.flatten(-2), x1_mask.flatten(-2)

        # stage 1
        
        hw0_c = data["hw0_d8"]
        hw1_c = data["hw1_d8"]

        ds0=[int(hw0_c[0]/self.coarsest_level[0]), int(hw0_c[1]/self.coarsest_level[1])]
        ds1=[int(hw1_c[0]/self.coarsest_level[0]), int(hw1_c[1]/self.coarsest_level[1])]
            
        # x0 "n c h w -> n (h w) c"
        x0, x1, flow_list = self.loftr_coarse(
            x0, x1, pos_encoding0, pos_encoding1, x0_mask, x1_mask, ds0, ds1) # [L,N,H,W,4]*1(2)
        
        assert self.d_model == x0.size(2), "the feature number of src and transformer must be equal"
        # stage 2
        feature_embed0 = self.feature_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [bs, num_q, c]
        feature_embed1 = self.feature_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt0 = torch.zeros_like(feature_embed0)
        tgt1 = torch.zeros_like(feature_embed1)

        hs0 = self.decoder(tgt0, x0, tgt_mask=None, memory_mask=x0_mask, tgt_pos=feature_embed0)
        hs1 = self.decoder(tgt1, x1, tgt_mask=None, memory_mask=x1_mask, tgt_pos=feature_embed1)

        for i, (layer, name) in enumerate(zip(self.layers2, self.layer_names2)):
            if not self.training and x0.shape == x1.shape and x0_mask is not None:
                x_, hs_ = self.transformer(
                    torch.cat([x0, x1], dim=0),
                    torch.cat([hs1, hs0], dim=0),
                    torch.cat([x0_mask, x1_mask], dim=0),
                    None,
                    name,
                    layer,
                )
                x0, x1 = x_.split(bs)
                hs1, hs0 = hs_.split(bs)
            else:
                x0, hs1 = self.transformer(x0, hs1, x0_mask, None, name, layer)
                x1, hs0 = self.transformer(x1, hs0, x1_mask, None, name, layer)

        # stage 3
        for i, (layer, name) in enumerate(zip(self.layers3, self.layer_names3)):
            x0, x1 = self.transformer(x0, x1, x0_mask, x1_mask, name, layer)

        return x0, x1, flow_list

    def forward(self, data, x0, x1, pos_encoding0, pos_encoding1, x0_mask=None, x1_mask=None, use_cas=True):
        """
        Args:
            x0 (torch.Tensor): [N, C, H, W]
            x1 (torch.Tensor): [N, C, H, W]
            pos0, pos1:  [N, C, H, W]
        Outputs:
            out0: [N,-1,C]
            out1: [N,-1,C]
            flow_list: [L,N,H,W,4]*1(2)
        """

        out0, out1, flow_list = self.feature_interaction(
            data, x0, x1, pos_encoding0, pos_encoding1, x0_mask, x1_mask
        )
        
        return out0, out1, flow_list