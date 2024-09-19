import copy
import pdb
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from einops.einops import rearrange
from itertools import product

from .linear_attention import (
    FullAttention,
    LinearAttention,
    MultiHeadAttention,
)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention="linear"):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = (
            LinearAttention()
            if attention == "linear"
            else FullAttention()
        )
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.pre_norm_q = nn.LayerNorm(d_model)
        self.pre_norm_kv = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        # pdb.set_trace()
        bs = x.size(0)
        query, key, value = (
            self.pre_norm_q(x),
            self.pre_norm_kv(source),
            self.pre_norm_kv(source),
        )

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        # message = self.norm1(message)

        # feed-forward network
        x = x + message
        message2 = self.mlp(self.norm2(x))

        return x + message2


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (AdaMatcher) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.layer_names = config["layer_names"]
        encoder_layer = EncoderLayer(
            config["d_model"], config["nhead"], config["attention"]
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(
            2
        ), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


######################################################################
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, attention="linear"):
        super(DecoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_pos=None, m_pos=None
    ):
        """
        Args:
            tgt (torch.Tensor): [N, L, C]
            memory (torch.Tensor): [N, S, C]
            tgt_mask (torch.Tensor): [N, L] (optional)
            memory_mask (torch.Tensor): [N, S] (optional)
        """
        # pdb.set_trace()
        bs = tgt.size(0)
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, tgt_pos)
        tgt2 = self.self_attn(q, k, v=tgt2, q_mask=tgt_mask, kv_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            q=self.with_pos_embed(tgt2, tgt_pos),
            k=self.with_pos_embed(memory, m_pos),
            v=memory,
            q_mask=tgt_mask,
            kv_mask=memory_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.mlp(tgt2)
        tgt = tgt + tgt2

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
    ):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_pos=tgt_pos,
                m_pos=m_pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

class HierachicalAttention(nn.Module):
    def __init__(self,d_model,nhead,nsample,radius_scale,nlevel=3):
        super().__init__()
        self.d_model=d_model
        self.nhead=nhead
        self.nsample=nsample
        self.nlevel=nlevel
        self.radius_scale=radius_scale
        self.merge_head = nn.Sequential(
            nn.Conv1d(d_model*3, d_model, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv1d(d_model, d_model, kernel_size=1,bias=False),
        )
        self.fullattention=FullAttentionInFlow(d_model,nhead)
        self.temp=nn.parameter.Parameter(torch.tensor(1.),requires_grad=True) 
        sample_offset=torch.tensor([[pos[0]-nsample[1]/2+0.5, pos[1]-nsample[1]/2+0.5] for pos in product(range(nsample[1]), range(nsample[1]))]) #r^2*2
        self.sample_offset=nn.parameter.Parameter(sample_offset,requires_grad=False)

    def forward(self,query,key,value,flow,size_q,size_kv,mask0=None, mask1=None,ds0=[4,4],ds1=[4,4]):
        """
        Args:
            q,k,v (torch.Tensor): [B, C, L]
            mask (torch.Tensor): [B, L]
            flow (torch.Tensor): [B, H, W, 4]
        Return:
            all_message (torch.Tensor): [B, C, H, W]
        """
        
        variance=flow[:,:,:,2:]
        offset=flow[:,:,:,:2]  #B*H*W*2
        bs=query.shape[0]
        h0,w0=size_q[0],size_q[1]
        h1,w1=size_kv[0],size_kv[1]
        variance=torch.exp(0.5*variance)*self.radius_scale #b*h*w*2(pixel scale)
        span_scale=torch.clamp((variance*2/self.nsample[1]),min=1) #b*h*w*2

        sub_sample0,sub_sample1=[ds0,2,1],[ds1,2,1]
        q_list=[F.avg_pool2d(query.view(bs,-1,h0,w0),kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample0]
        k_list=[F.avg_pool2d(key.view(bs,-1,h1,w1),kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample1]
        v_list=[F.avg_pool2d(value.view(bs,-1,h1,w1),kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample1] #n_level
        
        offset_list=[F.avg_pool2d(offset.permute(0,3,1,2),kernel_size=sub_size*self.nsample[0],stride=sub_size*self.nsample[0]).permute(0,2,3,1)/sub_size for sub_size in sub_sample0[1:]] #n_level-1
        span_list=[F.avg_pool2d(span_scale.permute(0,3,1,2),kernel_size=sub_size*self.nsample[0],stride=sub_size*self.nsample[0]).permute(0,2,3,1) for sub_size in sub_sample0[1:]] #n_level-1

        if mask0 is not None:
            mask0,mask1=mask0.view(bs,1,h0,w0),mask1.view(bs,1,h1,w1)
            mask0_list=[-F.max_pool2d(-mask0,kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample0]
            mask1_list=[-F.max_pool2d(-mask1,kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample1]
        else:
            mask0_list=mask1_list=[None,None,None]

        message_list=[]
        #full attention at coarse scale
        mask0_flatten=mask0_list[0].view(bs,-1) if mask0 is not None else None
        mask1_flatten=mask1_list[0].view(bs,-1) if mask1 is not None else None
        message_list.append(self.fullattention(q_list[0],k_list[0],v_list[0],mask0_flatten,mask1_flatten,self.temp).view(bs,self.d_model,h0//ds0[0],w0//ds0[1]))

        for index in range(1,self.nlevel):
            q,k,v=q_list[index],k_list[index],v_list[index]
            mask0,mask1=mask0_list[index],mask1_list[index]
            s,o=span_list[index-1],offset_list[index-1] #B*h*w(*2)
            q,k,v,sample_pixel,mask_sample=self.partition_token(q,k,v,o,s,mask0) #B*Head*D*G*N(G*N=H*W for q)
            message_list.append(self.group_attention(q,k,v,1,mask_sample).view(bs,self.d_model,h0//sub_sample0[index],w0//sub_sample0[index]))
        #fuse
        all_message=torch.cat([F.upsample(message_list[idx],scale_factor=sub_sample0[idx],mode='nearest') \
                    for idx in range(self.nlevel)],dim=1).view(bs,-1,h0*w0) #b*3d*H*W
        
        all_message=self.merge_head(all_message).view(bs,-1,h0,w0) #b*d*H*W
        return all_message
      
    def partition_token(self,q,k,v,offset,span_scale,maskv):
        #q,k,v: B*C*H*W
        #o: B*H/2*W/2*2
        #span_scale:B*H*W
        bs=q.shape[0]
        h,w=q.shape[2],q.shape[3]
        hk,wk=k.shape[2],k.shape[3]
        offset=offset.view(bs,-1,2)
        span_scale=span_scale.view(bs,-1,1,2)
        #B*G*2
        offset_sample=self.sample_offset[None,None]*span_scale
        sample_pixel=offset[:,:,None]+offset_sample#B*G*r^2*2
        sample_norm=sample_pixel/torch.tensor([wk/2,hk/2]).cuda()[None,None,None]-1
        
        q = q.view(bs, -1 , h // self.nsample[0], self.nsample[0], w // self.nsample[0], self.nsample[0]).\
                permute(0, 1, 2, 4, 3, 5).contiguous().view(bs, self.nhead,self.d_model//self.nhead, -1,self.nsample[0]**2)#B*head*D*G*N(G*N=H*W for q)
        #sample token
        k=F.grid_sample(k, grid=sample_norm).view(bs, self.nhead,self.d_model//self.nhead,-1, self.nsample[1]**2) #B*head*D*G*r^2
        v=F.grid_sample(v, grid=sample_norm).view(bs, self.nhead,self.d_model//self.nhead,-1, self.nsample[1]**2) #B*head*D*G*r^2
        #import pdb;pdb.set_trace()
        if maskv is not None:
            mask_sample=F.grid_sample(maskv.view(bs,-1,h,w).float(),grid=sample_norm,mode='nearest')==1 #B*1*G*r^2
        else:
            mask_sample=None
        return q,k,v,sample_pixel,mask_sample


    def group_attention(self,query,key,value,temp,mask_sample=None):
        #q,k,v: B*Head*D*G*N(G*N=H*W for q)
        bs=query.shape[0]
        #import pdb;pdb.set_trace()
        QK = torch.einsum("bhdgn,bhdgm->bhgnm", query, key)
        if mask_sample is not None:
            num_head,number_n=QK.shape[1],QK.shape[3]
            QK.masked_fill_(~(mask_sample[:,:,:,None]).expand(-1,num_head,-1,number_n,-1).bool(), float(-1e8))
        # Compute the attention and the weighted average
        softmax_temp = temp / query.size(2)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-1)
        queried_values = torch.einsum("bhgnm,bhdgm->bhdgn", A, value).contiguous().view(bs,self.d_model,-1)
        return queried_values
    
class layernorm2d(nn.Module):
     
     def __init__(self,dim) :
         super().__init__()
         self.dim=dim
         self.affine=nn.parameter.Parameter(torch.ones(dim), requires_grad=True)
         self.bias=nn.parameter.Parameter(torch.zeros(dim), requires_grad=True) 
    
     def forward(self,x):
        #x: B*C*H*W
        mean,std=x.mean(dim=1,keepdim=True),x.std(dim=1,keepdim=True)
        return self.affine[None,:,None,None]*(x-mean)/(std+1e-6)+self.bias[None,:,None,None]

class FullAttentionInFlow(nn.Module):
    def __init__(self,d_model,nhead):
        super().__init__()
        self.d_model=d_model
        self.nhead=nhead

    def forward(self, q, k,v , mask0=None, mask1=None, temp=1):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            q,k,v: [N, D, L]
            mask: [N, L]
        Returns:
            msg: [N,L]
        """
        bs=q.shape[0]
        q,k,v=q.view(bs,self.nhead,self.d_model//self.nhead,-1),k.view(bs,self.nhead,self.d_model//self.nhead,-1),v.view(bs,self.nhead,self.d_model//self.nhead,-1)
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nhdl,nhds->nhls", q, k)
        if mask0 is not None:
            QK.masked_fill_(~(mask0[:,None, :, None] * mask1[:, None, None]).bool(), float(-1e8))
        # Compute the attention and the weighted average
        softmax_temp = temp / q.size(2)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-1)
        queried_values = torch.einsum("nhls,nhds->nhdl", A, v).contiguous().view(bs,self.d_model,-1)
        return queried_values


class messageLayer_ini(nn.Module):

    def __init__(self, d_model, d_flow,d_value, nhead):
        super().__init__()
        super(messageLayer_ini, self).__init__()

        self.d_model = d_model
        self.d_flow = d_flow
        self.d_value=d_value
        self.nhead = nhead
        self.attention = FullAttentionInFlow(d_model, nhead)

        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1,bias=False)
        self.k_proj = nn.Conv1d(d_model, d_model, kernel_size=1,bias=False)
        self.v_proj = nn.Conv1d(d_value, d_model, kernel_size=1,bias=False)
        self.merge_head=nn.Conv1d(d_model,d_model,kernel_size=1,bias=False)

        self.merge_f= self.merge_f = nn.Sequential(
            nn.Conv2d(d_model*2, d_model*2, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d_model*2, d_model, kernel_size=1, bias=False),
        )

        self.norm1 = layernorm2d(d_model)
        self.norm2 = layernorm2d(d_model)


    def forward(self, x0, x1,pos0,pos1,mask0=None,mask1=None):
        #x1,x2: b*d*L
        x0,x1=self.update(x0,x1,pos1,mask0,mask1),\
                self.update(x1,x0,pos0,mask1,mask0)
        return x0,x1


    def update(self,f0,f1,pos1,mask0,mask1):
        """
        Args:
            f0: [N, D, H, W]
            f1: [N, D, H, W]
        Returns:
            f0_new: (N, d, h, w)
        """
        bs,h,w=f0.shape[0],f0.shape[2],f0.shape[3]

        f0_flatten,f1_flatten=f0.view(bs,self.d_model,-1),f1.view(bs,self.d_model,-1)
        pos1_flatten=pos1.view(bs,self.d_value-self.d_model,-1)
        f1_flatten_v=torch.cat([f1_flatten,pos1_flatten],dim=1)

        queries,keys=self.q_proj(f0_flatten),self.k_proj(f1_flatten)
        values=self.v_proj(f1_flatten_v).view(bs,self.nhead,self.d_model//self.nhead,-1)
        
        queried_values=self.attention(queries,keys,values,mask0,mask1)
        msg=self.merge_head(queried_values).view(bs,-1,h,w)
        msg=self.norm2(self.merge_f(torch.cat([f0,self.norm1(msg)],dim=1)))
        return f0+msg
    
class flow_initializer(nn.Module):

    def __init__(self, dim, dim_flow, nhead, layer_num):
        super().__init__()
        self.layer_num= layer_num # default:2
        self.dim = dim
        self.dim_flow = dim_flow

        encoder_layer = messageLayer_ini(
            dim ,dim_flow,dim+dim_flow , nhead) # 一个full cross attention
        self.layers_coarse = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(layer_num)])
        self.decoupler = nn.Conv2d(
                self.dim, self.dim+self.dim_flow, kernel_size=1)
        self.up_merge = nn.Conv2d(2*dim, dim, kernel_size=1)

    def forward(self, feat0, feat1,pos0,pos1,mask0=None,mask1=None,ds0=[4,4],ds1=[4,4]):
        # feat0: [B, C, H0, W0]
        # feat1: [B, C, H1, W1]
        # use low-res MHA to initialize flow feature
        bs = feat0.size(0)
        h0,w0,h1,w1=feat0.shape[2],feat0.shape[3],feat1.shape[2],feat1.shape[3]

        # coarse level
        sub_feat0, sub_feat1 = F.avg_pool2d(feat0, ds0, stride=ds0), \
                            F.avg_pool2d(feat1, ds1, stride=ds1)

        sub_pos0,sub_pos1=F.avg_pool2d(pos0, ds0, stride=ds0), \
                            F.avg_pool2d(pos1, ds1, stride=ds1)
    
        if mask0 is not None:
            mask0,mask1=-F.max_pool2d(-mask0.view(bs,1,h0,w0),ds0,stride=ds0).view(bs,-1),\
                        -F.max_pool2d(-mask1.view(bs,1,h1,w1),ds1,stride=ds1).view(bs,-1)
        
        for layer in self.layers_coarse:
            sub_feat0, sub_feat1 = layer(sub_feat0, sub_feat1,sub_pos0,sub_pos1,mask0,mask1)
        # decouple flow and visual features -- dim --> dim+dim_flow
        decoupled_feature0, decoupled_feature1 = self.decoupler(sub_feat0),self.decoupler(sub_feat1) 

        sub_feat0, sub_flow_feature0 = decoupled_feature0[:,:self.dim], decoupled_feature0[:, self.dim:]
        sub_feat1, sub_flow_feature1 = decoupled_feature1[:,:self.dim], decoupled_feature1[:, self.dim:]

        update_feat0, flow_feature0 = F.upsample(sub_feat0, scale_factor=ds0, mode='bilinear'),\
                                        F.upsample(sub_flow_feature0, scale_factor=ds0, mode='bilinear')
        update_feat1, flow_feature1 = F.upsample(sub_feat1, scale_factor=ds1, mode='bilinear'),\
                                        F.upsample(sub_flow_feature1, scale_factor=ds1, mode='bilinear')
        
        feat0 = feat0+self.up_merge(torch.cat([feat0, update_feat0], dim=1))
        feat1 = feat1+self.up_merge(torch.cat([feat1, update_feat1], dim=1))
    
        return feat0,feat1,flow_feature0,flow_feature1 #b*c*h*w

class messageLayer_gla(nn.Module):

    def __init__(self,d_model,d_flow,d_value,
                    nhead,radius_scale,nsample,update_flow=True):
        super().__init__()
        self.d_model = d_model
        self.d_flow=d_flow
        self.d_value=d_value
        self.nhead = nhead
        self.radius_scale=radius_scale
        self.update_flow=update_flow
        self.flow_decoder=nn.Sequential(
                    nn.Conv1d(d_flow, d_flow//2, kernel_size=1, bias=False),
                    nn.ReLU(True),
                    nn.Conv1d(d_flow//2, 4, kernel_size=1, bias=False))
        self.attention=HierachicalAttention(d_model,nhead,nsample,radius_scale)

        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1,bias=False)
        self.k_proj = nn.Conv1d(d_model, d_model, kernel_size=1,bias=False)
        self.v_proj = nn.Conv1d(d_value, d_model, kernel_size=1,bias=False)

        d_extra=d_flow if update_flow else 0
        self.merge_f=nn.Sequential(
                     nn.Conv2d(d_model*2+d_extra, d_model+d_flow, kernel_size=1, bias=False),
                     nn.ReLU(True),
                     nn.Conv2d(d_model+d_flow, d_model+d_extra, kernel_size=3,padding=1, bias=False),
                )
        self.norm1 = layernorm2d(d_model)
        self.norm2 = layernorm2d(d_model+d_extra)

    def forward(self, x0, x1, flow_feature0,flow_feature1,pos0,pos1,mask0=None,mask1=None,ds0=[4,4],ds1=[4,4]):
        """
        Args:
            x0 (torch.Tensor): [B, C, H, W]
            x1 (torch.Tensor): [B, C, H, W]
            flow_feature0 (torch.Tensor): [B, C', H, W]
            flow_feature1 (torch.Tensor): [B, C', H, W]
        """
        flow0,flow1=self.decode_flow(flow_feature0,flow_feature1.shape[2:]),self.decode_flow(flow_feature1,flow_feature0.shape[2:]) # 把flow feature变成flow [N, H, W, 4]
        x0_new,flow_feature0_new=self.update(x0,x1,flow0.detach(),flow_feature0,pos1,mask0,mask1,ds0,ds1)
        x1_new,flow_feature1_new=self.update(x1,x0,flow1.detach(),flow_feature1,pos0,mask1,mask0,ds1,ds0)
        return x0_new,x1_new,flow_feature0_new,flow_feature1_new,flow0,flow1

    def update(self,x0,x1,flow0,flow_feature0,pos1,mask0,mask1,ds0,ds1):
        bs=x0.shape[0]
        queries,keys=self.q_proj(x0.view(bs,self.d_model,-1)),self.k_proj(x1.view(bs,self.d_model,-1))
        x1_pos=torch.cat([x1,pos1],dim=1)
        values=self.v_proj(x1_pos.view(bs,self.d_value,-1))
        msg=self.attention(queries,keys,values,flow0,x0.shape[2:],x1.shape[2:],mask0,mask1,ds0,ds1)

        if self.update_flow:
            update_feature=torch.cat([x0,flow_feature0],dim=1)
        else:
            update_feature=x0
        msg=self.norm2(self.merge_f(torch.cat([update_feature,self.norm1(msg)],dim=1)))
        update_feature=update_feature+msg

        x0_new,flow_feature0_new=update_feature[:,:self.d_model],update_feature[:,self.d_model:]
        return x0_new,flow_feature0_new

    def decode_flow(self,flow_feature,kshape):
        bs,h,w=flow_feature.shape[0],flow_feature.shape[2],flow_feature.shape[3]
        scale_factor=torch.tensor([kshape[1],kshape[0]]).cuda()[None,None,None]
        flow=self.flow_decoder(flow_feature.view(bs,-1,h*w)).permute(0,2,1).view(bs,h,w,4)
        flow_coordinates=torch.sigmoid(flow[:,:,:,:2])*scale_factor
        flow_var=flow[:,:,:,2:]
        flow=torch.cat([flow_coordinates,flow_var],dim=-1) #B*H*W*4
        return flow
    
class LocalFeatureTransformer_Flow(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer_Flow, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']

        self.pos_transform=nn.Conv2d(config['d_model'],config['d_flow'],kernel_size=1,bias=False)
        self.ini_layer = flow_initializer(self.d_model, config['d_flow'], config['nhead'],config['ini_layer_num'])
        
        encoder_layer = messageLayer_gla(
            config['d_model'], config['d_flow'], config['d_flow']+config['d_model'], config['nhead'],config['radius_scale'],config['nsample'])
        encoder_layer_last=messageLayer_gla(
            config['d_model'], config['d_flow'], config['d_flow']+config['d_model'], config['nhead'],config['radius_scale'],config['nsample'],update_flow=False)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(config['layer_num']-1)]+[encoder_layer_last])
        self._reset_parameters()   
        
    def _reset_parameters(self):
        for name,p in self.named_parameters():
            if 'temp' in name or 'sample_offset' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, pos0, pos1, mask0=None, mask1=None, ds0=[4,4], ds1=[4,4]):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            pos1,pos2:  [N, C, H, W]
        Outputs:
            feat0: [N,-1,C]
            feat1: [N,-1,C]
            flow_list: [L,N,H,W,4]*1(2)
        """
        flow_list=[[],[]]# [px,py,sx,sy] 
        bs = feat0.size(0)
        
        pos0,pos1=self.pos_transform(pos0),self.pos_transform(pos1)
        pos0,pos1=pos0.expand(bs,-1,-1,-1),pos1.expand(bs,-1,-1,-1)
        assert self.d_model == feat0.size(
            1), "the feature number of src and transformer must be equal"
                  
        if mask0 is not None:
            mask0,mask1=mask0[:,None].float(),mask1[:,None].float()
        feat0,feat1, flow_feature0, flow_feature1 = self.ini_layer(feat0, feat1,pos0,pos1,mask0,mask1,ds0,ds1)
        for layer in self.layers:
            feat0,feat1,flow_feature0,flow_feature1,flow0,flow1=layer(feat0,feat1,flow_feature0,flow_feature1,pos0,pos1,mask0,mask1,ds0,ds1)
            flow_list[0].append(flow0)
            flow_list[1].append(flow1)
        flow_list[0]=torch.stack(flow_list[0],dim=0)
        flow_list[1]=torch.stack(flow_list[1],dim=0)
        feat0, feat1 = feat0.permute(0, 2, 3, 1).view(bs, -1, self.d_model), feat1.permute(0, 2, 3, 1).view(bs, -1, self.d_model)
        return feat0, feat1, flow_list
       