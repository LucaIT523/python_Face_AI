import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.vecquan_arch import *

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.data.size()[:2] == style_feat.data.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

class TransLayer(nn.Module):
    def __init__(self, dim, dimMlp, dpout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, 8, dpout)
        self.linear1 = nn.Linear(dim, dimMlp)
        self.linear2 = nn.Linear(dimMlp, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dpout = nn.Dropout(dpout)
        self.dpout1 = nn.Dropout(dpout)
        self.dpout2 = nn.Dropout(dpout)
        self.activation = F.gelu

    def pos_emb(self, tens, query: Optional[Tensor]):
        return tens if query is None else tens + query

    def forward(self, tgt, tgt_mask: Optional[Tensor] = None,  tgt_key_padding_mask: Optional[Tensor] = None,  query: Optional[Tensor] = None):
        qtg = self.pos_emb(self.norm1(tgt), query)
        tgt = tgt + self.dpout1(self.self_attn(qtg, qtg, value=self.norm1(tgt), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0])
        tgt2 = self.linear2(self.dpout(self.activation(self.linear1(self.norm2(tgt)))))
        return tgt + self.dpout2(tgt2)


class Fuseblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)
        self.scale = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.shift = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(out_ch, out_ch, 3, padding=1))

    def forward(self, encft, decft, w=1):
        encft = self.encode_enc(torch.cat([encft, decft], dim=1))
        return decft + w * (decft * self.scale(encft) + self.shift(encft))


@ARCH_REGISTRY.register()
class FaceSuper(VQAutoEncoderEx):
    def __init__(self, dim, cbSize, conlist):
        super(FaceSuper, self).__init__(512, cbSize)
        # check param
        for module in ['quantize', 'generator']:
            for param in getattr(self, module).parameters():
                param.requires_grad = False

        self.layers = 9
        self.conlist = conlist
        self.position_emb = nn.Parameter(torch.zeros(256, dim))
        self.feat_emb = nn.Linear(256, dim)
        self.ft_layers = nn.Sequential(*[TransLayer(dim, 2 * dim, 0.0) for _ in range(self.layers)])
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, cbSize, bias=False))

        self.channels = {'16': 512, '32': 256, '64': 256, '128': 128, '256': 128, '512': 64, }
        self.fuse_encoder_block = {'512': 2, '256': 5, '128': 8, '64': 11, '32': 14, '16': 18}
        self.fuse_generator_block = {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}
        self.fuse_convs_dict = nn.ModuleDict()

        for k in self.conlist:
            ch = self.channels[k]
            self.fuse_convs_dict[k] = Fuseblock(ch, ch)

    def forward(self, x, w_f, ada_in):
        encftdict = {}
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in [self.fuse_encoder_block[f_size] for f_size in self.conlist]:
                encftdict[str(x.shape[-1])] = x.clone()

        rtfeat = x
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        query_emb = self.feat_emb(rtfeat.flatten(2).permute(2, 0, 1))
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query=pos_emb)

        rtlg_tenso = self.idx_pred_layer(query_emb).permute(1, 0, 2)
        _, _idx = torch.topk(F.softmax(rtlg_tenso, 2), 1, 2)
        qt_ft = self.quantize.get_codebook_entry(_idx, shape=[x.shape[0], 16, 16, 256]).detach()

        if ada_in: qt_ft = adaptive_instance_normalization(qt_ft, rtfeat)
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.conlist]

        rt = qt_ft
        for i, block in enumerate(self.generator.blocks):
            rt = block(rt)
            if i in fuse_list:
                k = str(rt.shape[-1])
                if w_f > 0: rt = self.fuse_convs_dict[k](encftdict[k].detach(), rt, w_f)

        return rt, rtlg_tenso, rtfeat