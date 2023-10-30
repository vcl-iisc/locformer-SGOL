# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------
"""Build a VIDT detector for object detection."""

from turtle import pos
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import (nested_tensor_from_tensor_list,
                        inverse_sigmoid, NestedTensor)
from methods.swin_w_ram import swin_nano, swin_tiny, swin_small, swin_base_win7, swin_large_win7
from methods.coat_w_ram import coat_lite_tiny, coat_lite_mini, coat_lite_small
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessor import PostProcess
from .deformable_transformer import build_deforamble_transformer
from methods.vidt.fpn_fusion import FPNFusionModule
import copy
import math
import torchvision
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def _get_clones(module, N):
    """ Clone a moudle N times """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")




class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(150, num_pos_feats)
        self.col_embed = nn.Embedding(150, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, image_tensor):
        x = image_tensor
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0)
        # .repeat(x.shape[0], 1, 1, 1)
        return pos
def default(val, default_val):
    return val if val is not None else default_val
def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor




class CrossMHAFusion(nn.Module):
    """ A decoder layer.

    Parameters:
        d_model: the channel dimension for attention [default=256]
        d_ffn: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        n_levels: the number of scales for extracted features [default=4]
        n_heads: the number of heads [default=8]
        n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
    """

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.4, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, drop_path=0.):
        super().__init__()

        # [DET x PATCH] deformable cross-attention
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # [DET x DET] self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.self_attn_sk = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn for multi-heaed
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)


        # stochastic depth
        self.drop_path = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt, tgt2):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
    
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, sk, pos_sk):

        # [DET] self-attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(sk, pos_sk)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), sk.transpose(0, 1))[0].transpose(0, 1)
     
        tgt = self.forward_ffn(tgt, tgt2)

        return tgt, sk

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, image_tensor):
        x = image_tensor        
        y_embed = torch.ones_like(x[:,0,:,:]).squeeze(1).cumsum(1, dtype=torch.float32)
        x_embed = torch.ones_like(x[:,0,:,:]).squeeze(1).cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb


from collections import OrderedDict

class Detector(nn.Module):
    """ This is a combination of "Swin with RAM" and a "Neck-free Deformable Decoder" """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, with_box_refine=False,
                 # The three techniques were not used in ViDT paper.
                 # After submitting our paper, we saw the ViDT performance could be further enhanced with them.
                 cross_scale_fusion=None, iou_aware=False, token_label=False,
                 distil=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries (i.e., det tokens). This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            cross_scale_fusion: None or fusion module available
            iou_aware: True if iou_aware is to be used.
              see the original paper https://arxiv.org/abs/1912.05992
            token_label: True if token_label is to be used.
              see the original paper https://arxiv.org/abs/2104.10858
            distil: whether to use knowledge distillation with token matching
        """

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed_v2 = nn.Linear(hidden_dim*2, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.backbone = backbone

        # two essential techniques used [default use]
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        # three additional techniques not used in the ViDT paper
        # optional use, we will revise our paper for the below techniques
        self.iou_aware = iou_aware
        self.token_label = token_label

        # distillation
        self.distil = distil
        num_backbone_outs = len(backbone.num_channels)
        # [PATCH] token channel reduction for the input to transformer decoder
        # if cross_scale_fusion is None:
        if True:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                  # This is 1x1 conv -> so linear layer
                  nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                  nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)

            # initialize the projection layer for [PATCH] tokens
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            self.fusion = None
        else:
            # the cross scale fusion module has its own reduction layers
            self.fusion = cross_scale_fusion

        # channel dim reduction for [DET] tokens
        self.tgt_proj = nn.Sequential(
              # This is 1x1 conv -> so linear layer
              nn.Conv2d(self.backbone.num_channels[-2], hidden_dim, kernel_size=1),
              nn.GroupNorm(32, hidden_dim),
            )

        # channel dim reductionfor [DET] learnable pos encodings
        self.query_pos_proj = nn.Sequential(
              # This is 1x1 conv -> so linear layer
              nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
              nn.GroupNorm(32, hidden_dim),
            )

        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed_v2.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # initialize projection layer for [DET] tokens and encodings
        nn.init.xavier_uniform_(self.tgt_proj[0].weight, gain=1)
        nn.init.constant_(self.tgt_proj[0].bias, 0)
        nn.init.xavier_uniform_(self.query_pos_proj[0].weight, gain=1)
        nn.init.constant_(self.query_pos_proj[0].bias, 0)

        self.sketch_embedding = torchvision.models.resnet50(pretrained=True)
        if True:
            state_dict = torch.load("/path/to/sketch-encoder/best_model.pt")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[13:] # remove `module.`
                new_state_dict[name] = v
            del new_state_dict['fc.weight']
            del new_state_dict['fc.bias']
        
            self.sketch_embedding.load_state_dict(new_state_dict, strict=False)

  
        self.gp_norm = nn.GroupNorm(32, 2048)



        self.sketch_embedding.avgpool = nn.Identity()
        self.sketch_embedding.fc = nn.Identity()

        self.sketch_proj = nn.Conv2d(in_channels=2048, out_channels = 256, kernel_size=1, stride=1, padding=0)

        self.sketch_proj_query = nn.Conv2d(in_channels=2048, out_channels = 768, kernel_size=1, stride=1, padding=0)

        

        self.query_fusion = []
        self.query_fusion_sketch = []
        for i in range(7):
            self.query_fusion.append(CrossMHAFusion())
            self.query_fusion_sketch.append(CrossMHAFusion())
           
        self.query_fusion = nn.ModuleList(self.query_fusion)
        self.query_fusion_sketch = nn.ModuleList(self.query_fusion_sketch)
        self.sketch_query_pos = PositionEmbeddingSine()


        # the prediction is made for each decoding layers + the standalone detector (Swin with RAM)
        num_pred = transformer.decoder.num_layers + 1

        # set up all required nn.Module for additional techniques
        if with_box_refine:
            self.class_embed_v2 = _get_clones(self.class_embed_v2, num_pred)
            self.trans = _get_clones(self.trans, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed_v2 = nn.ModuleList([self.class_embed_v2 for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if self.iou_aware:
            self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            if with_box_refine:
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            else:
                self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])
        
        self.pos_proj = nn.Linear(256,256)



    def forward(self, samples: NestedTensor, sketches:torch.Tensor):
        """ The forward step of ViDT

        Parameters:
            The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dictionary having the key and value pairs below:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
                            If iou_aware is True, "pred_ious" is also returns as one of the key in "aux_outputs"
            - "enc_tokens": If token_label is True, "enc_tokens" is returned to be used

            Note that aux_loss and box refinement is used in ViDT in default. The detailed ablation of using
            the cross_scale_fusion, iou_aware & token_lablel loss will be discussed in a later version
        """


        x = samples[0]
        mask = samples[1]

        sketches = self.sketch_embedding(sketches)
        
        sketches = sketches.view(sketches.shape[0],2048,7,7)
        sketches = self.gp_norm(sketches)
        #  project sketches to the same dimension as x
        sketches = self.sketch_proj(sketches)

        sketches = sketches.view(bs,256,7,7)


        # return multi-scale [PATCH] tokens along with final [DET] tokens and their pos encodings
        # use the sketch guided transformer to generate features for images (x)
        features, det_tgt, det_pos = self.backbone(x, mask, sketches)
        

        # [DET] token and encoding projection to compact representation for the input to the Neck-free transformer
        
        det_tgt = self.tgt_proj(det_tgt.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        det_pos = self.query_pos_proj(det_pos.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        
        
        
        # [PATCH] token projection
        shapes = []
        for l, src in enumerate(features):
            shapes.append(src.shape[-2:])
        
        srcs = []
        # if self.fusion is None:
        if True:
            for l, src in enumerate(features):
                src = self.input_proj[l](src)
                srcs.append(src)
        else:
            # multi-scale fusion is used if fusion is not None
            srcs = self.fusion(features)
        

        masks = []
        for l, src in enumerate(srcs):
            # resize mask
            shapes.append(src.shape[-2:])
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            assert mask is not None

        outputs_classes = []
        outputs_coords = []

        # return the output of the neck-free decoder
        hs, init_reference, inter_references, enc_token_class_unflat = \
          self.transformer(srcs, masks, det_tgt, det_pos, sketches)
        
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            a = hs[lvl]
            
            if lvl >2:
                sketch_query_pos = self.sketch_query_pos(sketches)
                bs,d,w,h = sketches.shape
                
                tokens_sk = sketches.view(bs,d,-1).permute(0,2,1)
                pos_img = det_pos
                pos_sk = sketch_query_pos.view(bs,d,-1).permute(0,2,1)
                # refine features of objects given the sketch features
                a, sket = self.query_fusion[lvl](a,pos_img, tokens_sk, pos_sk)
                # refine features of sketches given the object features
                sketches, a_tgt = self.query_fusion_sketch[lvl](tokens_sk, pos_sk, a , pos_img)
                

                sketches = sketches.permute(0,2,1).view(bs,d,w,h)

            else:
                glob_sketch = sketches.max(-1)[0].max(-1)[0]
                glob_sketch = glob_sketch.unsqueeze(1).repeat(1, self.num_queries,1)
                a = hs[lvl]

            
            
            sk = self.trans[lvl](glob_sketch)
                        
            # a = torch.cat([sk, hs[lvl]], dim=-1)
            a = torch.cat([sk, a], dim=-1)
            outputs_class = self.class_embed_v2[lvl](a)
        
            ## bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl])
            
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                # print(reference.shape)
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # stack all predictions made from each decoding layers
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
     
        # final prediction is made the last decoding layer
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # aux loss is defined by using the rest predictions
        if self.aux_loss and self.transformer.decoder.num_layers > 0:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.iou_aware:
            outputs_ious = []
            for lvl in range(hs.shape[0]):
                outputs_ious.append(self.iou_embed[lvl](hs[lvl]))
            outputs_iou = torch.stack(outputs_ious)
            out['pred_ious'] = outputs_iou[-1]

            if self.aux_loss:
                for i, aux in enumerate(out['aux_outputs']):
                    aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

        if self.distil:
            # 'patch_token': multi-scale patch tokens from each stage
            # 'body_det_token' and 'neck_det_tgt': the input det_token for multiple detection heads
            out['distil_tokens'] = {'patch_token': srcs, 'body_det_token': det_tgt, 'neck_det_token': hs}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
  """ Very simple multi-layer perceptron (also called FFN)"""

  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
      super().__init__()
      self.num_layers = num_layers
      h = [hidden_dim] * (num_layers - 1)
      self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

  def forward(self, x):
      for i, layer in enumerate(self.layers):
          x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
      return x

def build(args, is_teacher=False):

    # a teacher model for distilation
    if is_teacher:
        return build_teacher(args)
    #

    if args.dataset_file == 'coco':
        # num_classes = 91
        num_classes = 2

    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    if args.backbone_name == 'swin_nano':
        backbone, hidden_dim = swin_nano(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_tiny':
        backbone, hidden_dim = swin_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_small':
        backbone, hidden_dim = swin_small(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_base_win7_22k':
        backbone, hidden_dim = swin_base_win7(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_large_win7_22k':
        backbone, hidden_dim = swin_large_win7(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_tiny':
        backbone, hidden_dim = coat_lite_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_mini':
        backbone, hidden_dim = coat_lite_mini(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_small':
        backbone, hidden_dim = coat_lite_small(pretrained=args.pre_trained)
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    backbone.finetune_det(method=args.method,
                          det_token_num=args.det_token_num,
                          pos_dim=args.reduced_dim,
                          cross_indices=args.cross_indices)

    cross_scale_fusion = None
    # if args.cross_scale_fusion:
    if False:
        cross_scale_fusion = FPNFusionModule(backbone.num_channels, fuse_dim=args.reduced_dim)

    deform_transformers = build_deforamble_transformer(args)

    model = Detector(
        backbone,
        deform_transformers,
        num_classes=num_classes,
        num_queries=args.det_token_num,
        # two essential techniques used in ViDT
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        # three additional techniques (optionally)
        cross_scale_fusion=cross_scale_fusion,
        iou_aware=args.iou_aware,
        token_label=args.token_label,
        # distil
        distil=False if args.distil_model is None else True,
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    ##
    if args.iou_aware:
        weight_dict['loss_iouaware'] = args.iouaware_loss_coef

    if args.token_label:
        weight_dict['loss_token_focal'] = args.token_loss_coef
        weight_dict['loss_token_dice'] = args.token_loss_coef

    if args.distil_model is not None:
        weight_dict['loss_distil'] = args.distil_loss_coef

    # aux decoding loss
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1 + 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.iou_aware:
        losses += ['iouaware']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args.dataset_file)}

    return model, criterion, postprocessors


def build_teacher(args):

    if args.dataset_file == 'coco':
        num_classes = 91

    if args.dataset_file == "coco_panoptic":
        num_classes = 250

    if args.distil_model == 'vidt_nano':
        backbone, hidden_dim = swin_nano()
    elif args.distil_model == 'vidt_tiny':
        backbone, hidden_dim = swin_tiny()
    elif args.distil_model == 'vidt_small':
        backbone, hidden_dim = swin_small()
    elif args.distil_model == 'vidt_base':
        backbone, hidden_dim = swin_base_win7()
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    backbone.finetune_det(method=args.method,
                          det_token_num=args.det_token_num,
                          pos_dim=args.reduced_dim,
                          cross_indices=args.cross_indices)

    cross_scale_fusion = None
    if args.cross_scale_fusion:
        cross_scale_fusion = FPNFusionModule(backbone.num_channels, fuse_dim=args.reduced_dim, all=args.cross_all_out)

    deform_transformers = build_deforamble_transformer(args)

    model = Detector(
        backbone,
        deform_transformers,
        num_classes=num_classes,
        num_queries=args.det_token_num,
        # two essential techniques used in ViDT
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        # three additional techniques (optionally)
        cross_scale_fusion=cross_scale_fusion,
        iou_aware=args.iou_aware,
        token_label=args.token_label,
        # distil
        distil=False if args.distil_model is None else True,
    )

    return model
