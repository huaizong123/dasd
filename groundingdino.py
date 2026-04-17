# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import os
import sys
sys.path.append('/workspace/data/SpeechCLIP')
import copy
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss

from avssl.module.speech_encoder_plus import FairseqSpeechEncoder_Hubert
from avssl.module.projections import MLPLayers
from avssl.module.kw_modules.TransformerModels import TransformerEncoder

import torch
import torch.nn as nn

class AudioQFormer768(nn.Module):
    def __init__(self, num_queries=32, nheads=12, num_layers=2):
        super().__init__()
        self.hidden_dim = 768
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, self.hidden_dim))
        nn.init.normal_(self.query_embed, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=nheads, dim_feedforward=self.hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, audio_feat, audio_pad_mask):
        bs = audio_feat.shape[0]
        queries = self.query_embed.expand(bs, -1, -1)
        return self.transformer(tgt=queries, memory=audio_feat, memory_key_padding_mask=audio_pad_mask)

class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 512
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

#        # bert
#        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
#        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
#        self.bert.pooler.dense.weight.requires_grad_(False)
#        self.bert.pooler.dense.bias.requires_grad_(False)
#        self.bert = BertModelWarper(bert_model=self.bert)
#
#        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
#        nn.init.constant_(self.feat_map.bias.data, 0)
#        nn.init.xavier_uniform_(self.feat_map.weight.data)
#        # freeze
#
        # 🚨 絕對凍結機制：讓 BERT 當一個無情的解答機，不參與梯度更新
#        for param in self.bert.parameters():
#            param.requires_grad = False
#        for param in self.feat_map.parameters():
#            param.requires_grad = False
#        # special tokens
#        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        ## add HuBERT module
        self.speech_encoder = FairseqSpeechEncoder_Hubert(
            name="hubert_base",            # 使用 Base 模型
            pretrained=False,              # 設為 False，因為我們 train.py 會手動載入權重
            trainable=False,               # Stage 1 凍結語音大腦
            feat_select_idx="weighted_sum", # 🔑 致命關鍵：啟動 12 層加權融合！
            normalize_hiddenstates=True,    # 🔑 致命關鍵：啟動特徵正規化！
            normalize_type="s3prl"
        )

        # ==========================================
        # 🌟 究極版 Speech Projection (帶防護罩的雙層 MLP)
        # ==========================================
        self.audio_qformer = AudioQFormer768(num_queries=32)
        self.speech_proj = nn.Sequential(
            nn.LayerNorm(768), 
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, self.hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 🌟 正交初始化 (Orthogonal Initialization)
        # 在對比學習中，這能保證初始的矩陣不會把空間壓扁，最大化特徵的分離度！
        for layer in self.speech_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # 讓 global_audio 轉換為適合 Query 的隱藏維度
        # self.speech_prompt_proj = nn.Linear(256, self.hidden_dim)
        #nn.init.constant_(self.speech_prompt_proj.bias.data, 0)

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob) # 算出來大約是 -4.59
        self.class_bias = nn.Parameter(torch.ones(1) * bias_value)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def set_image_tensor(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self,'poss'):
            del self.poss 

    def set_image_features(self, features , poss):
        self.features = features
        self.poss = poss

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, samples, wavs, audio_mask=None, wav_lens=None, targets=None, **kwargs):
        # ==========================================
        # 1. 影像特徵提取 (訓練期：每次強制重新計算)
        # ==========================================
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            
        # 🌟 致命修正：拔掉 hasattr 快取，每次都重新呼叫 backbone！
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features): # 👈 注意這裡不是 self.features
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
            
        # 處理多尺度特徵 (Multi-scale Feature Levels)
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors) # 👈 注意這裡
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l) # 👈 注意這裡是 poss，不是 self.poss

        # ==========================================
        # 2. 🌟 語音特徵提取與 Q-Former 壓縮
        # ==========================================
        if wavs is None:
            raise ValueError("Audio inputs 'wavs' must be provided.")
        
        # A. HuBERT 提取特徵 (B, T, 768)
        encoded_audio, feat_len = self.speech_encoder(wav=wavs, wav_len=wav_lens if wav_lens is not None else [])
        
        # B. 長度截斷防護與 Mask 生成
        MAX_LEN = 500 
        T = encoded_audio.shape[1]
        if T > MAX_LEN:
            encoded_audio = encoded_audio[:, :MAX_LEN, :]
            feat_len = torch.clamp(feat_len, max=MAX_LEN)
            T = MAX_LEN

        device = encoded_audio.device
        seq_range = torch.arange(T, device=device).unsqueeze(0)
        audio_padding_mask = seq_range >= feat_len.unsqueeze(1) # (B, T)
        
        # C. Q-Former 壓縮與 MLP 降維
        # 送入 Q-Former: 輸出 (B, 32, 768)
        compressed_audio = self.audio_qformer(audio_feat=encoded_audio, audio_pad_mask=audio_padding_mask) 
        # 送入 MLP: 輸出 (B, 32, 256)
        speech_features = self.speech_proj(compressed_audio) 

        # ==========================================
        # 3. 🌟 世紀大騙局：偽裝成原版 BERT 輸出字典
        # ==========================================
        bs = speech_features.shape[0]
        NUM_Q = speech_features.shape[1] # 固定為 32
        
        # 原版 DINO 需要知道哪些 Token 是 Padding。
        # 我們的 32 個 Token 全都是黃金特徵，沒有 Padding，所以 Mask 全為 True
        speech_token_mask = torch.ones((bs, NUM_Q), dtype=torch.bool, device=device)
        # 原版 DINO 需要文字順序。我們直接給予 0~31 的流水號
        position_ids = torch.arange(NUM_Q, dtype=torch.long, device=device).unsqueeze(0).expand(bs, -1)
        # 原版 DINO 需要 Self-Attention Mask (用來防止偷看)。我們允許這 32 個 Token 互相交流，全開 True
        speech_self_attention_masks = torch.ones((bs, NUM_Q, NUM_Q), dtype=torch.bool, device=device)

        # 封裝成原版 DINO 認識的格式！
        speech_dict = {
            "encoded_text": speech_features,             # (B, 32, 256)
            "text_token_mask": speech_token_mask,        # (B, 32)
            "position_ids": position_ids,                # (B, 32)
            "text_self_attention_masks": speech_self_attention_masks, # (B, 32, 32)
        }

        # ==========================================
        # 4. 進入 Transformer 融合 (完全不動原版邏輯)
        # ==========================================
        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        
        # 把 speech_dict 傳給原本吃 text_dict 的位置
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, speech_dict
        )

        # ==========================================
        # 5. 預測 BBox 座標 (完全不動原版邏輯)
        # ==========================================
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # ==========================================
        # 6. 預測分類 (Contrastive Alignment) (完全不動原版邏輯)
        # ==========================================
        # 原版的 class_embed 會自動把 layer_hs (影像區塊特徵) 跟 speech_dict 裡的 32 個 Token 做點積計算！
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, speech_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        outputs_class = outputs_class + self.class_bias
        outputs_class = torch.clamp(outputs_class.float(), min=-50.0, max=50.0)

        # ==========================================
        # 7. 輸出與清理
        # ==========================================
        out = {
            "pred_logits": outputs_class[-1], 
            "pred_boxes": outputs_coord_list[-1],
            # 供後續分析或畫圖使用的特徵
            "audio_tokens": speech_features,  # (B, 32, 256)
            "image_dense_feat": srcs[0],      # (B, 256, H/8, W/8)
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        unset_image_tensor = kwargs.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.unset_image_tensor() 
            
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )

    return model

