# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..initializer import bias_init_with_prob, constant_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.heads.ppyoloe_head import PPYOLOEHead, ESEAttn


__all__ = ['PPYOLOEPromptHead']


class PromptContrast(nn.Layer):
    __shared__ = [
        'num_classes'
    ]

    def __init__(self, num_classes=2, temperature=100, sample_num=2048, thresh=0.75):
        super(PromptContrast, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.sample_num = sample_num
        self.thresh = thresh
        self.prompt_fc = None

    def forward(self, features, labels, scores):
        assert features.shape[0] == labels.shape[0] == scores.shape[0]
        positive_mask = (labels < self.num_classes)
        positive_features, positive_labels, positive_scores = features[positive_mask], labels[positive_mask], \
            scores[positive_mask]

        negative_mask = (labels == self.num_classes)
        negative_features, negative_labels, negative_scores = features[negative_mask], labels[negative_mask], \
            scores[negative_mask]

        N = negative_features.shape[0]
        S = self.sample_num - positive_mask.sum()
        index = paddle.to_tensor(random.sample(range(N), int(S)), dtype='int32')

        negative_features = paddle.index_select(x=negative_features, index=index, axis=0)
        negative_labels = paddle.index_select(x=negative_labels, index=index, axis=0)
        negative_scores = paddle.index_select(x=negative_scores, index=index, axis=0)

        features = paddle.concat([positive_features, negative_features], 0)
        labels = paddle.concat([positive_labels, negative_labels], 0)
        scores = paddle.concat([positive_scores, negative_scores], 0)

        if len(labels.shape) == 1:
            labels = labels.reshape([-1, 1])
        label_mask = paddle.equal(labels, labels.T).detach()
        similarity = (paddle.matmul(features, features.T) / self.temperature)

        sim_row_max = paddle.max(similarity, axis=1, keepdim=True)
        similarity = similarity - sim_row_max

        logits_mask = paddle.ones_like(similarity).detach()
        logits_mask.fill_diagonal_(0)

        exp_sim = paddle.exp(similarity) * logits_mask
        log_prob = similarity - paddle.log(exp_sim.sum(axis=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)
        keep = scores > self.thresh
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        return loss.mean()


class CrossAttention(nn.Layer):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        attn_weights = (x1.transpose((0, 2, 1)) @ x2) / (self.dim ** 0.5)
        attn_weights = F.softmax(attn_weights, axis=-1)
        return attn_weights


class PromptHead(nn.Layer):
    __shared__ = [
        'num_classes'
    ]

    def __init__(self, in_chs, num_classes=7, ):
        super(PromptHead, self).__init__()
        self.num_classes = num_classes
        self.prompt_fc = None
        self.att = CrossAttention(in_chs)


    def forward(self, features, labels, scores):
        n, c, h, w = features.shape
        features_flat = features.reshape((n, c, -1))
        features_prompt = features_flat[:1]
        features_search = features_flat
        att_weight = self.att(features_search, features_prompt)
        return att_weight


@register
class PPYOLOEPromptHead1(PPYOLOEHead):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv', 'for_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms', 'contrast_loss']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 contrast_loss='SupContrast',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 attn_conv='convbn',
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True,
                 for_distill=False):
        super().__init__(in_channels, num_classes, act, fpn_strides,
                         grid_cell_scale, grid_cell_offset, reg_max, reg_range,
                         static_assigner_epoch, use_varifocal_loss,
                         static_assigner, assigner, nms, eval_size, loss_weight,
                         trt, attn_conv, exclude_nms, exclude_post_process,
                         use_shared_conv, for_distill)

        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.contrast_loss = contrast_loss
        self.contrast_encoder = nn.LayerList()
        for in_c in self.in_channels:
            self.contrast_encoder.append(nn.Conv2D(in_c, in_c, 3, padding=1))
        self.contrast_decoder = nn.LayerList()
        for in_c in self.in_channels:
            self.contrast_decoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
        self.prompt_head = nn.LayerList()

        # stem
        self.stem_cls = nn.LayerList()

        for in_c in self.in_channels:
            self.prompt_head.append(PromptHead(in_c, self.num_classes))
            self.stem_cls.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
        self._init_contrast_encoder()



    def _init_contrast_encoder(self):
        bias_en = bias_init_with_prob(0.01)
        for en_ in self.contrast_encoder:
            constant_(en_.weight)
            constant_(en_.bias, bias_en)


    def _filterweight(self, weights, labels, weight_shape):
        labels[labels>=self.num_classes] = 0
        label_onehot = F.one_hot(labels, num_classes=self.num_classes)
        label_onehot_rep = label_onehot.transpose((0, 2, 1)).unsqueeze(-1)
        weights_rep = weights.unsqueeze(1).repeat_interleave(repeats=self.num_classes, axis=1)
        weights_filter = weights_rep*label_onehot_rep
        # n, c, hw(TF), hw
        weights_filter_rec = paddle.sum(weights_filter, axis=2).reshape(weight_shape)
        return weights_filter_rec

    def forward_train(self, feats, targets, aux_pred=None):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        contrast_encoder_list = []
        for i, feat in enumerate(feats):
            n, c, h, w = feat.shape
            num_anchors_list_i = [num_anchors_list[i]]
            anchors_i = anchors[sum(num_anchors_list[:i]):sum(num_anchors_list[:i])+num_anchors_list_i[0]]
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors_i,
                    num_anchors_list_i,
                    targets['gt_class'],
                    targets['gt_bbox'],
                    targets['pad_gt_mask'],
                    bg_index=self.num_classes,
                    pred_bboxes=None)

            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            stem_feat = self.stem_cls[i](feat, avg_feat)
            prompt_weight = self.prompt_head[i](stem_feat, assigned_labels, assigned_scores)
            prompt_weight = self._filterweight(prompt_weight, assigned_labels, weight_shape=(n, self.num_classes, h, w))

            cls_logit = self.pred_cls[i](stem_feat + feat)
            cls_logit = cls_logit*prompt_weight + cls_logit
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            contrast_logit = self.contrast_encoder[i](stem_feat + feat)
            contrast_encoder_list.append(
                contrast_logit.flatten(2).transpose([0, 2, 1]))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        contrast_encoder_list = paddle.concat(contrast_encoder_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, contrast_encoder_list, anchors,
            anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, pred_contrast_encoder, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            if self.sm_use:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    stride_tensor,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            else:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)
        # contrast loss
        loss_contrast = self.contrast_loss(pred_contrast_encoder.reshape([-1, pred_contrast_encoder.shape[-1]]), \
            assigned_labels.reshape([-1]), assigned_scores.max(-1).reshape([-1]))

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['contrast'] * loss_contrast

        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
            'loss_contrast': loss_contrast
        }
        return out_dict

class PPYOLOEPromptHead2(PPYOLOEHead):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv', 'for_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms', 'contrast_loss']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 contrast_loss='SupContrast',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 attn_conv='convbn',
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True,
                 for_distill=False):
        super().__init__(in_channels, num_classes, act, fpn_strides,
                         grid_cell_scale, grid_cell_offset, reg_max, reg_range,
                         static_assigner_epoch, use_varifocal_loss,
                         static_assigner, assigner, nms, eval_size, loss_weight,
                         trt, attn_conv, exclude_nms, exclude_post_process,
                         use_shared_conv, for_distill)

        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.contrast_loss = contrast_loss
        self.contrast_encoder = nn.LayerList()
        for in_c in self.in_channels:
            self.contrast_encoder.append(nn.Conv2D(in_c, in_c, 3, padding=1))
        self.contrast_decoder = nn.LayerList()
        for in_c in self.in_channels:
            self.contrast_decoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
        self.prompt_head = nn.LayerList()

        # stem
        self.stem_cls = nn.LayerList()

        for in_c in self.in_channels:
            self.prompt_head.append(PromptHead(in_c, self.num_classes))
            self.stem_cls.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
        self._init_contrast_encoder()



    def _init_contrast_encoder(self):
        bias_en = bias_init_with_prob(0.01)
        for en_ in self.contrast_encoder:
            constant_(en_.weight)
            constant_(en_.bias, bias_en)


    def _filterweight(self, weights, labels, weight_shape):
        labels[labels>=self.num_classes] = 0
        label_onehot = F.one_hot(labels, num_classes=self.num_classes)
        label_onehot_rep = label_onehot.transpose((0, 2, 1)).unsqueeze(-1)
        weights_rep = weights.unsqueeze(1).repeat_interleave(repeats=self.num_classes, axis=1)
        weights_filter = weights_rep*label_onehot_rep
        # n, c, hw(TF), hw
        weights_filter_rec = paddle.sum(weights_filter, axis=2).reshape(weight_shape)
        return weights_filter_rec

    def forward_train(self, feats, targets, aux_pred=None):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        contrast_encoder_list = []
        for i, feat in enumerate(feats):
            n, c, h, w = feat.shape
            num_anchors_list_i = [num_anchors_list[i]]
            anchors_i = anchors[sum(num_anchors_list[:i]):sum(num_anchors_list[:i])+num_anchors_list_i[0]]
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors_i,
                    num_anchors_list_i,
                    targets['gt_class'],
                    targets['gt_bbox'],
                    targets['pad_gt_mask'],
                    bg_index=self.num_classes,
                    pred_bboxes=None)

            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            stem_feat = self.stem_cls[i](feat, avg_feat)
            prompt_weight = self.prompt_head[i](stem_feat, assigned_labels, assigned_scores)
            prompt_weight = self._filterweight(prompt_weight, assigned_labels, weight_shape=(n, self.num_classes, h, w))

            cls_logit = self.pred_cls[i](stem_feat + feat)
            cls_logit = cls_logit*prompt_weight + cls_logit
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            contrast_logit = self.contrast_encoder[i](stem_feat + feat)
            contrast_encoder_list.append(
                contrast_logit.flatten(2).transpose([0, 2, 1]))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        contrast_encoder_list = paddle.concat(contrast_encoder_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, contrast_encoder_list, anchors,
            anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, pred_contrast_encoder, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            if self.sm_use:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    stride_tensor,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            else:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)
        # contrast loss
        loss_contrast = self.contrast_loss(pred_contrast_encoder.reshape([-1, pred_contrast_encoder.shape[-1]]), \
            assigned_labels.reshape([-1]), assigned_scores.max(-1).reshape([-1]))

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['contrast'] * loss_contrast

        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
            'loss_contrast': loss_contrast
        }
        return out_dict


class PPYOLOEPromptConstrastHead(PPYOLOEHead):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv', 'for_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms', 'contrast_loss']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 contrast_loss='SupContrast',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 attn_conv='convbn',
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True,
                 for_distill=False):
        super().__init__(in_channels, num_classes, act, fpn_strides,
                         grid_cell_scale, grid_cell_offset, reg_max, reg_range,
                         static_assigner_epoch, use_varifocal_loss,
                         static_assigner, assigner, nms, eval_size, loss_weight,
                         trt, attn_conv, exclude_nms, exclude_post_process,
                         use_shared_conv, for_distill)

        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.contrast_loss = contrast_loss
        self.contrast_encoder = nn.LayerList()
        for in_c in self.in_channels:
            self.contrast_encoder.append(nn.Conv2D(in_c, in_c, 3, padding=1))
        self.contrast_decoder = nn.LayerList()
        for in_c in self.in_channels:
            self.contrast_decoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
        self.prompt_head = nn.LayerList()

        # stem
        self.stem_cls = nn.LayerList()

        for in_c in self.in_channels:
            self.prompt_head.append(PromptHead(in_c, self.num_classes))
            self.stem_cls.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
        self._init_contrast_encoder()



    def _init_contrast_encoder(self):
        bias_en = bias_init_with_prob(0.01)
        for en_ in self.contrast_encoder:
            constant_(en_.weight)
            constant_(en_.bias, bias_en)


    def _filterweight(self, weights, labels, weight_shape):
        labels[labels>=self.num_classes] = 0
        label_onehot = F.one_hot(labels, num_classes=self.num_classes)
        label_onehot_rep = label_onehot.transpose((0, 2, 1)).unsqueeze(-1)
        weights_rep = weights.unsqueeze(1).repeat_interleave(repeats=self.num_classes, axis=1)
        weights_filter = weights_rep*label_onehot_rep
        # n, c, hw(TF), hw
        weights_filter_rec = paddle.sum(weights_filter, axis=2).reshape(weight_shape)
        return weights_filter_rec

    def forward_train(self, feats, targets, aux_pred=None):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        contrast_encoder_list = []
        for i, feat in enumerate(feats):
            n, c, h, w = feat.shape
            num_anchors_list_i = [num_anchors_list[i]]
            anchors_i = anchors[sum(num_anchors_list[:i]):sum(num_anchors_list[:i])+num_anchors_list_i[0]]
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors_i,
                    num_anchors_list_i,
                    targets['gt_class'],
                    targets['gt_bbox'],
                    targets['pad_gt_mask'],
                    bg_index=self.num_classes,
                    pred_bboxes=None)

            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            stem_feat = self.stem_cls[i](feat, avg_feat)
            prompt_weight = self.prompt_head[i](stem_feat, assigned_labels, assigned_scores)
            prompt_weight = self._filterweight(prompt_weight, assigned_labels, weight_shape=(n, self.num_classes, h, w))

            cls_logit = self.pred_cls[i](stem_feat + feat)
            cls_logit = cls_logit*prompt_weight + cls_logit
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            contrast_logit = self.contrast_encoder[i](stem_feat + feat)
            contrast_encoder_list.append(
                contrast_logit.flatten(2).transpose([0, 2, 1]))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        contrast_encoder_list = paddle.concat(contrast_encoder_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, contrast_encoder_list, anchors,
            anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, pred_contrast_encoder, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            if self.sm_use:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    stride_tensor,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            else:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)
        # contrast loss
        loss_contrast = self.contrast_loss(pred_contrast_encoder.reshape([-1, pred_contrast_encoder.shape[-1]]), \
            assigned_labels.reshape([-1]), assigned_scores.max(-1).reshape([-1]))

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['contrast'] * loss_contrast

        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
            'loss_contrast': loss_contrast
        }
        return out_dict


# @register
# class PPYOLOEPromptHead(PPYOLOEHead):
#     __shared__ = [
#         'num_classes', 'eval_size', 'trt', 'exclude_nms',
#         'exclude_post_process', 'use_shared_conv', 'for_distill'
#     ]
#     __inject__ = ['static_assigner', 'assigner', 'nms', 'contrast_loss']
#
#     def __init__(self,
#                  in_channels=[1024, 512, 256],
#                  num_classes=80,
#                  act='swish',
#                  fpn_strides=(32, 16, 8),
#                  grid_cell_scale=5.0,
#                  grid_cell_offset=0.5,
#                  reg_max=16,
#                  reg_range=None,
#                  static_assigner_epoch=4,
#                  use_varifocal_loss=True,
#                  static_assigner='ATSSAssigner',
#                  assigner='TaskAlignedAssigner',
#                  contrast_loss='SupContrast',
#                  nms='MultiClassNMS',
#                  eval_size=None,
#                  loss_weight={
#                      'class': 1.0,
#                      'iou': 2.5,
#                      'dfl': 0.5,
#                  },
#                  trt=False,
#                  attn_conv='convbn',
#                  exclude_nms=False,
#                  exclude_post_process=False,
#                  use_shared_conv=True,
#                  for_distill=False):
#         super().__init__(in_channels, num_classes, act, fpn_strides,
#                          grid_cell_scale, grid_cell_offset, reg_max, reg_range,
#                          static_assigner_epoch, use_varifocal_loss,
#                          static_assigner, assigner, nms, eval_size, loss_weight,
#                          trt, attn_conv, exclude_nms, exclude_post_process,
#                          use_shared_conv, for_distill)
#
#         assert len(in_channels) > 0, "len(in_channels) should > 0"
#         self.contrast_loss = contrast_loss
#         self.contrast_encoder = nn.LayerList()
#         for in_c in self.in_channels:
#             self.contrast_encoder.append(nn.Conv2D(in_c, in_c, 3, padding=1))
#         self.contrast_decoder = nn.LayerList()
#         for in_c in self.in_channels:
#             self.contrast_decoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
#         self.prompt_head = nn.LayerList()
#
#         # stem
#         self.stem_cls = nn.LayerList()
#
#         for in_c in self.in_channels:
#             self.prompt_head.append(PromptHead(in_c, self.num_classes))
#             self.stem_cls.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
#         self._init_contrast_encoder()
#
#
#
#     def _init_contrast_encoder(self):
#         bias_en = bias_init_with_prob(0.01)
#         for en_ in self.contrast_encoder:
#             constant_(en_.weight)
#             constant_(en_.bias, bias_en)
#
#     def _attweight(self, feature, weight):
#         n, c, h, w = feature.shape
#         feature_flat = feature.reshape((n, c, -1))
#         feature_fuse = paddle.matmul(feature_flat, weight)
#         feature_fuse = feature_fuse.reshape((n, c, h, w))
#         return feature_fuse
#
#     def forward_train(self, feats, targets, aux_pred=None):
#         anchors, anchor_points, num_anchors_list, stride_tensor = \
#             generate_anchors_for_grid_cell(
#                 feats, self.fpn_strides, self.grid_cell_scale,
#                 self.grid_cell_offset)
#
#         cls_score_list, reg_distri_list = [], []
#         contrast_encoder_list = []
#         prompt_weights = []
#         for i, feat in enumerate(feats):
#             n, c, h, w = feat.shape
#             num_anchors_list_i = [num_anchors_list[i]]
#             anchors_i = anchors[sum(num_anchors_list[:i]):sum(num_anchors_list[:i])+num_anchors_list_i[0]]
#             avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
#
#             contrast_encoder_feature = self.contrast_encoder[i](self.stem_cls[i](
#                 feat, avg_feat))
#             contrast_logit = contrast_encoder_feature.flatten(2).transpose([0, 2, 1])
#
#
#             assigned_labels, assigned_bboxes, assigned_scores = \
#                 self.static_assigner(
#                     anchors_i,
#                     num_anchors_list_i,
#                     targets['gt_class'],
#                     targets['gt_bbox'],
#                     targets['pad_gt_mask'],
#                     bg_index=self.num_classes,
#                     pred_bboxes=None)
#
#             prompt_weight = self.prompt_head[i](contrast_logit, assigned_labels, assigned_scores)
#             prompt_weights.append(prompt_weight)
#
#             contrast_logit = self.contrast_decoder[i](contrast_encoder_feature+feat)
#             contrast_logit = contrast_logit.flatten(2).transpose([0, 2, 1])
#             contrast_encoder_list.append(contrast_logit)
#
#
#         for i, feat in enumerate(feats):
#             avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
#             cls_emb = self.stem_cls[i](feat, avg_feat)
#             cls_emb = self._attweight(cls_emb, prompt_weights[i]) + cls_emb
#             cls_logit = self.pred_cls[i](cls_emb + feat)
#             reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
#             # cls and reg
#             cls_score = F.sigmoid(cls_logit)
#             cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
#             reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
#         cls_score_list = paddle.concat(cls_score_list, axis=1)
#         reg_distri_list = paddle.concat(reg_distri_list, axis=1)
#         contrast_encoder_list = paddle.concat(contrast_encoder_list, axis=1)
#
#         return self.get_loss([
#             cls_score_list, reg_distri_list, contrast_encoder_list, anchors,
#             anchor_points, num_anchors_list, stride_tensor
#         ], targets)
#
#     def get_loss(self, head_outs, gt_meta):
#         pred_scores, pred_distri, pred_contrast_encoder, anchors,\
#         anchor_points, num_anchors_list, stride_tensor = head_outs
#
#         anchor_points_s = anchor_points / stride_tensor
#         pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)
#
#         gt_labels = gt_meta['gt_class']
#         gt_bboxes = gt_meta['gt_bbox']
#         pad_gt_mask = gt_meta['pad_gt_mask']
#         # label assignment
#         if gt_meta['epoch_id'] < self.static_assigner_epoch:
#             assigned_labels, assigned_bboxes, assigned_scores = \
#                 self.static_assigner(
#                     anchors,
#                     num_anchors_list,
#                     gt_labels,
#                     gt_bboxes,
#                     pad_gt_mask,
#                     bg_index=self.num_classes,
#                     pred_bboxes=pred_bboxes.detach() * stride_tensor)
#             alpha_l = 0.25
#         else:
#             if self.sm_use:
#                 assigned_labels, assigned_bboxes, assigned_scores = \
#                     self.assigner(
#                     pred_scores.detach(),
#                     pred_bboxes.detach() * stride_tensor,
#                     anchor_points,
#                     stride_tensor,
#                     gt_labels,
#                     gt_bboxes,
#                     pad_gt_mask,
#                     bg_index=self.num_classes)
#             else:
#                 assigned_labels, assigned_bboxes, assigned_scores = \
#                     self.assigner(
#                     pred_scores.detach(),
#                     pred_bboxes.detach() * stride_tensor,
#                     anchor_points,
#                     num_anchors_list,
#                     gt_labels,
#                     gt_bboxes,
#                     pad_gt_mask,
#                     bg_index=self.num_classes)
#             alpha_l = -1
#         # rescale bbox
#         assigned_bboxes /= stride_tensor
#         # cls loss
#         if self.use_varifocal_loss:
#             one_hot_label = F.one_hot(assigned_labels,
#                                       self.num_classes + 1)[..., :-1]
#             loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
#                                             one_hot_label)
#         else:
#             loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)
#
#         assigned_scores_sum = assigned_scores.sum()
#         if paddle.distributed.get_world_size() > 1:
#             paddle.distributed.all_reduce(assigned_scores_sum)
#             assigned_scores_sum /= paddle.distributed.get_world_size()
#         assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
#         loss_cls /= assigned_scores_sum
#
#         loss_l1, loss_iou, loss_dfl = \
#             self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
#                             assigned_labels, assigned_bboxes, assigned_scores,
#                             assigned_scores_sum)
#         # contrast loss
#         loss_contrast = self.contrast_loss(pred_contrast_encoder.reshape([-1, pred_contrast_encoder.shape[-1]]), \
#             assigned_labels.reshape([-1]), assigned_scores.max(-1).reshape([-1]))
#
#         loss = self.loss_weight['class'] * loss_cls + \
#                self.loss_weight['iou'] * loss_iou + \
#                self.loss_weight['dfl'] * loss_dfl + \
#                self.loss_weight['contrast'] * loss_contrast
#
#         out_dict = {
#             'loss': loss,
#             'loss_cls': loss_cls,
#             'loss_iou': loss_iou,
#             'loss_dfl': loss_dfl,
#             'loss_l1': loss_l1,
#             'loss_contrast': loss_contrast
#         }
#         return out_dict
