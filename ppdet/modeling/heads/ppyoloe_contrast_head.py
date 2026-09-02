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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..initializer import bias_init_with_prob, constant_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.heads.ppyoloe_head import PPYOLOEHead

__all__ = ['PPYOLOEContrastHead']


@register
class PPYOLOEContrastHead(PPYOLOEHead):
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
            self.contrast_encoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
        self._init_contrast_encoder()

    def _init_contrast_encoder(self):
        bias_en = bias_init_with_prob(0.01)
        for en_ in self.contrast_encoder:
            constant_(en_.weight)
            constant_(en_.bias, bias_en)

    def forward_train(self, feats, targets, aux_pred=None):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        contrast_encoder_list = []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            contrast_logit = self.contrast_encoder[i](self.stem_cls[i](
                feat, avg_feat) + feat)
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

@register
class PPYOLOEContrastHead1(PPYOLOEHead):
    '''
    11-->33
    '''
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
            self.contrast_encoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
        self._init_contrast_encoder()
        self.pool = nn.MaxPool2D(kernel_size=3, stride=3)

    def _init_contrast_encoder(self):
        bias_en = bias_init_with_prob(0.01)
        for en_ in self.contrast_encoder:
            constant_(en_.weight)
            constant_(en_.bias, bias_en)

    def forward_eval(self, feats):
        feats = [F.interpolate(feat, scale_factor=[3, 3], mode='BILINEAR',) for feat in feats]
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w // 9
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            cls_logit = self.pool(cls_logit)
            reg_dist = self.pool(reg_dist)

            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(
                    reg_dist, axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)

            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        if self.use_shared_conv:
            reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        else:
            reg_dist_list = paddle.concat(reg_dist_list, axis=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward_train(self, feats, targets, aux_pred=None):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        feats = [F.interpolate(feat, scale_factor=[3, 3], mode='BILINEAR',) for feat in feats]

        cls_score_list, reg_distri_list = [], []
        contrast_encoder_list = []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            contrast_logit = self.contrast_encoder[i](self.stem_cls[i](
                feat, avg_feat) + feat)

            cls_logit = self.pool(cls_logit)
            reg_distri = self.pool(reg_distri)
            contrast_logit = self.pool(contrast_logit)

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


@register
class PPYOLOEContrastHead2(PPYOLOEHead):
    '''
    11-->33
    '''
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
            self.contrast_encoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
        self._init_contrast_encoder()
        self.pool = nn.AvgPool2D(kernel_size=3, stride=3)

    def _init_contrast_encoder(self):
        bias_en = bias_init_with_prob(0.01)
        for en_ in self.contrast_encoder:
            constant_(en_.weight)
            constant_(en_.bias, bias_en)

    def forward_eval(self, feats):
        feats = [F.interpolate(feat, scale_factor=[3, 3], mode='BILINEAR',) for feat in feats]
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w // 9
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            cls_logit = self.pool(cls_logit)
            reg_dist = self.pool(reg_dist)

            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(
                    reg_dist, axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)

            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        if self.use_shared_conv:
            reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        else:
            reg_dist_list = paddle.concat(reg_dist_list, axis=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward_train(self, feats, targets, aux_pred=None):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        feats = [F.interpolate(feat, scale_factor=[3, 3], mode='BILINEAR',) for feat in feats]

        cls_score_list, reg_distri_list = [], []
        contrast_encoder_list = []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            contrast_logit = self.contrast_encoder[i](self.stem_cls[i](
                feat, avg_feat) + feat)

            cls_logit = self.pool(cls_logit)
            reg_distri = self.pool(reg_distri)
            contrast_logit = self.pool(contrast_logit)

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

@register
class PPYOLOEContrastHead3(PPYOLOEHead):
    '''
    11-->33
    '''
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
            self.contrast_encoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
        self._init_contrast_encoder()
        self.pool31 = nn.MaxPool2D(kernel_size=(3, 1), stride=(3, 1))
        self.pool13 = nn.MaxPool2D(kernel_size=(1, 3), stride=(1, 3))

    def _init_contrast_encoder(self):
        bias_en = bias_init_with_prob(0.01)
        for en_ in self.contrast_encoder:
            constant_(en_.weight)
            constant_(en_.bias, bias_en)

    def forward_eval(self, feats):
        feats1 = [F.interpolate(feat, scale_factor=[3, 1], mode='BILINEAR',) for feat in feats]
        feats2 = [F.interpolate(feat, scale_factor=[1, 3], mode='BILINEAR', ) for feat in feats]

        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_score_list1, reg_dist_list1 = [], []
        for i, feat in enumerate(feats1):
            _, _, h, w = feat.shape
            l = h * w // 3
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            cls_logit = self.pool31(cls_logit)
            reg_dist = self.pool31(reg_dist)

            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(
                    reg_dist, axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)


            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list1.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list1.append(reg_dist)

        cls_score_list2, reg_dist_list2 = [], []
        for i, feat in enumerate(feats2):
            _, _, h, w = feat.shape
            l = h * w // 3
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            cls_logit = self.pool13(cls_logit)
            reg_dist = self.pool13(reg_dist)

            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(
                    reg_dist, axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)


            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list2.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list2.append(reg_dist)

        cls_score_list = [paddle.maximum(f1, f2) for f1, f2 in zip(cls_score_list1, cls_score_list2)]
        reg_dist_list = [paddle.maximum(f1, f2) for f1, f2 in zip(reg_dist_list1, reg_dist_list2)]

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        if self.use_shared_conv:
            reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        else:
            reg_dist_list = paddle.concat(reg_dist_list, axis=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward_train(self, feats, targets, aux_pred=None):
        feats1 = [F.interpolate(feat, scale_factor=[3, 1], mode='BILINEAR',) for feat in feats]
        feats2 = [F.interpolate(feat, scale_factor=[1, 3], mode='BILINEAR', ) for feat in feats]

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list1, reg_distri_list1, contrast_encoder_list1 = [], [], []
        for i, feat in enumerate(feats1):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            contrast_logit = self.contrast_encoder[i](self.stem_cls[i](
                feat, avg_feat) + feat)

            cls_logit = self.pool31(cls_logit)
            reg_distri = self.pool31(reg_distri)
            contrast_logit = self.pool31(contrast_logit)

            contrast_encoder_list1.append(
                contrast_logit.flatten(2).transpose([0, 2, 1]))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list1.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list1.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list1 = paddle.concat(cls_score_list1, axis=1)
        reg_distri_list1 = paddle.concat(reg_distri_list1, axis=1)
        contrast_encoder_list1 = paddle.concat(contrast_encoder_list1, axis=1)

        cls_score_list2, reg_distri_list2, contrast_encoder_list2 = [], [], []
        for i, feat in enumerate(feats2):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            contrast_logit = self.contrast_encoder[i](self.stem_cls[i](
                feat, avg_feat) + feat)

            cls_logit = self.pool13(cls_logit)
            reg_distri = self.pool13(reg_distri)
            contrast_logit = self.pool13(contrast_logit)

            contrast_encoder_list2.append(
                contrast_logit.flatten(2).transpose([0, 2, 1]))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list2.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list2.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list2 = paddle.concat(cls_score_list2, axis=1)
        reg_distri_list2 = paddle.concat(reg_distri_list2, axis=1)
        contrast_encoder_list2 = paddle.concat(contrast_encoder_list2, axis=1)

        cls_score_list = paddle.maximum(cls_score_list1, cls_score_list2)
        reg_distri_list = paddle.maximum(reg_distri_list1, reg_distri_list2)
        contrast_encoder_list = paddle.maximum(contrast_encoder_list1, contrast_encoder_list2)

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

@register
class PPYOLOEContrastHead4(PPYOLOEHead):
    '''
    11-->33
    '''
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
            self.contrast_encoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
        self._init_contrast_encoder()
        self.pool31 = nn.MaxPool2D(kernel_size=(3, 1), stride=(3, 1))
        self.pool13 = nn.MaxPool2D(kernel_size=(1, 3), stride=(1, 3))

    def _init_contrast_encoder(self):
        bias_en = bias_init_with_prob(0.01)
        for en_ in self.contrast_encoder:
            constant_(en_.weight)
            constant_(en_.bias, bias_en)

    def forward_eval(self, feats):
        feats1 = [F.interpolate(feat, scale_factor=[3, 1], mode='BILINEAR',) for feat in feats]
        feats2 = [F.interpolate(feat, scale_factor=[1, 3], mode='BILINEAR', ) for feat in feats]

        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_score_list1, reg_dist_list1 = [], []
        for i, feat in enumerate(feats1):
            _, _, h, w = feat.shape
            l = h * w // 3
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            cls_logit = self.pool31(cls_logit)
            reg_dist = self.pool31(reg_dist)

            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(
                    reg_dist, axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)


            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list1.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list1.append(reg_dist)

        cls_score_list2, reg_dist_list2 = [], []
        for i, feat in enumerate(feats2):
            _, _, h, w = feat.shape
            l = h * w // 3
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            cls_logit = self.pool13(cls_logit)
            reg_dist = self.pool13(reg_dist)

            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(
                    reg_dist, axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)


            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list2.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list2.append(reg_dist)

        cls_score_list = [(f1 + f2)/2 for f1, f2 in zip(cls_score_list1, cls_score_list2)]
        reg_dist_list = [(f1 + f2)/2 for f1, f2 in zip(reg_dist_list1, reg_dist_list2)]

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        if self.use_shared_conv:
            reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        else:
            reg_dist_list = paddle.concat(reg_dist_list, axis=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward_train(self, feats, targets, aux_pred=None):
        feats1 = [F.interpolate(feat, scale_factor=[3, 1], mode='BILINEAR',) for feat in feats]
        feats2 = [F.interpolate(feat, scale_factor=[1, 3], mode='BILINEAR', ) for feat in feats]

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list1, reg_distri_list1, contrast_encoder_list1 = [], [], []
        for i, feat in enumerate(feats1):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            contrast_logit = self.contrast_encoder[i](self.stem_cls[i](
                feat, avg_feat) + feat)

            cls_logit = self.pool31(cls_logit)
            reg_distri = self.pool31(reg_distri)
            contrast_logit = self.pool31(contrast_logit)

            contrast_encoder_list1.append(
                contrast_logit.flatten(2).transpose([0, 2, 1]))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list1.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list1.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list1 = paddle.concat(cls_score_list1, axis=1)
        reg_distri_list1 = paddle.concat(reg_distri_list1, axis=1)
        contrast_encoder_list1 = paddle.concat(contrast_encoder_list1, axis=1)

        cls_score_list2, reg_distri_list2, contrast_encoder_list2 = [], [], []
        for i, feat in enumerate(feats2):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            contrast_logit = self.contrast_encoder[i](self.stem_cls[i](
                feat, avg_feat) + feat)

            cls_logit = self.pool13(cls_logit)
            reg_distri = self.pool13(reg_distri)
            contrast_logit = self.pool13(contrast_logit)

            contrast_encoder_list2.append(
                contrast_logit.flatten(2).transpose([0, 2, 1]))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list2.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list2.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list2 = paddle.concat(cls_score_list2, axis=1)
        reg_distri_list2 = paddle.concat(reg_distri_list2, axis=1)
        contrast_encoder_list2 = paddle.concat(contrast_encoder_list2, axis=1)

        cls_score_list = (cls_score_list1 + cls_score_list2)/2
        reg_distri_list = (reg_distri_list1 + reg_distri_list2)/2
        contrast_encoder_list = (contrast_encoder_list1 + contrast_encoder_list2)/2

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
