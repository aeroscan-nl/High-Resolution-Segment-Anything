import warnings

import torch
import torch.nn as nn

from mmseg.models.builder import LOSSES


@LOSSES.register_module()
class NormalizedSoftFocalLoss(nn.Module):

    def __init__(self,
                 max_label_value,
                 alpha=0.5,
                 gamma=2.0,
                 eps=torch.finfo(torch.float).eps,
                 use_sigmoid=True,
                 stop_delimiter_grad=True,
                 loss_weight=1.0,
                 ignore_index=None,
                 loss_name='nsfl_loss'):
        if ignore_index is not None:
            warnings.warn('ignore_index is invalid in soft version')
        super(NormalizedSoftFocalLoss, self).__init__()
        self.max_label_value = max_label_value
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.use_sigmoid = use_sigmoid
        self.stop_delimiter_grad = stop_delimiter_grad
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, seg_logit, seg_label, **kwargs):
        if seg_label.dim() == 3:
            seg_label = seg_label.unsqueeze(1)
        assert seg_logit.dim() == seg_label.dim() == 4
        assert seg_logit.size(0) == seg_label.size(0)
        assert seg_logit.size()[-2:] == seg_label.size()[-2:]

        def format_logit(x, dim):
            if x.dim() == dim + 1:
                if x.size(1) == 1:
                    x = x.squeeze(1)
                elif x.size(1) == 2:
                    x = x[:, 1] - x[:, 0]
            assert x.dim() == dim, \
                f'cannot format x with shape: {x.size()} ' \
                f'to match the dim {dim}'
            return x

        if seg_label.dim() == 2:
            seg_logit = format_logit(seg_logit, 2)
        elif seg_label.dim() in [3, 4]:
            seg_logit = format_logit(seg_logit, 3)
            if seg_label.dim() == 4:
                assert seg_label.size(1) == 1, \
                    f'cannot handle seg_label with shape: {seg_label.size()}'
                seg_label = seg_label.squeeze(1)
        else:
            raise NotImplementedError(
                f'cannot handle seg_label with shape: {seg_label.size()}')

        pixel_pred = torch.sigmoid(seg_logit) \
            if self.use_sigmoid else seg_logit
        pixel_weight = torch.ones_like(pixel_pred)
        pixel_pred_min = pixel_pred.min().item()
        pixel_pred_max = pixel_pred.max().item()
        if pixel_pred_min < 0 or pixel_pred_max > 1:
            raise ValueError(f'pixel_pred out of range [0, 1], got '
                             f'[{pixel_pred_min}, {pixel_pred_max}]')
        seg_label = (seg_label.float() / self.max_label_value).to(pixel_pred)

        fg_pt = pixel_pred
        bg_pt = 1.0 - pixel_pred
        fg_beta = (1.0 - fg_pt) ** self.gamma * seg_label
        bg_beta = (1.0 - bg_pt) ** self.gamma * (1 - seg_label)
        scale = pixel_weight.sum(dim=(-2, -1), keepdim=True) / \
                (fg_beta.sum(dim=(-2, -1), keepdim=True) +
                 bg_beta.sum(dim=(-2, -1), keepdim=True) + self.eps)
        if self.stop_delimiter_grad:
            scale = scale.detach()
        fg_beta = scale * fg_beta
        bg_beta = scale * bg_beta

        pixel_loss = -self.alpha * fg_beta * \
                     (fg_pt + self.eps).clip(None, 1.0).log() \
                     - (1.0 - self.alpha) * bg_beta * \
                     (bg_pt + self.eps).clip(None, 1.0).log()
        pixel_loss = self.loss_weight * pixel_weight * pixel_loss

        return pixel_loss
