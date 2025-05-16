""" =================================================

# @Time: 2024/9/2 14:13

# @Author: Gringer

# @File: loss.py

# @Software: PyCharm

================================================== """
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss
    """

    def __init__(self,
                 num_labels,
                 gamma=2.0,
                 alpha=0.25,
                 epsilon=1.e-9,
                 reduction='mean',
                 activation_type='softmax'
                 ):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type
        self.reduction = reduction

    def forward(self, preds, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = F.softmax(preds, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = F.sigmoid(preds)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



class FocalCosineLoss(nn.Module):
    def __init__(self, eps=0.1, alpha=1, gamma=2.0, xent=0.1, cosine_weight=1.0, focal_weight=1.0,
                 reduction='mean', temperature=1.0, ignore_index=-100):
        super(FocalCosineLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction
        self.cosine_weight = cosine_weight
        self.focal_weight = focal_weight
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.cosine_target = torch.tensor([1.0], dtype=torch.float32)

    def forward(self, preds, target):
        # 将 cosine_target 移动到正确设备
        self.cosine_target = self.cosine_target.to(target.device)

        # 保证 alpha 在正确设备上
        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(target.device)

        # ======== 余弦损失部分 ========
        preds_norm = F.normalize(preds, p=2, dim=-1)  # 归一化预测
        cosine_loss = F.cosine_embedding_loss(
            preds_norm,
            torch.nn.functional.one_hot(target, num_classes=preds.size(-1)).float(),
            self.cosine_target,
            reduction=self.reduction,
        )

        # ======== 带温度缩放的交叉熵和标签平滑 ========
        preds_scaled = preds / self.temperature  # 使用温度缩放 logits
        n_classes = preds.size(-1)

        # 标签平滑处理
        if self.eps > 0:
            smoothed_labels = (1 - self.eps) * F.one_hot(target, n_classes).float() + self.eps / n_classes
            cent_loss = torch.sum(-smoothed_labels * F.log_softmax(preds_scaled, dim=-1), dim=-1)
        else:
            cent_loss = F.cross_entropy(preds_scaled, target, reduction="none", ignore_index=self.ignore_index)

        # ======== 焦点损失部分 ========
        pt = torch.exp(-cent_loss)  # 预测的概率
        pt = torch.clamp(pt, min=1e-10, max=1.0)  # 避免极端值
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[target]
        else:
            alpha_t = self.alpha

        focal_factor = (1 - pt) ** self.gamma
        focal_loss = alpha_t * focal_factor * cent_loss

        # 根据 reduction 聚合焦点损失
        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == "sum":
            focal_loss = torch.sum(focal_loss)

        # ======== 合并损失 ========
        total_loss = self.cosine_weight * cosine_loss + self.xent * self.focal_weight * focal_loss
        return total_loss
