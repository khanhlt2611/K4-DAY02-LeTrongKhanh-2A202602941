"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm/lớp có `raise NotImplementedError`.
Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    TODO: tạo đúng loss, hoặc gọi các lớp bên dưới.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", .1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần weight tính từ train")
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"Loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    TODO: tự cài đặt hoặc dùng torch.nn.CrossEntropyLoss(label_smoothing=eps), rồi ghi rõ
    bạn đã chọn cách nào. Kiểm tra: eps = 0 phải cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, target):
        return self.loss(logits, target)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    TODO:
      - tính log_softmax, lấy p_t của lớp đúng, nhân (1 - p_t)^gamma, lấy trung bình batch
      - alpha: None hoặc vector trọng số theo lớp
    BẮT BUỘC viết một kiểm tra nhỏ: gamma = 0 phải cho đúng cross-entropy (sai số < 1e-6).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma phải >= 0")
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp

    TODO: trả về tensor độ dài 9. Chỉ dùng số liệu của train, không dùng val hay test.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.shape != (9,) or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("counts phải có 9 số dương hữu hạn")
    if not 0 <= beta < 1:
        raise ValueError("beta phải nằm trong [0, 1)")
    weights = 1 / counts if beta == 0 else (1 - beta) / (-torch.expm1(counts * np.log(beta)))
    return (weights / weights.mean()).float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]

    TODO: tự cài đặt. Kiểm tra bằng mắt: vẽ vài ảnh sau khi trộn và in lam.
    """
    if mode not in ("mixup", "cutmix") or alpha < 0:
        raise ValueError("mode/alpha không hợp lệ")
    perm = torch.randperm(len(x), device=x.device)
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    if mode == "mixup":
        result = lam * x + (1 - lam) * x[perm]
    else:
        h, w = x.shape[-2:]
        ratio = np.sqrt(1 - lam)
        ch, cw = int(h * ratio), int(w * ratio)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, cy - ch // 2), min(h, cy + (ch + 1) // 2)
        x1, x2 = max(0, cx - cw // 2), min(w, cx + (cw + 1) // 2)
        result = x.clone()
        result[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (h * w)
    return result, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    TODO. Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
