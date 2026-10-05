"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    TODO: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    model.eval()
    names, labels, logits = [], [], []
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device, non_blocking=True)
            if view is not None:
                x = view(x)
            output = model(x)
            names.extend(filenames)
            labels.append(y.cpu().numpy())
            logits.append(output.float().cpu().numpy())
    if not logits:
        raise ValueError("Loader rỗng")
    return names, np.concatenate(labels), np.concatenate(logits)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). TODO: dùng torch.flip trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. TODO."""
    h, w = x.shape[-2:]
    if not 0 < crop <= min(h, w):
        raise ValueError("crop phải > 0 và không vượt kích thước ảnh")
    positions = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop),
                 ((h - crop) // 2, (w - crop) // 2)]
    return [x[..., top:top + crop, left:left + crop] for top, left in positions]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. TODO.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    if not sizes or any(int(size) <= 0 for size in sizes):
        raise ValueError("sizes phải là danh sách kích thước dương")
    return [F.interpolate(x, size=(int(size), int(size)), mode="bicubic",
                          align_corners=False, antialias=True) for size in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    TODO: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    if len(logits_per_view) == 0:
        raise ValueError("Cần ít nhất một view")
    values = [np.asarray(z, dtype=np.float64) for z in logits_per_view]
    if any(z.shape != values[0].shape for z in values):
        raise ValueError("Các view phải có cùng shape và thứ tự ảnh")
    if space == "prob":
        return ensemble_probs([apply_temperature(z, 1.0) for z in values])
    if space == "logit":
        return apply_temperature(np.mean(values, axis=0), 1.0)
    raise ValueError("space phải là prob hoặc logit")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). TODO.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    if len(list_of_probs) == 0:
        raise ValueError("Cần ít nhất một mô hình")
    arrays = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    for p in arrays:
        if p.shape != arrays[0].shape or p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Xác suất không hợp lệ hoặc shape khác nhau")
        if not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError("Xác suất chưa chuẩn hoá")
    probs = np.mean(arrays, axis=0)
    return probs / probs.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    TODO: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels)
    if z.ndim != 2 or len(z) == 0 or labels.shape != (len(z),):
        raise ValueError("val_logits/val_labels không hợp lệ")
    if not np.isfinite(z).all() or not np.isin(labels, range(z.shape[1])).all():
        raise ValueError("Logit/nhãn không hợp lệ")
    labels = labels.astype(int)
    # Tìm lưới trên log(T), sau đó tinh dần; NLL tính bằng log-sum-exp ổn định.
    def nll(log_t):
        scaled = z / np.exp(log_t)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        return float((np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(z)), labels]).mean())
    left, right = -5.0, 5.0
    best = 0.0
    for _ in range(5):
        grid = np.linspace(left, right, 101)
        idx = int(np.argmin([nll(v) for v in grid]))
        best = grid[idx]
        left, right = grid[max(0, idx - 1)], grid[min(100, idx + 1)]
    return float(np.exp(best))


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). TODO."""
    z = np.asarray(logits, dtype=np.float64)
    if T <= 0 or not np.isfinite(T) or z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("Cần logits hữu hạn dạng (N,K) và T > 0 hữu hạn")
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z)
    return probs / probs.sum(axis=1, keepdims=True)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    TODO:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    # Chỉ fuse cặp có quan hệ forward đã biết; không suy đoán từ thứ tự register module.
    net = copy.deepcopy(model).eval()
    def fuse(parent, conv_name, bn_name):
        conv, bn = getattr(parent, conv_name), getattr(parent, bn_name)
        if isinstance(conv, nn.Conv2d) and type(bn) is nn.BatchNorm2d:
            setattr(parent, conv_name, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
            setattr(parent, bn_name, nn.Identity())
    def visit(module):
        for child in list(module.children()):
            visit(child)
        if isinstance(module, nn.Sequential):
            keys = list(module._modules)
            for first, second in zip(keys, keys[1:]):
                fuse(module, first, second)
        # ResNet/ResNeXt chuẩn của timm; ConvNormAct giữ forward conv -> bn.
        if type(module).__module__.startswith("timm.models.resnet"):
            for c, b in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
                if hasattr(module, c) and hasattr(module, b):
                    fuse(module, c, b)
        if type(module).__name__ == "ConvNormAct" and type(module).__module__.startswith("timm"):
            if hasattr(module, "conv") and hasattr(module, "bn"):
                fuse(module, "conv", "bn")
    visit(net)
    net.lab_bn_fused = any(isinstance(m, nn.BatchNorm2d) for m in model.modules()) and \
                       sum(isinstance(m, nn.BatchNorm2d) for m in net.modules()) < \
                       sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    return net
