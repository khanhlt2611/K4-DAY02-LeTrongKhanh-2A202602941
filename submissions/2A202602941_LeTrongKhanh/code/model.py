"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    TODO:
      - timm.create_model(name, pretrained=..., num_classes=num_classes, drop_rate=...)
        (timm tự thay head mới; head khởi tạo ngẫu nhiên)
      - nếu init == "frozen": gọi freeze_backbone(model)
      - ghi lại tên tag trọng số thực sự được tải (model.pretrained_cfg)
    """
    import timm
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    net = timm.create_model(SUGGESTED_BACKBONES.get(name, name),
                            pretrained=pretrained and init != "scratch",
                            num_classes=num_classes, drop_rate=drop_rate)
    net.lab_init = init
    net.lab_pretrained = pretrained and init != "scratch"
    if init == "frozen":
        freeze_backbone(net)
    return net


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    TODO:
      - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
      - lưu ý (GUIDE.md mục 3.2): backbone đóng băng thì BatchNorm cũng phải ở chế độ eval.
        Hãy nghĩ nơi nào trong train loop phải gọi lại model.train() mà vẫn giữ BN ở eval.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.get_classifier().parameters():
        p.requires_grad_(True)
    model.eval()
    model.get_classifier().train()
    model.lab_init = "frozen"


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    TODO:
      - bỏ qua tham số requires_grad == False
      - trả về list[dict] dạng {"params": [...], "lr": ..., "weight_decay": ...}
      - (trục E) mở rộng: LR theo tầng nếu bạn muốn thử
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = [dict(params=[], lr=lr_backbone, weight_decay=weight_decay),
              dict(params=[], lr=lr_backbone, weight_decay=0.0),
              dict(params=[], lr=lr_head, weight_decay=weight_decay)]
    for p in model.parameters():
        if p.requires_grad:
            index = 2 if id(p) in head_ids else (0 if p.ndim > 1 else 1)
            groups[index]["params"].append(p)
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng. TODO."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    TODO: dùng thư viện đếm (fvcore, ptflops, thop...) hoặc tự đếm bằng hook.
    Ghi rõ công cụ đã dùng; số có thể lệch vài phần trăm giữa các công cụ.
    """
    import copy
    import torch
    from torch.profiler import profile, ProfilerActivity
    # PyTorch profiler: conv/matmul FLOPs / 2 = MACs. Không gồm phép elementwise/norm.
    net = copy.deepcopy(model).cpu().float().eval()
    # Tách SDPA thành matmul để profiler đếm cả attention của transformer.
    for module in net.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    x = torch.zeros(1, 3, img_size, img_size)
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        net(x)
    return sum(event.flops for event in prof.key_averages()) / 2e9
