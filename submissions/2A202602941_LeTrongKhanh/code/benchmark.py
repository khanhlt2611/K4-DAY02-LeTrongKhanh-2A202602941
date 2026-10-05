"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations
import copy
import time
import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    if warmup < 10 or iters < 50:
        raise ValueError("Cần warmup >= 10 và iters >= 50")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append((time.perf_counter() - start) * 1000)
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99),
            "mean": float(np.mean(times)), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError("dtype phải là fp32/amp/fp16")
    device = torch.device(device)
    if device.type != "cuda" and dtype != "fp32":
        raise ValueError("Benchmark AMP/FP16 này cần CUDA")
    net = copy.deepcopy(model).to(device).eval()
    net.half() if dtype == "fp16" else net.float()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=dtype == "amp"):
            return net(x)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    result = bench(forward, warmup, iters, sync)
    return {**result, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "images_per_s": batch_size / (result["p50"] / 1000), "torch": str(torch.__version__),
            "preprocessing": False, "warmup": warmup,
            "bn_fused": bool(getattr(model, "lab_bn_fused", False))}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). TODO: đo thật, so với K * p50."""
    if k_views < 1:
        raise ValueError("k_views phải >= 1")
    from inference import view_hflip
    batch = kw.get("batch_size", 1)
    size = kw.get("img_size", 224)
    device = torch.device(kw.get("device", "cuda"))
    dtype = kw.get("dtype", "fp32")
    baseline = latency_report(model, **kw)
    net = copy.deepcopy(model).to(device).eval()
    net.half() if dtype == "fp16" else net.float()
    x = torch.randn(batch, 3, size, size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=dtype == "amp"):
            predictions = [net(view_hflip(x) if k % 2 else x).float().softmax(1) for k in range(k_views)]
            return torch.stack(predictions).mean(0)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    result = bench(forward, kw.get("warmup", 10), kw.get("iters", 100), sync)
    return {**baseline, **result, "k_views": k_views, "single_p50": baseline["p50"],
            "expected_p50": k_views * baseline["p50"], "relative_cost": result["p50"] / baseline["p50"],
            "images_per_s": batch / (result["p50"] / 1000),
            "preprocessing": "alternating identity/hflip + probability aggregation"}
