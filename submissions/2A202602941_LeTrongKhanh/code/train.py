"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict
from typing import get_type_hints, get_args

import numpy as np
import pandas as pd
import torch
import timm
import dataset
import model as models
import losses
from inference import apply_temperature

# Tìm eval.py gốc khi chạy CLI từ code/ hoặc notebook từ thư mục bài nộp.
_repo = next((p for p in Path(__file__).resolve().parents if (p / "eval.py").is_file()), None)
if _repo is not None:
    sys.path.insert(0, str(_repo))
from eval import compute_metrics, save_predictions

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). TODO."""
    return torch.optim.AdamW(models.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). TODO.

    Cập nhật theo bước (iteration) hoặc theo epoch đều được; ghi rõ bạn chọn gì.
    Gợi ý kiểm tra: vẽ đường LR theo bước để thấy đúng hình warmup + cosine.
    """
    total = cfg.epochs * steps_per_epoch
    warmup = int(cfg.warmup_epochs * steps_per_epoch)
    if total <= 0 or not 0 <= warmup < total:
        raise ValueError("Cần epochs > warmup_epochs và loader không rỗng")
    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay phải nằm trong [0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    def update(self, model) -> None:
        with torch.no_grad():
            source = model.state_dict()
            for name, target in self.model.state_dict().items():
                value = source[name].detach()
                if target.is_floating_point():
                    target.mul_(self.decay).add_(value, alpha=1 - self.decay)
                else:
                    target.copy_(value)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    TODO:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.freeze_backbone)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast + GradScaler), clip gradient nếu cần, optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
    """
    model.train()
    if cfg.init == "frozen":
        model.eval()
        model.get_classifier().train()
    total_loss, n = 0.0, 0
    device = torch.device(device)
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        targets = y
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits = model(x)
            loss = losses.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Train loss không hữu hạn")
        previous_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler có thể bỏ optimizer.step khi overflow: không cập nhật LR/EMA lúc đó.
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
            if ema is not None:
                ema.update(model)
        total_loss += float(loss.detach()) * len(y)
        n += len(y)
    if not n:
        raise ValueError("Train loader rỗng")
    return {"train_loss": total_loss / n, "lr": optimizer.param_groups[-1]["lr"]}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    TODO: model.eval(), torch.inference_mode(), gom kết quả. Softmax khi cần xác suất.
    """
    model.eval()
    names, labels, outputs = [], [], []
    total_loss, n = 0.0, 0
    with torch.inference_mode():
        for x, y, filenames in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += float(criterion(logits, y)) * len(y)
            n += len(y)
            names.extend(filenames)
            labels.append(y.cpu().numpy())
            outputs.append(logits.float().cpu().numpy())
    if not n:
        raise ValueError("Eval loader rỗng")
    return names, np.concatenate(labels), np.concatenate(outputs), total_loss / n


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    TODO: tối thiểu loss train/val và macro-F1 val theo epoch; có tiêu đề, nhãn trục, chú thích;
    khuyến khích thêm LR theo bước. Lưu bằng matplotlib với dpi đủ nét để đọc số.
    """
    import matplotlib.pyplot as plt
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df.epoch, df.train_loss, label="train")
    axes[0].plot(df.epoch, df.val_loss, label="val")
    axes[0].set_ylabel("Loss")
    axes[1].plot(df.epoch, df.val_macro_f1, label="val macro-F1")
    axes[1].plot(df.epoch, df.val_top1, label="val top-1")
    axes[1].set_ylabel("Score")
    axes[2].plot(df.epoch, df.lr, label="head LR (end of epoch)")
    axes[2].set_ylabel("Learning rate")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=.25)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    TODO theo thứ tự:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    directory = run_dir(cfg)
    config_path = directory / "config.json"
    summary_path = directory / "summary.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != asdict(cfg):
            raise ValueError("exp_id/seed đã có cấu hình khác; dùng exp_id mới")
        if summary_path.exists():
            print(f"Dùng kết quả đã lưu: {directory}")
            return json.loads(summary_path.read_text(encoding="utf-8"))
    if cfg.save_test_predictions and pred_path(cfg, "test").exists():
        raise ValueError("Đã có dự đoán test; không chạy test lần nữa")
    if cfg.epochs < 1 or cfg.batch_size < 2:
        raise ValueError("epochs >= 1 và batch_size >= 2")
    set_seed(cfg.seed)
    directory.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_info = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    (directory / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    train_loader = dataset.make_loader(train_df, cfg.images_dir,
        dataset.build_transforms(True, cfg.img_size, cfg.aug), cfg.batch_size,
        True, cfg.sampler, cfg.num_workers)
    val_loader = dataset.make_loader(val_df, cfg.images_dir,
        dataset.build_transforms(False, cfg.img_size), cfg.batch_size, False, num_workers=cfg.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate).to(device)
    pretrained_cfg = getattr(net, "pretrained_cfg", {})
    mean = pretrained_cfg.get("mean", dataset.IMAGENET_MEAN)
    std = pretrained_cfg.get("std", dataset.IMAGENET_STD)
    train_loader.dataset.transform = dataset.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    val_loader.dataset.transform = dataset.build_transforms(False, cfg.img_size, mean=mean, std=std)
    versions = {"python": sys.version, "torch": str(torch.__version__), "timm": timm.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "pretrained_cfg": pretrained_cfg, "pretrained_loaded": net.lab_pretrained,
                "gmac_tool": "torch.profiler conv/matmul FLOPs / 2; elementwise/norm excluded",
                "scheduler": "per optimizer step", "cudnn_deterministic": True}
    (directory / "environment.json").write_text(json.dumps(versions, indent=2, default=str), encoding="utf-8")
    counts = train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
    weight = losses.class_weights(counts, cfg.class_weight_beta or 0.0) if cfg.loss == "ce_weighted" else None
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                                      gamma=cfg.focal_gamma, weight=weight).to(device)
    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay is not None else None
    history, best_f1, best_epoch = [], -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        stats = train_one_epoch(net, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - start
        evaluated = ema.model if ema is not None else net
        _, y_true, logits, val_loss = evaluate(evaluated, val_loader, criterion, device)
        probs = apply_temperature(logits, 1.0)
        metrics = compute_metrics(y_true, probs.argmax(1), probs)
        row = {"epoch": epoch, **stats, "val_loss": val_loss, "val_macro_f1": metrics["macro_f1"],
               "val_top1": metrics["top1"], "train_seconds": train_seconds}
        history.append(row)
        print(cfg.exp_id, cfg.seed, row)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save({"model": evaluated.state_dict(), "epoch": epoch, "macro_f1": best_f1,
                        "cfg": asdict(cfg)}, directory / "best.pt")
        torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.model.state_dict() if ema else None, "epoch": epoch,
                    "cfg": asdict(cfg)}, directory / "last.pt")
        plot_curves(history, Path(cfg.out_dir).parent / "curves" /
                    f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png", f"{cfg.exp_id} {cfg.backbone} seed{cfg.seed}")
    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["model"])
    names, y_true, logits, _ = evaluate(net, val_loader, criterion, device)
    probs = apply_temperature(logits, 1.0)
    val_metrics = compute_metrics(y_true, probs.argmax(1), probs)
    np.savez(directory / "val_logits.npz", filenames=np.asarray(names), y_true=y_true, logits=logits)
    save_predictions(pred_path(cfg, "val"), names, y_true, probs)
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir,
            dataset.build_transforms(False, cfg.img_size, mean=mean, std=std), cfg.batch_size,
            False, num_workers=cfg.num_workers)
        names, y_true, logits, _ = evaluate(net, test_loader, criterion, device)
        np.savez(directory / "test_logits.npz", filenames=np.asarray(names), y_true=y_true, logits=logits)
        save_predictions(pred_path(cfg, "test"), names, y_true, apply_temperature(logits, 1.0))
    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
               "best_epoch": best_epoch, "macro_f1_val": val_metrics["macro_f1"],
               "top1_val": val_metrics["top1"], "ece_val": val_metrics["ece"],
               "f1_chinee_val": float(val_metrics["f1"][0]), "f1_snake_val": float(val_metrics["f1"][7]),
               "train_seconds_per_epoch": float(np.mean([r["train_seconds"] for r in history])),
               "params_m": models.count_params(net), "gmacs": models.count_gmacs(net, cfg.img_size),
               "img_size": cfg.img_size, "epochs": cfg.epochs,
               "weight_tag": pretrained_cfg.get("architecture", cfg.backbone) + "." + pretrained_cfg.get("tag", ""),
               "checkpoint": str(directory / "best.pt"), "config": asdict(cfg)}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    types = get_type_hints(Config)
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Cần KEY=VALUE: {pair}")
        key, value = pair.split("=", 1)
        if key not in types:
            raise ValueError(f"Config không có field {key}")
        annotation = types[key]
        args = get_args(annotation)
        if type(None) in args and value.lower() in ("none", "null"):
            result[key] = None
            continue
        kind = next((t for t in args if t is not type(None)), annotation)
        try:
            if kind is bool:
                if value.lower() not in ("true", "false", "1", "0"):
                    raise ValueError("bool cần true/false/1/0")
                result[key] = value.lower() in ("true", "1")
            else:
                result[key] = kind(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Giá trị không hợp lệ cho {key}: {value}") from exc
    return result


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    import argparse
    parser = argparse.ArgumentParser(description="DeepWeeds training")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    try:
        cfg = Config(**parse_overrides(args.set))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
