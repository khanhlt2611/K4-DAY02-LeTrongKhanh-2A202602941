"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    TODO:
      - đọc ba file CSV bằng pandas
      - trả về (train_df, val_df, test_df)
    """
    if fold not in range(5):
        raise ValueError("fold phải nằm trong 0..4")
    frames = tuple(pd.read_csv(Path(labels_dir) / f"{split}_subset{fold}.csv")
                   for split in ("train", "val", "test"))
    # CSV chia sẵn của tác giả chỉ có Filename, Label; Species không bắt buộc.
    # Giữ nguyên hàng, nhãn và thứ tự của các CSV gốc.
    for split, df in zip(("train", "val", "test"), frames):
        if not {"Filename", "Label"}.issubset(df.columns):
            path = Path(labels_dir) / f"{split}_subset{fold}.csv"
            raise ValueError(
                f"{path}: cần cột Filename, Label; thực tế có {list(df.columns)}"
            )
    return frames


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    TODO kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    frames = dict(train=train_df, val=val_df, test=test_df)
    sets, counts = {}, {}
    for name, df in frames.items():
        if df[["Filename", "Label"]].isna().any().any() or df.Filename.duplicated().any():
            raise ValueError(f"{name}: thiếu dữ liệu hoặc trùng Filename")
        if not df.Label.isin(range(NUM_CLASSES)).all():
            raise ValueError(f"{name}: Label phải là số nguyên 0..8")
        sets[name] = set(df.Filename)
        counts[name] = {int(k): int(v) for k, v in
                        df.Label.value_counts().reindex(range(NUM_CLASSES), fill_value=0).items()}
    overlap = {f"{a}_{b}": len(sets[a] & sets[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlap.values()):
        raise ValueError(f"Split có giao khác rỗng: {overlap}")
    union = set.union(*sets.values())
    if len(union) != 17509:
        raise ValueError(f"Hợp split có {len(union)} ảnh, phải là 17509")
    for name, ratio in (("train", .6), ("val", .2), ("test", .2)):
        if abs(len(frames[name]) / len(union) - ratio) > .01:
            raise ValueError(f"{name}: tỷ lệ lệch hơn 1 điểm phần trăm")
    missing = [f for f in sorted(union) if not (Path(images_dir) / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh, ví dụ {missing[:5]}")
    stats = {"n": {k: len(v) for k, v in frames.items()}, "per_class": counts,
             "overlap": overlap, "union": len(union)}
    print(stats)
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    TODO: dùng torchvision.transforms (hoặc v2). Lưu ý: lật dọc có hợp lệ với ảnh cỏ dại không?
    """
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    if img_size <= 0 or aug not in ("basic", "color", "trivial", "randaug"):
        raise ValueError("img_size/aug không hợp lệ")
    if train:
        ops = [T.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
               T.RandomHorizontalFlip()]
        extra = {"color": T.ColorJitter(.2, .2, .2, .05),
                 "trivial": T.TrivialAugmentWide(), "randaug": T.RandAugment()}
        if aug in extra:
            ops.append(extra[aug])
    else:
        # 224: center crop từ ảnh 256; độ phân giải cao hơn: resize trước crop.
        ops = [T.Resize(max(256, img_size), interpolation=InterpolationMode.BICUBIC),
               T.CenterCrop(img_size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    TODO:
      - __init__(self, df, images_dir, transform): giữ df, mở ảnh bằng PIL, chuyển sang RGB
      - __len__
      - __getitem__ -> (tensor, int(label), filename)
      - (tuỳ chọn) nạp trước ảnh vào RAM nếu bị nghẽn đọc đĩa trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row.Filename)
        with Image.open(self.images_dir / filename) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision.transforms.functional import to_tensor
            image = to_tensor(image)
        return image, int(row.Label), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    TODO:
      - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
        (thứ tự phải ổn định để ghép logit với Filename)
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
        1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
      - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
      - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """
    if sampler not in (None, "balanced") or (not train and sampler is not None):
        raise ValueError("Sampler cân bằng chỉ dành cho train")
    data = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator().manual_seed(torch.initial_seed())
    sample = None
    if sampler == "balanced":
        counts = df.Label.value_counts()
        weights = torch.tensor([1.0 / counts[label] for label in df.Label], dtype=torch.double)
        sample = WeightedRandomSampler(weights, len(df), replacement=True, generator=generator)
    return DataLoader(data, batch_size=batch_size, shuffle=train and sample is None,
                      sampler=sample, num_workers=num_workers, pin_memory=True,
                      drop_last=train and len(data) >= batch_size,
                      worker_init_fn=_seed_worker, generator=generator)


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
