#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py —— 把"一堆图和标注"切成 train/val，且保证 val 不是随机抽出来的一坨。

为什么不用 listdir + shuffle 一刀切（原文那段脚本的三个真问题）：
  1) 不分层：小类别可能 0 张进 val，那该类的 mAP 直接失真，你还以为模型坏了；
  2) 遇到没有 .txt 的背景图会 FileNotFoundError —— 而背景图恰恰是抑制误检的重要手段；
  3) 不告诉你切完长什么样，等训练跑完看混淆矩阵才发现某类没被考到。

本脚本做的事：
  - 按"类"分层：从最稀有的类开始，优先保证每类都有 val 样本；
  - 一张图里有多个类时不会重复切分（先分配到 val 的类优先，后面的类让路）；
  - 背景图（无 txt）按比例随机分，不会被丢掉；
  - 切完打印每类的 train/val 分布，并列出"val 里没有的类"。

目录约定（和教程 §2.1 一致）：
    raw/
    ├── images/        # 你的原始全量图，可以带子目录
    └── labels/        # 与 images 同构的 .txt（可以缺，缺了就当背景图）

用法：
    python scripts/split_dataset.py --src raw --dst dataset --val-ratio 0.1
    python scripts/split_dataset.py --src raw --dst dataset --mode move --make-empty-labels

退出码：0 成功；2 参数/目录有问题。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: Path) -> list[Path]:
    """递归收集图片，返回相对 root 的路径（保留子目录结构）。"""
    return sorted(
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def read_classes(label_path: Path) -> list[int]:
    """
    读一个 YOLO txt，返回里面出现过的 class_id（去重）。
    文件不存在 / 空文件 / 行格式不对 → 返回空列表（= 背景图），不抛异常。
    """
    if not label_path.is_file():
        return []
    classes: set[int] = set()
    try:
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 5:  # class cx cy w h
                try:
                    classes.add(int(float(parts[0])))
                except ValueError:
                    continue
    except OSError:
        return []
    return sorted(classes)


def place(src: Path, dst_root: Path, rel: Path, split: str, mode: str) -> None:
    """把一张图放到 dst 的 images/<split>/ 下，并按 mode 决定 copy / move / hardlink。"""
    dst = dst_root / "images" / split / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "hardlink":
        try:
            os.link(src, dst)          # 同盘时零拷贝，最省时间和空间
        except OSError:
            shutil.copy2(src, dst)     # 跨盘/文件系统不支持时自动降级
    else:
        shutil.copy2(src, dst)


def place_label(label_src: Path, dst_root: Path, rel: Path, split: str, make_empty: bool) -> None:
    """标签跟着图走；图上没标签时，可选地生成一个空 txt（ultralytics 喜欢这种写法）。"""
    dst = dst_root / "labels" / split / rel.with_suffix(".txt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if label_src.is_file():
        if not dst.exists():
            shutil.copy2(label_src, dst)
    elif make_empty:
        dst.write_text("", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO 数据集分层划分（每类保证进 val）")
    ap.add_argument("--src", required=True, help="原始数据根目录，里面应有 images/ 和 labels/")
    ap.add_argument("--dst", required=True, help="输出目录（会生成 images/train|val 和 labels/train|val）")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例，默认 0.1")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，默认 42（可复现）")
    ap.add_argument("--mode", choices=["copy", "move", "hardlink"], default="copy",
                    help="copy=复制（安全，占空间）/ move=移动（省空间）/ hardlink=硬链接（同盘零拷贝）")
    ap.add_argument("--make-empty-labels", action="store_true",
                    help="给没有标注的图生成空 txt（ultralytics 更推荐这种写法）")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    img_root, lbl_root = src / "images", src / "labels"
    if not img_root.is_dir():
        print(f"[FAIL] 找不到 {img_root}，--src 应指向含 images/ 和 labels/ 的目录", file=sys.stderr)
        return 2
    if not 0 < args.val_ratio < 1:
        print("[FAIL] --val-ratio 必须在 (0,1) 之间", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rels = find_images(img_root)
    if not rels:
        print(f"[FAIL] {img_root} 下没有图片", file=sys.stderr)
        return 2

    # ---- 1. 先摸清每张图含哪些类 ----
    img_classes: dict[Path, list[int]] = {}
    class_to_imgs: dict[int, list[Path]] = defaultdict(list)
    backgrounds: list[Path] = []
    for rel in rels:
        cls = read_classes(lbl_root / rel.with_suffix(".txt"))
        img_classes[rel] = cls
        if cls:
            for c in cls:
                class_to_imgs[c].append(rel)
        else:
            backgrounds.append(rel)

    for imgs in class_to_imgs.values():
        imgs.sort()

    # ---- 2. 分层：从"最稀有的类"开始分配，保证每类都有 val ----
    val_set: set[Path] = set()
    # 稀有类优先：样本少的类先挑，免得被大类把名额占光
    for cls in sorted(class_to_imgs, key=lambda c: len(class_to_imgs[c])):
        imgs = class_to_imgs[cls]
        n = len(imgs)
        # 期望的 val 张数：向上取整，但不超过总数-1（至少留 1 张给 train）
        want = min(max(1, math.ceil(n * args.val_ratio)), max(1, n - 1)) if n > 1 else 1
        have = sum(1 for p in imgs if p in val_set)
        if have >= want:
            continue
        # 候选项：还没进 val 的该类图片，随机打乱后取够为止
        candidates = [p for p in imgs if p not in val_set]
        rng.shuffle(candidates)
        for p in candidates[: want - have]:
            val_set.add(p)

    # ---- 3. 背景图按比例随机分（它们同样要出现在 val 里，用来测误检）----
    rng.shuffle(backgrounds)
    n_bg_val = round(len(backgrounds) * args.val_ratio)
    val_set.update(backgrounds[:n_bg_val])

    # ---- 4. 落盘 ----
    for rel in rels:
        split = "val" if rel in val_set else "train"
        place(img_root / rel, dst, rel, split, args.mode)
        place_label(lbl_root / rel.with_suffix(".txt"), dst, rel, split, args.make_empty_labels)

    # ---- 5. 打印分布（切完立刻能看出问题）----
    train_cnt, val_cnt = Counter(), Counter()
    for rel in rels:
        target = val_cnt if rel in val_set else train_cnt
        for c in img_classes[rel]:
            target[c] += 1

    all_cls = sorted(class_to_imgs)
    print(f"总图片 {len(rels)}（含背景图 {len(backgrounds)}） → "
          f"train {len(rels) - len(val_set)} / val {len(val_set)}")
    print(f"{'class':>6} | {'train':>5} | {'val':>5}")
    print("-" * 22)
    missing = []
    for c in all_cls:
        print(f"{c:>6} | {train_cnt[c]:>5} | {val_cnt[c]:>5}")
        if val_cnt[c] == 0 and class_to_imgs[c]:
            missing.append(c)

    if missing:
        print(f"\n[WARN] 这些类在 val 里是 0 张：{missing}")
        print("       张数太少（1~2 张）时无法既进 val 又留 train，建议补图或合并类别。")
    if n_bg_val == 0 and backgrounds:
        print(f"\n[WARN] 有 {len(backgrounds)} 张背景图，但 --val-ratio 太小导致 val 里一张都没有。")
    print("\n[OK] 完成。检查：① 随便抽一对 val 的图和 txt 看坐标对不对；② 上面的分布没有 0。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
