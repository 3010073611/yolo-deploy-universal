#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_onnx_vs_pt.py —— 导出自检：同一张图，.pt 和 .onnx 的输出必须对得上。

这是整条流水线里性价比最高的 5 分钟（教程 §4.5）：
    量化掉精度 / 板端乱框 / 后处理对不上 …… 第一步永远是排除"ONNX 本身就导错了"。
如果这里就不通过，后面所有优化都是在错误的地基上施工。

判据：最大绝对误差 < 1e-3（浮点顺序差异导致的 1e-5 量级误差是正常的）。

用法：
    python scripts/check_onnx_vs_pt.py --pt best.pt --onnx best_3scale.onnx --image val/1.jpg
    python scripts/check_onnx_vs_pt.py --pt best.pt --onnx best.onnx --image val/1.jpg --imgsz 320

说明：
  - 这里刻意用**拉伸 resize**（不做 letterbox），目的是让 PyTorch 和 ONNX 吃进**完全相同的
    输入张量**，比较的是"导出有没有导错"，不是端到端精度。端到端精度对比时两端预处理必须一致。
  - 自动识别形态：1 个 3 维输出 = 形态 A（解码单输出）；3 个 4 维输出 = 形态 B（三尺度特征图）。
  - 依赖：pip install ultralytics torch onnxruntime opencv-python

退出码：0 通过；1 不通过；2 环境/参数问题。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_blob(image: Path, imgsz: int) -> np.ndarray:
    """读图 → RGB → 拉伸到 imgsz×imgsz → /255 → NCHW float32。"""
    import cv2

    img = cv2.imread(str(image))
    if img is None:
        raise SystemExit(f"[FAIL] 读不到图片：{image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # OpenCV 默认 BGR，模型要 RGB
    img = cv2.resize(img, (imgsz, imgsz))                # 与下方注释一致：故意不 letterbox
    blob = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob)


def to_numpy(x):
    """把 torch.Tensor / numpy / list 统一成 numpy 数组。"""
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> int:
    ap = argparse.ArgumentParser(description="对比 .pt 与 .onnx 在同一个输入上的输出误差")
    ap.add_argument("--pt", required=True, help="训练产物 best.pt")
    ap.add_argument("--onnx", required=True, help="导出的 onnx（形态 A 或 B 都行）")
    ap.add_argument("--image", required=True, help="一张真实图片（建议用 val 集里的）")
    ap.add_argument("--imgsz", type=int, default=None, help="输入边长；不填则从 ONNX 输入 shape 自动读")
    ap.add_argument("--tol", type=float, default=1e-3, help="判定阈值，默认 1e-3")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[FAIL] 缺依赖：{e}\n        pip install ultralytics torch onnxruntime opencv-python", file=sys.stderr)
        return 2

    onnx_path, image = Path(args.onnx), Path(args.image)
    if not onnx_path.is_file():
        print(f"[FAIL] 找不到 {onnx_path}", file=sys.stderr)
        return 2

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    # ONNX 输入 shape： [1, 3, H, W]；动态维是字符串，取不到就要求用户手填 --imgsz
    imgsz = args.imgsz
    if imgsz is None:
        h = in_meta.shape[2]
        imgsz = h if isinstance(h, int) else None
    if not isinstance(imgsz, int):
        print("[FAIL] ONNX 输入是动态 shape，请显式给 --imgsz", file=sys.stderr)
        return 2

    blob = load_blob(image, imgsz)
    print(f"输入：{image} → {blob.shape}   （拉伸 resize，非 letterbox）")
    print(f"ONNX 输入名：{in_meta.name}  shape={in_meta.shape}")

    # ---------- 参考答案：PyTorch ----------
    yolo = YOLO(str(Path(args.pt)))
    model = yolo.model.float().cpu().eval()
    with torch.no_grad():
        out = model(torch.from_numpy(blob))

    # ultralytics 的 Detect 在 eval 态返回 (解码输出, [三尺度特征图])；
    # 若模型被置为 export 态则只返回特征图列表，这里都兼容。
    if isinstance(out, (tuple, list)) and len(out) > 1:
        ref_main, ref_feats = to_numpy(out[0]), [to_numpy(f) for f in out[1]]
    elif isinstance(out, (tuple, list)):
        ref_main, ref_feats = to_numpy(out[0]), []
    else:
        ref_main, ref_feats = to_numpy(out), []

    # ---------- 学生答卷：ONNX ----------
    onnx_outs = [np.asarray(o) for o in sess.run(None, {in_meta.name: blob})]
    print(f"ONNX 输出 {len(onnx_outs)} 个：" + ", ".join(str(o.shape) for o in onnx_outs))

    diffs: list[tuple[str, float]] = []

    if len(onnx_outs) == 1:
        # ----- 形态 A：解码单输出，直接和 PyTorch 的解码输出比 -----
        if ref_main.shape != onnx_outs[0].shape:
            print(f"[FAIL] shape 不一致：pt {ref_main.shape} vs onnx {onnx_outs[0].shape}")
            print("       常见原因：导出时开了 NMS / dynamic / 或 imgsz 与训练不同")
            return 1
        diffs.append(("形态A 解码输出", float(np.abs(ref_main - onnx_outs[0]).max())))
    else:
        # ----- 形态 B：三尺度特征图。按 shape 一一配对，避免顺序不同导致误判 -----
        if not ref_feats:
            print("[FAIL] .pt 只返回了单个输出，但 ONNX 有多个输出——可能导出脚本包错了层")
            return 1
        remaining = list(onnx_outs)
        for idx, feat in enumerate(ref_feats):
            match = next((o for o in remaining if o.shape == feat.shape), None)
            if match is None:
                print(f"[FAIL] 尺度{idx} 在 ONNX 里找不到 shape={feat.shape} 的输出")
                print("       检查：是否把解码后的输出导成了特征图，或 imgsz 不一致")
                return 1
            remaining.remove(match)
            diffs.append((f"尺度{idx} {feat.shape}", float(np.abs(feat - match).max())))

    # ---------- 出结论 ----------
    print()
    ok = True
    for name, d in diffs:
        flag = "OK  " if d < args.tol else "FAIL"
        print(f"[{flag}] {name}: max|pt-onnx| = {d:.3e}")
        ok = ok and d < args.tol

    print()
    if ok:
        print(f"[OK] 全部输出误差 < {args.tol:g} → ONNX 导出正确，可以进入量化/转换。")
        return 0

    print(f"[FAIL] 存在误差 ≥ {args.tol:g} 的输出。按顺序排查：")
    print("       1) imgsz 是否与训练一致（训练=导出=部署）")
    print("       2) 预处理是否 RGB + /255（不是 BGR，也不是 0~255 直喂）")
    print("       3) 导出脚本是否包错了层（形态 B 应拿 Detect 的三尺度特征图）")
    print("       4) opset / simplify 是否把算子改坏了（试 opset=13 且关掉 simplify）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
