#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rknn-convert.py —— Rockchip RKNN-Toolkit2 转换 + 量化 + 精度自查模板。

为什么把它做成脚本而不是几行代码：
    1) `accuracy_analysis` 必须在同一个 RKNN 对象上、build() 之后调用，分开跑会报模型未构建；
    2) 校准清单、目标芯片、量化算法这些参数一改就要重跑整条链，脚本化才不容易漏；
    3) 转换完立刻做精度自查，比上板后再怀疑人生便宜得多。

前置（本机是 x86 Linux / Windows / WSL，板子是另一台）：
    pip install rknn-toolkit2          # 版本必须与板端 librknnrt.so 配套！
    echo dataset/images/val/0001.jpg > dataset.txt   # 每行一个校准图路径

用法：
    python assets/rknn-convert.py --onnx best_3scale.onnx --dataset dataset.txt \
        --platform rk3588 --out best.rknn --check-image dataset/images/val/0001.jpg

注意：
    * 形态 B（三尺度）配 rknn_model_zoo 的官方后处理最省事；形态 A 只差 NMS。
    * mean_values / std_values 已把 /255 内嵌进模型 → 板端直接喂 0~255 的 uint8 RGB 图。
    * 掉精度先用 snapshot 报告找敏感层（余弦相似度 < 0.9），再走 hybrid 两步法。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="RKNN 转换 + 量化 + 精度自查")
    ap.add_argument("--onnx", required=True, help="形态 B 或 A 的 onnx")
    ap.add_argument("--dataset", required=True, help="校准清单 dataset.txt（每行一个图片路径）")
    ap.add_argument("--platform", default="rk3588",
                    help="rk3588 / rk3576 / rk3566 / rk3568 / rv1103 / rv1106 ...")
    ap.add_argument("--out", default="best.rknn", help="输出 rknn 路径")
    ap.add_argument("--algo", default="normal",
                    choices=["normal", "kl_divergence", "mmse"],
                    help="normal=MinMax（默认）；掉精度时换 kl_divergence 或 mmse")
    ap.add_argument("--mean", default="0,0,0", help="按 RGB 顺序，默认 0")
    ap.add_argument("--std", default="255,255,255", help="默认 255 → 等价于 /255 归一化")
    ap.add_argument("--check-image", help="做 accuracy_analysis 的样例图（不填则跳过自查）")
    args = ap.parse_args()

    onnx_path, dataset = Path(args.onnx), Path(args.dataset)
    if not onnx_path.is_file():
        print(f"[FAIL] 找不到 {onnx_path}", file=sys.stderr)
        return 2
    if not dataset.is_file():
        print(f"[FAIL] 找不到校准清单 {dataset}", file=sys.stderr)
        return 2

    try:
        from rknn.api import RKNN
    except ImportError:
        print("[FAIL] 缺 rknn-toolkit2：请在 x86 转换机上 pip install rknn-toolkit2（版本要与板端配套）",
              file=sys.stderr)
        return 2

    def triples(s: str) -> list[float]:
        parts = [float(x) for x in s.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"mean/std 需要 3 个值，收到：{s}")
        return parts

    rknn = RKNN(verbose=True)

    # mean/std 写在 config 里 = 归一化进模型，板端就不用再 /255；
    # 若你选择在板端代码里做归一化，这里就必须改成 0/1，否则会做两次（见铁律 1）。
    rknn.config(
        mean_values=[triples(args.mean)],
        std_values=[triples(args.std)],
        target_platform=args.platform,
        quantized_dtype="w8a8",          # 权重 8bit + 激活 8bit
        quantized_algorithm=args.algo,   # normal / kl_divergence / mmse
        quantized_method="channel",      # 权重 per-channel，现代工具链默认就该开
        optimization_level=3,
    )

    print(f"--> load_onnx {onnx_path}")
    if rknn.load_onnx(str(onnx_path)) != 0:
        print("[FAIL] load_onnx 失败（检查是否含 NPU 不支持的算子，如 NMS/TopK）", file=sys.stderr)
        return 1

    print(f"--> build（target={args.platform}, algo={args.algo}）")
    if rknn.build(do_quantization=True, dataset=str(dataset)) != 0:
        print("[FAIL] build 失败（校准清单路径、图片格式、shape 都要看）", file=sys.stderr)
        return 1

    # ★ 必须在 build 之后、export 之前，用同一个 rknn 对象调用
    if args.check_image:
        print(f"--> accuracy_analysis（样例图 {args.check_image}）")
        rknn.accuracy_analysis(inputs=[args.check_image], output_dir="./snapshot")
        print("    看 ./snapshot 报告：每层余弦相似度，<0.9 的层就是混合精度要保的敏感层")

    print(f"--> export_rknn {args.out}")
    if rknn.export_rknn(str(args.out)) != 0:
        print("[FAIL] export_rknn 失败", file=sys.stderr)
        return 1

    rknn.release()
    print(f"[OK] 产物：{args.out}")
    print("     下一步：把 rknn 拷到板上，用 rknn-toolkit-lite2（版本配套）或 C API 推理。")
    print("     精度掉 >3% 时：换 --algo 重转 → 补校准集 → hybrid 两步法保敏感层。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
