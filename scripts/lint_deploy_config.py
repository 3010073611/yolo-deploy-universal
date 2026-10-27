#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lint_deploy_config.py —— 转换前先查配置，把"抄错名字"这类错误挡在编译之前。

为什么需要它：
    部署阶段最高频的报错不是"工具不支持"，而是**名字/顺序/形状对不上**：
      * config.json 里的输出名写成了 YOLO11 的 /model.23/...，但模型是 YOLOv8（/model.22/...）
      * dst_perm 是按 4 维写的，模型输出却是 3 维
      * MUD 的 labels 少一个/多一个，或顺序和训练时不一样
      * layer_configs 里的层名在 ONNX 里根本不存在
    这些都能在转换前 1 秒内查出来，不用等 Docker 跑十分钟报错。

它不联网、不改文件，只读三样东西并交叉比对：
    ONNX（真相来源） ↔ config.json / .mud（转换配置） ↔ dataset.yaml（标签顺序）

用法：
    python scripts/lint_deploy_config.py --onnx best_3scale.onnx \
        --config config.json --mud model.mud --data dataset.yaml

    只查一部分也可以（比如只查 ONNX 与 MUD 的 labels）：
    python scripts/lint_deploy_config.py --onnx best_3scale.onnx --mud model.mud

依赖：pip install onnx（config/mud/yaml 的解析用标准库）

退出码：0 全部通过（可能有 WARN）；1 有 FAIL；2 文件/依赖问题
"""

from __future__ import annotations

import argparse
import configparser
import json
import re
import sys
from pathlib import Path

FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"[FAIL] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


# --------------------------------------------------------------------------
# 1) ONNX：读取真实的输入/输出名与 shape
# --------------------------------------------------------------------------
def read_onnx(path: Path):
    try:
        import onnx
    except ImportError:
        print("[FAIL] 缺依赖 onnx：pip install onnx", file=sys.stderr)
        raise SystemExit(2)

    model = onnx.load(str(path))
    initializers = {i.name for i in model.graph.initializer}

    def shape_of(vi):
        return [
            d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in vi.type.tensor_type.shape.dim
        ]

    # graph.input 里会混进权重常量（initializer），要排掉
    inputs = [(i.name, shape_of(i)) for i in model.graph.input if i.name not in initializers]
    outputs = [(o.name, shape_of(o)) for o in model.graph.output]
    node_names = {n.name for n in model.graph.node if n.name}
    return inputs, outputs, node_names


def detect_form(outputs):
    """判断形态与类别数：A=单输出解码；B=三尺度特征图。返回 (form, nc)。"""
    if len(outputs) == 1 and len(outputs[0][1]) == 3:
        c = outputs[0][1][1] if len(outputs[0][1]) > 1 else None
        return "A", (c - 4 if isinstance(c, int) else None)

    if len(outputs) == 3 and all(len(s) == 4 for _, s in outputs):
        channels = {s[1] for _, s in outputs}
        if len(channels) != 1:
            return "B", None  # 三个尺度通道数不一致，肯定不对
        c = next(iter(channels))
        if not isinstance(c, int):
            return "B", None
        if c > 64:                      # v8/v11：64 = 4 * reg_max(16)
            return "B", c - 64
        if (c - 15) % 3 == 0:           # v5：3 * (5 + nc)
            return "B", (c - 15) // 3
    return "?", None


# --------------------------------------------------------------------------
# 2) config.json（以 Pulsar2 为例，RKNN 的字段名不同，逻辑一样）
# --------------------------------------------------------------------------
def check_config(path: Path, inputs, outputs, node_names) -> None:
    print(f"\n== 检查转换配置 {path.name} ==")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - 这里就是要抓住所有解析问题
        fail(f"config.json 解析失败：{e}")
        return

    onnx_in_names = [n for n, _ in inputs]
    onnx_out_names = [n for n, _ in outputs]
    out_rank = {n: len(s) for n, s in outputs}

    # --- 输入 ---
    for proc in cfg.get("input_processors", []):
        name = proc.get("tensor_name")
        if name not in onnx_in_names:
            fail(f"input_processors.tensor_name='{name}' 在 ONNX 输入里不存在（实际：{onnx_in_names}）")
        else:
            ok(f"输入节点名匹配：{name}")

        fmt = str(proc.get("src_format", "")).upper()
        tfmt = str(proc.get("tensor_format", "")).upper()
        if "BGR" in (fmt, tfmt):
            warn(f"{path.name} 声明了 BGR（src_format={fmt or '?'}, tensor_format={tfmt or '?'}）："
                 "只有训练和板端都用 BGR 时才成立。ultralytics 训练用 RGB，多数情况应写 RGB + csc_mode=NoCSC")

        if "tensor_layout" not in proc:
            warn("input_processors 缺少 tensor_layout（模型内部布局，通常 NCHW）；"
                 "src_layout 是板端内存布局（通常 NHWC），两者别写反")
        if "csc_mode" not in proc:
            warn("input_processors 缺少 csc_mode；官方示例用 NoCSC（不做颜色空间转换）")

    if "input_shapes" in cfg:
        warn("config 里写了 input_shapes：输入尺寸本应直接从 ONNX 读，多写一遍就多一个写错的机会")

    # --- 量化 ---
    quant = cfg.get("quant", {})
    if quant:
        for item in quant.get("input_configs", []):
            if item.get("tensor_name") not in onnx_in_names:
                fail(f"quant.input_configs.tensor_name='{item.get('tensor_name')}' 不存在于 ONNX 输入")
            ds = str(item.get("calibration_dataset", ""))
            # 只有本地相对路径才检查是否存在；/data/... 是容器内路径，宿主机查不到很正常
            if ds and not ds.startswith("/") and not Path(ds).exists():
                warn(f"校准集路径 '{ds}' 在宿主机上不存在（若这是容器内路径可忽略）")
            if int(item.get("calibration_size", 0) or 0) <= 0:
                fail("quant.input_configs.calibration_size 必须 > 0")

    # --- 输出：本文件最容易出错的地方 ---
    procs = cfg.get("output_processors", [])
    proc_names = [p.get("tensor_name") for p in procs]
    if not procs:
        warn("没有 output_processors：多数 NPU 工具需要显式指定输出节点")

    missing = [n for n in proc_names if n not in onnx_out_names]
    if missing:
        fail(f"output_processors 里的 {missing} 在 ONNX 里不存在")
        print("       ONNX 真实输出：")

        for n, s in outputs:
            print(f"         - {n}  {s}")
        # 给出针对性提示：最常见的两种层号写错
        if any("model.22" in n for n in onnx_out_names) and any("model.23" in str(m) for m in missing):
            print("       → 看起来是 YOLOv8 的模型（model.22），却写了 YOLO11（model.23）的名字。")
        if any("model.23" in n for n in onnx_out_names) and any("model.22" in str(m) for m in missing):
            print("       → 看起来是 YOLO11 的模型（model.23），却写了 YOLOv8（model.22）的名字。")

    if not missing:
        ok(f"输出节点名全部匹配（{len(proc_names)} 个）")

    for p in procs:
        name, perm = p.get("tensor_name"), p.get("dst_perm")
        if perm is not None and name in out_rank:
            if len(perm) != out_rank[name]:
                fail(f"{name} 是 {out_rank[name]} 维输出，但 dst_perm 有 {len(perm)} 个元素：{perm}")
            elif sorted(perm) != list(range(len(perm))):
                fail(f"{name} 的 dst_perm={perm} 不是合法排列（应为 0..{len(perm) - 1} 的重排）")

    # --- 混合精度保层：层名在不在图里 ---
    for lc in cfg.get("layer_configs", []):
        lname = lc.get("layer_name")
        if node_names and lname not in node_names:
            fail(f"layer_configs 里的层名不存在于 ONNX：{lname}（Netron 里查确切节点名）")
        dt = str(lc.get("data_type", "")).upper()
        if dt and dt not in {"FP32", "FP16", "INT8", "U8", "AUTO"}:
            warn(f"layer_configs.data_type='{dt}' 不是常见取值（FP32/FP16/INT8）")

    if cfg.get("compiler", {}).get("check") in (None, 0, 1):
        warn("compiler.check 建议设为 2 或 3（让编译器对比 NPU/ONNX 输出，能提前暴露量化坏层）")


# --------------------------------------------------------------------------
# 3) MUD（MaixPy 的模型描述文件）
# --------------------------------------------------------------------------
def read_mud(path: Path) -> dict:
    cp = configparser.ConfigParser(interpolation=None, strict=False, allow_no_value=True)
    cp.read_string(path.read_text(encoding="utf-8", errors="ignore"))
    data = {}
    for section in cp.sections():
        for key, value in cp.items(section):
            # MUD 里常有人把说明写在值后面（"labels = a,b  ← ★ 顺序..."），先切掉
            data[key.strip().lower()] = (value or "").split("←")[0].strip()
    return data


def parse_labels(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def check_mud(path: Path, outputs, form: str, nc) -> list[str] | None:
    print(f"\n== 检查 MUD {path.name} ==")
    mud = read_mud(path)
    labels = parse_labels(mud.get("labels", ""))

    if not labels:
        fail("MUD 里没有 labels（MaixPy 靠它显示类别名）")
        return None

    if nc:
        if len(labels) != nc:
            fail(f"labels 有 {len(labels)} 个，但模型输出对应 nc={nc} 个类别"
                 "（类别数不一致 → 后处理越界或类别全错）")
        else:
            ok(f"labels 数量与模型类别数一致：{nc}")

    mt = mud.get("model_type", "").lower()
    if mt and mt not in {"yolo11", "yolov8", "yolov5", "detector", "classifier"}:
        warn(f"model_type='{mt}' 不是 MaixPy 常见取值（yolo11 / yolov8 / classifier…）")
    if mt == "yolo11" and any("/model.22" in n for n, _ in outputs):
        warn("model_type 写的是 yolo11，但 ONNX 输出是 /model.22（YOLOv8 的层号），确认一下")
    if mt == "yolov8" and any("/model.23" in n for n, _ in outputs):
        warn("model_type 写的是 yolov8，但 ONNX 输出是 /model.23（YOLO11 的层号），确认一下")

    it = mud.get("input_type", "").lower()
    if it and it != "rgb":
        warn(f"input_type='{it}'：ultralytics 训练用 RGB，MaixPy 相机默认也是 RGB，通常应写 rgb")

    for key in ("model_npu", "model_vnpu"):
        fn = mud.get(key)
        if fn and not (path.parent / fn).exists():
            warn(f"{key} 指向的 {fn} 不在 {path.parent} 下（板端会加载失败）")

    if "mean" not in mud or "scale" not in mud:
        warn("缺少 mean/scale：虽然预处理已编译进模型，但 MaixPy 的 MUD 通常还是带着这两个字段")

    return labels


# --------------------------------------------------------------------------
# 4) dataset.yaml（标签顺序的真相来源）
# --------------------------------------------------------------------------
def parse_yaml_names(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r"^\s*names\s*:", ln)), None)
    if start is None:
        return None

    head = lines[start].split(":", 1)[1].strip()
    if head.startswith("["):                      # names: ['cat', 'dog']
        body = head + " " + " ".join(lines[start + 1:])
        return re.findall(r"['\"]([^'\"]+)['\"]", body) or None

    names: dict[int, str] = {}
    for ln in lines[start + 1:]:
        if ln.strip() and not ln.startswith((" ", "\t", "-")):
            break
        m = re.match(r"\s*(-?\d+)\s*:\s*(.+?)\s*$", ln)   # 0: cat
        if m:
            names[int(m.group(1))] = m.group(2).strip().strip("'\"")
            continue
        m = re.match(r"\s*-\s*(.+?)\s*$", ln)             # - cat
        if m:
            names[len(names)] = m.group(1).strip().strip("'\"")
    return [names[k] for k in sorted(names)] if names else None


def check_data(path: Path, nc, mud_labels) -> None:
    print(f"\n== 检查数据集标签 {path.name} ==")
    names = parse_yaml_names(path)
    if not names:
        warn("没解析出 names（可能是别的写法），跳过标签顺序比对")
        return
    if nc and len(names) != nc:
        fail(f"dataset.yaml 的 names 有 {len(names)} 个，模型 nc={nc}，不一致")
    else:
        ok(f"names 数量 {len(names)}")

    if mud_labels:
        if names != mud_labels:
            first = next((i for i, (a, b) in enumerate(zip(names, mud_labels)) if a != b), None)
            if first is not None:
                fail(f"labels 顺序与 names 不一致：第 {first} 个是 "
                     f"{mud_labels[first]!r}，但 names 里是 {names[first]!r}（类别显示错位的病根）")
            else:
                fail("labels 与 names 数量一致但内容不同")
        else:
            ok("labels 顺序与 dataset.yaml 的 names 逐字一致")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="转换前校验 ONNX 与部署配置是否自洽")
    ap.add_argument("--onnx", required=True, help="导出的 ONNX（真相来源）")
    ap.add_argument("--config", help="Pulsar2 的 config.json（或其他工具的等价配置）")
    ap.add_argument("--mud", help="MaixPy 的 .mud")
    ap.add_argument("--data", help="dataset.yaml（用来核对标签顺序）")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"[FAIL] 找不到 {onnx_path}", file=sys.stderr)
        return 2

    inputs, outputs, node_names = read_onnx(onnx_path)
    form, nc = detect_form(outputs)

    print(f"== ONNX {onnx_path.name} ==")
    for n, s in inputs:
        print(f"  输入  {n}  {s}")
    for n, s in outputs:
        print(f"  输出  {n}  {s}")

    if form == "?":
        fail("识别不出形态：期望 1 个 3 维输出（A）或 3 个 4 维输出（B）。"
             "若这是形态 C（含 NMS），多数 NPU 工具链不支持，请改导 A/B")
    else:
        ok(f"形态 {form}，推断 nc={nc}")

    # 三尺度 shape 自检：H/W 必须是输入尺寸的 1/8、1/16、1/32
    if form == "B" and len(inputs) == 1 and len(inputs[0][1]) == 4:
        h, w = inputs[0][1][2], inputs[0][1][3]
        if isinstance(h, int) and isinstance(w, int):
            expected = [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
            got = [(s[2], s[3]) for _, s in outputs]
            if got != expected:
                fail(f"三尺度特征图尺寸 {got} 与输入 {h}x{w} 应得的 {expected} 不符（imgsz 或导出方式不对）")
            else:
                ok(f"三尺度尺寸与输入 {h}x{w} 匹配")

    if args.config:
        check_config(Path(args.config), inputs, outputs, node_names)

    mud_labels = None
    if args.mud:
        mud_labels = check_mud(Path(args.mud), outputs, form, nc)

    if args.data:
        check_data(Path(args.data), nc, mud_labels)

    print("\n" + "-" * 60)
    if FAILS:
        print(f"[FAIL] 共 {len(FAILS)} 项必须修复，否则转换/上板大概率失败：")

        for i, m in enumerate(FAILS, 1):
            print(f"  {i}. {m}")
        return 1
    print("[OK] 没有发现硬性错误。有 WARN 的话按提示确认一下，再进 Docker 转换。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
