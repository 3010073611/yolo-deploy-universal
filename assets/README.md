# 模板说明

三个模板都**不能直接跑**——里面每一处都要按你的模型改。这里逐项说明"改什么、为什么"。

## `pulsar2-maixcam2.json`

用于 MaixCAM2（爱芯 AX630C），转换时目标平台写 `AX620E`。

| 字段 | 怎么改 | 为什么 |
|---|---|---|
| `quant.input_configs[0].calibration_dataset` | 指向你打包的 `train.tar` | Pulsar2 吃 tar 包形式的校准集 |
| `calibration_size` | ≤ tar 里的图片数，官方建议 20~100 | 少了定标不准，多了收益递减 |
| `calibration_mean/std` | 训练用什么就写什么；`0/255` 表示把 `/255` 编进模型 | 板端就不用再做归一化（**只能做一次**） |
| `input_processors[].tensor_format/src_format` | 训练用 RGB 就写 `RGB` | 写 `BGR` 会让模型看到 R/B 互换，精度悄悄掉 |
| `input_processors[].csc_mode` | `NoCSC` | 不做颜色空间转换 |
| `output_processors[]` | **改成 `python scripts/lint_deploy_config.py` 打印出来的真实输出名** | 名字抄错是最常见的转换失败原因 |
| `output_processors[].dst_perm` | 4 维输出用 `[0,2,3,1]`；3 维输出不要套 4 维排列 | 维度不匹配会直接报错 |
| `compiler.check` | 保持 3 + `CheckOutput` | 让编译器对比 NPU 与 ONNX 输出，低于 0.9 告警 |

> YOLO11 输出名形如 `/model.23/Concat{,1,2}_output_0`；**YOLOv8 是 `/model.22/Concat_{1,2,3}_output_0`**（从 `_1` 起）。
> 模型规模 n/s/m/l/x 不改变层号。

## `model.mud`

MaixPy 靠它认识模型。

| 字段 | 怎么改 |
|---|---|
| `model_npu` / `model_vnpu` | 分别是 NPU2、NPU1 编译出来的两个 `.axmodel` 文件名 |
| `model_type` | `yolo11` 或 `yolov8`（与你的训练模型一致） |
| `labels` | **顺序必须与 `dataset.yaml` 的 `names` 逐字一致**，数量必须等于 nc |
| `mean` / `scale` | 预处理已编译进模型，这里只是记录，MaixPy 不会再算一遍 |
| `input_cache_flush` | YOLO11 第一层是 CPU 预处理，设 `false` |

改完用 `scripts/lint_deploy_config.py --mud` 对一遍 labels 数量与顺序。

## `rknn-convert.py`

RKNN-Toolkit2 转换模板，已把三件事做对：

1. `accuracy_analysis` 放在 `build()` 之后、**同一个 rknn 对象**上；
2. `mean/std` 写进 config（归一化进模型），与板端"直接喂 uint8"一致；
3. 校准清单、目标芯片、量化算法都走参数，方便换方案重跑。

```bash
# 校准清单：每行一个图片路径
python assets/rknn-convert.py --onnx best_3scale.onnx --dataset dataset.txt \
    --platform rk3588 --out best.rknn --check-image dataset/images/val/0001.jpg
```

⚠️ Toolkit2 版本必须与板端 `librknnrt.so` 配套，否则会出现段错误 / `rknn init failed` 这类没法自解释的错误。
