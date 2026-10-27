---
name: yolo-deploy
description: YOLO 训练→量化→部署 全平台手册：选路线、导出 ONNX A/B 形态、INT8 量化、上板（MaixCAM2 / Rockchip RKNN / Jetson / 地平线 / 全志 / 海思 / 算能）、掉精度与 FPS 排障。训练或部署 YOLO 检测模型到边缘硬件前先加载。
---

# YOLO 部署（全平台）

主线：**选路线 → 数据 → 训练 → 导出 ONNX → 量化 → 上板 → 验收**。

## 五条铁律（排障前先默背）

1. 预处理四要素**恰好做一次**、训练=推理：颜色序（RGB）· 归一化 · 布局（NCHW/NHWC）· 几何变换（letterbox vs 拉伸）
2. **imgsz 训练 = 导出 = 部署**（32 的倍数）
3. `labels` 顺序 = `dataset.yaml` 的 `names` = `class_id`
4. 转换工具 ↔ 板端 runtime 版本配套（RKNN 是一票否决级）
5. 量化模型必须与 FP32 基线同卷对比后才算验收（掉幅 < 3%）

## 选路线（30 秒版）

| 硬件 | 路线 | 工具链 | 需要形态 |
|---|---|---|---|
| MaixCAM2（AX630C） | M | Pulsar2（Docker，`--target_hardware AX620E`） | B |
| Rockchip RK3588/3566/3576/RV1106 | R | RKNN-Toolkit2 | B（配 zoo demo）或 A |
| Jetson Orin/Xavier | J | 板上 TensorRT | A |
| ARM CPU / 手机 | C | NCNN / MNN | A |
| x86 PC / 服务器 | X | OpenVINO / ONNX Runtime | A |
| 地平线 RDK X3/X5 | H | OpenExplorer（`hb_mapper`） | 按官方样例 |
| 全志 / 晶晨 | V | 厂商 SDK | 按官方样例 |
| 海思 | S | SVP_NPU（DV500 新）/ NNIE+Caffe（旧，慎） | 按 SDK |
| 算能 SE 盒子、**MaixCAM 一代（SG2002）** | B | TPU-MLIR | A / B |

⚠️ 两个最常见的选路错误：

- **MaixCAM / MaixCAM-Pro 一代 ≠ 路线 M**：它是算能 SG2002，走算能 TPU-MLIR（产物 `.cvimodel`），不是 Pulsar2。
- **MaixCAM2 的 Pulsar2 目标平台写 `AX620E`**，不是 `AX630C`（芯片型号 ≠ 平台名）。

## 三种 ONNX 形态

- **A 解码单输出** `[1, 4+nc, N]`：只剩 NMS → Jetson / x86 / NCNN
- **B 三尺度** 3×`[1, 64+nc, H, W]`：需要 decode+NMS（抄官方 demo）→ MaixPy / RKNN / 多数 NPU
- **C 端到端含 NMS**：NPU 基本不支持 TopK/动态 shape，不要依赖

输出节点名经验值：**YOLO11** `/model.23/Concat_output_0`、`Concat_1`、`Concat_2`；**YOLOv8** `/model.22/Concat_1`、`Concat_2`、`Concat_3`（从 `_1` 起）。永远以导出后打印结果为准。

## 文档地图（按需读，不要一次全读）

- `references/tutorial.md` —— 新手跟做主线：数据 / 训练 / 导出 / 量化 / 九张路线卡 / 验收
- `references/handbook.md` —— 决策树、速查表、量化方法论、排障树（**遇到问题先查这本**）
- `references/route-status.md` —— 每条路线的「已验证 / 仅二手资料」标签与权威文档入口
- `references/changelog.md` —— 本版相对原始文档改了什么、依据是什么

## 先跑脚本，再相信文档

- `scripts/lint_deploy_config.py` —— 拿 ONNX 的真实输入/输出名核对 `config.json`、`.mud`、`dataset.yaml`；把「层名写错 / labels 顺序错 / 输出数量不匹配」当场抓出来（比读十条注意事项管用）
- `scripts/check_onnx_vs_pt.py` —— 同一张图跑 `.pt` 与 `.onnx`，最大误差 < 1e-3 才算导出成功
- `scripts/split_dataset.py` —— 分层划分（每类都进 val）+ 容忍无 txt 的背景图

## 模板

`assets/pulsar2-maixcam2.json`、`assets/rknn-convert.py`、`assets/model.mud` —— 改类名 / 尺寸即可用；里面所有 `★` 都必须替换，否则跑不通。
