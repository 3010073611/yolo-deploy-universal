# 修订记录

## v1.1.0（2026-10-27）— 开源前系统性校对

原始材料是两份个人笔记（教程 1224 行 + 手册 457 行）。开源前逐条对照官方文档校对，改动分三类：**会直接导致跟做失败**、**自相矛盾/违反自身铁律**、**影响精度的遗漏**。每条都注明依据，便于复核。

### 一、会直接导致跟做失败（必修）

| # | 位置 | 原文 | 改为 | 依据 |
|---|---|---|---|---|
| 1 | 教程 §6.1、路线总表、§1.2 决策树；手册 §0.1、表1 | MaixCAM（一代）= AX620E，两代都用 Pulsar2 | MaixCAM 一代是**算能 SG2002**，走算能 TPU-MLIR（`.cvimodel`）；路线 M 只针对 MaixCAM2 | [Sipeed MaixCAM 转换](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html)、[MaixCAM 硬件页](https://wiki.sipeed.com/hardware/zh/maixcam/maixcam.html) |
| 2 | 教程 §6.1 转换命令 | `--target_hardware AX630C`，且注释"AX620E 设备换成 AX620E" | **固定 `AX620E`**（MaixCAM2 芯片是 AX630C，平台名不同），并明确"芯片型号 ≠ 平台名" | [Sipeed MaixCAM2 转换](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam2.html)：「固定写 AX620E」 |
| 3 | 教程 §6.9 TPU-MLIR | `model_transform --model_file best.onnx ... --processor bm1684x` | ONNX 用 `--model_def`；`--processor` 只属于 `model_deploy`；补 `run_calibration.py` / `model_deploy.py` 后缀 | TPU-MLIR 官方参数（`--model_def` 指定 onnx/tflite/prototxt；`--model_file` 给 Caffe 权重） |
| 4 | 教程 §6.4 NCNN | 产物写成 `model.param + model.bin`，后续命令也用这两个名字 | 实际是 `model.ncnn.param` / `model.ncnn.bin`（+ `metadata.yaml`） | ultralytics `utils/export/ncnn.py` |
| 5 | 教程 §6.4 MNN | `MNNConvert --modelFile best.onnx --fp16`，量化工具写"MNN-Quant" | 补 `-f ONNX` 与 `--MNNModel`；量化改用仓库自带 `tools/quantization`；优先 `yolo export format=mnn` | MNNConvert 参数要求；MNN 仓库目录结构 |
| 6 | 教程 §6.1 config.json | `tensor_format/src_format = BGR`，与 MUD 的 `input_type = rgb` 冲突，缺 `tensor_layout`/`csc_mode` | 统一为 **RGB + NoCSC**，补 `tensor_layout: NCHW`，`compiler.check` 用官方推荐值 3/CheckOutput | Sipeed 官方 config 示例 |
| 7 | 教程 §7.2 | 标题要求"在量化模型上调阈值"，代码却在 `best.pt` 上扫 | 拆成两种情况：形态 A 用 `YOLO("best_int8.onnx").val(...)`；形态 B 只能在板上扫 | 与本节结论自洽性要求 |
| 8 | 教程 §6.2 | `accuracy_analysis` 注释"或单独跑" | 明确必须在同一 `rknn` 对象 `build()` 之后调用 | RKNN-Toolkit2 API 行为（依赖已构建模型） |
| 9 | 教程 §6.1 Docker | `docker run ... 镜像名 -c "pulsar2 build ..."` | 改为官方推荐的"起交互容器再执行"，并说明 `-c` 只在入口是 bash 时成立 | Sipeed 官方用法 |

### 二、自相矛盾 / 违反自身铁律

| # | 位置 | 问题 | 处理 |
|---|---|---|---|
| 10 | 教程 §3.3 vs §6.3/§6.9 | 训练 320，示例却 640，违反"imgsz 三处一致" | 示例统一 320，并加醒目注释 |
| 11 | 教程 §2.2 `split.py` | 标题"分层划分"，脚本纯随机；遇到无 txt 背景图直接崩 | 替换为指向 `scripts/split_dataset.py`（分层 + 容忍背景图 + 打印分布） |
| 12 | 教程 §4.2 | "R 选 B 省 NMS 代码"与"形态 A 只需 NMS"矛盾 | 改写为"B 需 decode+NMS，但可直接抄官方 demo" |
| 13 | 教程 §4.4/§6.1/§6.2；手册 §3.3/§4.4 | 硬编码 `/model.23` 只适用 YOLO11；且称"模型规模会改层号" | 补 YOLOv8 是 `/model.22/Concat_1..3`（从 `_1` 起）；说明改层号的是**代数**不是规模 |
| 14 | 教程 §6.1 MUD | config 的 `std=255` 与 MUD 的 `scale=1/255` 并存，读者会以为归一化做两次 | 补官方说明：MUD 里只是记录，预处理已编译进模型 |
| 15 | 教程 附录A、§8.3 | 与手册表1/表2、§8 重复且已漂移（opset 13 vs 19、workers 0 vs 1） | 改为指回手册（单一事实源），删掉重复表格与模板 |
| 16 | 教程 §3.4 | `val_batch_pred.jpg` | 实际文件名是 `val_batch0_pred.jpg` |
| 17 | 手册 表2 | 校准集"50+ 张" | 官方建议 20~100 张，50 是实测下限 |

### 三、影响精度的实质遗漏

| # | 内容 | 处理 |
|---|---|---|
| 18 | **letterbox**：训练用等比缩放+padding，板上示例全是 `cv2.resize` 拉伸，但铁律里没有它 | 升级为"预处理**四**要素"，教程 §1.3 增加专门说明与判定方法；手册铁律 1、§2.5 checklist、§5.2 三查同步更新 |
| 19 | 验收第三板斧（板端跑全量 val 算 mAP）只有一句"可选" | 补具体三步做法与"掉幅 <3%"判定 |
| 20 | MaixCAM2 被一律建议 320 | 补注：3.2 TOPS@INT8，官方 YOLO11n@640 约 113 FPS，320 只是保守起点 |

### 四、其他

- 删除对未开源文件 `yolo-to-maixcam2-workflow.md` 的悬空引用说明，改为"未随仓库发布，结论已并入"。
- 附录B 补 Sipeed 两代设备转换文档入口（避免读者照着旧博客走错工具链）。
- 新增 `references/route-status.md`：给 9 条路线打「实测 / 文档对齐 / 仅索引」标签——原先所有路线写得一样笃定，是这份资料最大的隐性风险。
- 新增 3 个脚本与 3 个模板，把"抄错名字、labels 顺序错"这类高频错误变成机器校验。

### 未改（有意保留）

- 原文的行文风格、节奏与"工程经验型"判断（如"n 比 s 对量化更敏感""50 张是下限"）保留原样，只在能被官方文档证伪处修改。
- 路线 H / V / S / B 仍偏薄，但已明确标注为「仅索引 / 未实测」，不再伪装成操作手册。补齐需要真实硬件验证，留给后续贡献者。
