# YOLO 部署工程手册（决策树 · 速查表 · 量化方法论）

> **读者画像**：有深度学习基础，要的是决策依据、参数对照和排障路径，不是手把手教程。
> **配套**：《YOLO 训练→量化→部署 新手教程》（跟做主线，本手册的 §4/§5 是它的深化）；
> `yolo-to-maixcam2-workflow.md`（路线 M 的已验证实战记录）。
>
> 📌 注：`yolo-to-maixcam2-workflow.md` 是作者的本地实战记录，**未随本仓库发布**；可复用结论已并入 §7 与教程 §6.1。
> 章节独立可检索：§0 总览 → §1 平台速查 → §2 各阶段 checklist → §3 形态与后处理 → §4 量化方法论 → §5 排障树 → §6 性能调优 → §7 已验证案例 → §8 求助模板。

---

## §0 一页总览

### §0.1 硬件 → 路线决策树

```
部署硬件?
├─ 边缘 NPU 板/盒
│   ├─ Rockchip (RK35xx/RV110x)          → R
│   ├─ Axera (AX630C, MaixCAM2)          → M
│   ├─ Horizon (RDK X3/X5)               → H
│   ├─ Allwinner V853 / Amlogic A311D    → V
│   ├─ Hisilicon: DV500系→S(新) | 老NNIE→S(慎,只认Caffe)
│   └─ Sophgo (BM1684/1684X/1688；含 MaixCAM 一代 SG2002) → B
├─ NVIDIA Jetson                          → J (FP16 起步)
├─ 无 NPU ARM 板 / 手机                    → C (NCNN/MNN)
├─ x86 CPU/工控机
│   ├─ Intel                              → X (OpenVINO 首选)
│   └─ AMD                                → X (ONNX Runtime)
└─ 云端/服务器 NVIDIA GPU                  → J 同思路 (TensorRT/ORT-GPU)

NPU 路线(R/M/H/V/S/B) = 必须INT8量化; J/X/C = FP16/FP32起步, INT8为提速选项
```

### §0.2 全流程与交付物

| 阶段 | 输入 | 动作 | 交付物 / 通过标准 |
|------|------|------|------------------|
| 0 选型 | 业务需求(帧率/精度/成本) | 定路线、imgsz、模型规模 | 一页决策记录 |
| 1 数据 | 原始图+标注 | YOLO 格式化、分层划分、**留金标准图** | `dataset/` + 10~20 张场外图 |
| 2 训练 | dataset | 训练 + 曲线/混淆矩阵确认 | `best.pt` + **基线 mAP 记录** |
| 3 导出 | best.pt | 按路线选形态 A/B | onnx 与 pt 输出误差 **< 1e-3** |
| 4 量化 | onnx + 校准集 | PTQ + 逐层精度分析 | 平台模型 + 敏感层清单 |
| 5 部署 | 平台模型 | 上板 + 官方 demo 后处理 | 实时运行 |
| 6 验证 | 金标准图 + 基线 | 三板斧对比 + 阈值重标定 | mAP 掉幅 **<3%**、行为无异常 |

### §0.3 五条铁律（排障前先默背）

1. **预处理四要素**（颜色序 RGB / 归一化 / 布局 / 几何变换 letterbox）恰好做一次，训练=推理
2. **imgsz** 训练 = 导出 = 部署
3. **labels 顺序 = dataset.yaml 的 names = class_id**，三处必须逐字一致
4. **转换工具 ↔ 板端 runtime 版本配套**（路线 R 一票否决级）
5. 量化模型**必须与 FP32 基线同卷对比**后才算验收，不验收不上线

---

## §1 平台速查总表

### 表1：部署基础

| 路线 | 芯片/平台 | 典型板卡 | 转换工具链（跑在哪） | 模型格式 | ONNX 形态 |
|------|----------|---------|--------------------|---------|----------|
| **M** | Axera AX630C（Pulsar2 目标平台要写 `AX620E`） | MaixCAM2 | Pulsar2（Docker，x86 电脑） | `.axmodel` + `.mud` | **B** 三尺度 |
| **R** | RK3588/3566/3568/3576/RV1103/1106 | Orange Pi 5、Luckfox | RKNN-Toolkit2（x86 Linux/Win/WSL） | `.rknn` | **B**（zoo demo）/ A |
| **J** | Jetson Orin/Xavier | Orin Nano/NX | 板上 TensorRT（在板上导） | `.engine` | A |
| **C** | ARM CPU | 树莓派、安卓 | pnnx + ncnn 工具（x86 电脑） | `.param`+`.bin` | A |
| **X** | x86 CPU / Intel GPU | 工控机、服务器 | OpenVINO / ORT（本机） | `.xml`+`.bin` / `.onnx` | A |
| **H** | 旭日 X3 / X5 | RDK X3 / X5 | OpenExplorer（Docker） | `.bin` | 按官方样例 |
| **V** | 全志 V853 / 晶晨 A311D | V853 板、Khadas VIM3 | 厂商 SDK 工具链（Docker） | `.nb` | 按官方样例 |
| **S** | 海思（新 DV500 系） | Hi3516DV500 等 | SVP_NPU SDK | 厂商格式 | 按 SDK |
| **B** | 算能 BM1684/1684X/1688 | SE 系列盒子 | TPU-MLIR（Docker） | `.bmodel` | A / B |

> 📌 表1 只收"当前主推 + 路线卡已验证"的形态，两点补充：
> - **MaixCAM / MaixCAM-Pro（一代）是算能 SG2002**：转换走算能 TPU-MLIR（产物 `.cvimodel`），**不适用路线 M 的 Pulsar2 流程**，请按路线 B 的工具链逻辑理解、并以 Sipeed 官方文档为准。
> - 路线 M 的芯片是 **AX630C**，但 Pulsar2 的 `--target_hardware` **固定写 `AX620E`**（芯片型号 ≠ 平台名）。

### 表2：量化与验证

| 路线 | 量化算法选项 | 校准输入 | 混合精度手段 | 板端 runtime | 精度分析工具 |
|------|-------------|---------|-------------|-------------|-------------|
| **M** | MinMax / Entropy（以版本文档为准） | `.tar` 包（官方建议 20~100 张，50 张是实测下限） | config `layer_configs` 指定层 FP32/FP16 | MaixPy `nn.YOLO11` | `precision_analysis`（EndToEnd） |
| **R** | `normal`(MinMax) / `kl_divergence` / `mmse` | `dataset.txt` 路径清单 | hybrid quantization 两步法 | RKNNLite / C API `librknnrt` | `accuracy_analysis`（逐层余弦） |
| **J** | FP16 默认；INT8 熵校准（ultralytics 自动） | `data` 指定数据集 | TRT per-layer 精度（API/polygraphy） | TensorRT / ultralytics | polygraphy / trtexec dump |
| **C** | `ncnn2table` 的 `method=kl`（另有 aciq 等） | `imagelist.txt` 清单 | 少用（拆模型或整网 FP16） | ncnn API（C++/JNI） | 人工 FP16/INT8 对比 |
| **X** | ORT: MinMax/Entropy/Percentile；NNCF 默认 | 文件夹生成器 | NNCF `ignored_scope` / 部分量化 | OpenVINO / ORT | NNCF 敏感度 / ORT debug |
| **H** | 工具内置（配置选择） | 配置指定目录 | mix 混合模式 | hrt / dnn node（ROS）/ C API | 工具自带报告 |
| **V** | 工具内置 | 工具指定 | 按 SDK | AWNN / 厂商 runtime | 按 SDK |
| **S** | 工具内置 | 工具指定 | 按 SDK | SVP runtime | 按 SDK |
| **B** | INT8 PTQ（`run_calibration`） | 图片目录 → `.table` | 按文档 | sophon-sail / bmrt | TPU-MLIR 工具 |

### 表3：每平台要点速记

**M（MaixCAM 系）** — nn.YOLO11 只吃形态 B（三尺度），单输出报 `Tensors get tensor idx error`；AI-ISP 场景必须备 NPU1 编译的 `model_vnpu`；分类层量化压死 → `layer_configs` 保 cv3 末端三层 FP32（§7 实战）。Docker 命令用 PowerShell 跑（Git Bash 转义路径）。

**R（RKNN）** — 版本配套是第一坑；`mean/std` 写进 config，板端直接喂 uint8；掉精度先用 `accuracy_analysis` 找敏感层再 hybrid 两步法；RK3588 三核要 `CORE_AUTO`/多实例才吃得满；OpenCV 读图记得 BGR→RGB。

**J（Jetson）** — engine 绑定 GPU 架构+TRT 版本，在哪跑在哪导；FP16 起步几乎无损；INT8 用 ultralytics `int8=True data=...` 自动校准；多路流终极形态 DeepStream。

**C（NCNN/MNN）** — 校准表 `mean/norm/pixel` 必须与推理代码预处理逐字一致（第一坑）；ARM INT8 提速看 dotprod 指令；手机绑大核；ncnn examples 的 yolov8/yolo11 demo 是后处理模板。

**X（x86）** — Intel 选 OpenVINO+NNCF（`yolo export format=openvino int8=True data=...` 一行）；INT8 是否加速看 VNNI/AMX 指令，老 CPU 可能白干；形态 A 输出量化后语义不变，后处理代码不用改。

**H（地平线）** — hb_mapper 转 `.bin`，yaml 配置；检测头可能要按官方 yolov5/v8 样例微调；SDK/文档在开发者社区账号体系内。

**V（全志/晶晨）** — Verisilicon 系工具链，ONNX→校准→`.nb`；V853 小算力（imgsz 160~320 + n），A311D 5 TOPS；文档分散在厂商 wiki/SDK。

**S（海思）** — 新项目只选 DV500 系（SVP_NPU，ONNX 直转）；老 NNIE 只认 Caffe，维护成本极高，除非存量产品否则不碰。

**B（算能）** — TPU-MLIR 三步曲：`model_transform` → `run_calibration` → `model_deploy --quantize INT8`；`--processor` 三种芯片互不通用；板端 sophon-sail；注意盒子功耗墙降频。

---

## §2 各阶段 Checklist

### §2.0 选型
- [ ] 路线代号确定（§0.1 树）
- [ ] imgsz 按 §1 表确定，32 的倍数，记入决策记录
- [ ] 模型规模确定（NPU 从 n 起步）
- [ ] 帧率 / 精度 / 内存预算写下来（后面验收对着它）
- [ ] 确认是否必须 INT8（NPU 路线 = 是）

### §2.1 数据
- [ ] YOLO 格式：同名 `.txt`，`class_id cx cy w h` 全归一化
- [ ] class_id 从 0 连续，与 `dataset.yaml` 的 names 一一对应
- [ ] 分层划分：每个类在 val 都有样本
- [ ] **留 10~20 张部署现场金标准图，永不进训练集**
- [ ] 路径无中文、无空格
- [ ] 适量无目标背景图已混入

### §2.2 训练
- [ ] imgsz 与选型一致（此后全程不变）
- [ ] Windows 下 `workers=0~2`
- [ ] 记录基线：val mAP50 / mAP50-95（验收的分母）
- [ ] 混淆矩阵确认每类可用、无明显类间混淆
- [ ] `best.pt` 已备份（云盘/版本库）

### §2.3 导出
- [ ] 形态按 §3.2 表确认（拿不准 A/B 都导）
- [ ] opset 13 起步，按工具链报错调整
- [ ] onnxsim 简化完成
- [ ] ★ **onnx 与 pt 同图输出误差 < 1e-3**（不过不往下走）
- [ ] 记录：输入名、输入 shape、输出名列表（后续配置逐字抄）

### §2.4 量化
- [ ] 校准集 50~500 张：覆盖全类 + 真实部署场景分布
- [ ] 校准预处理与部署一致（mean/std/颜色序——工具配置里那套）
- [ ] per-channel（逐通道）保持开启
- [ ] 跑过逐层精度分析，敏感层已记录
- [ ] 掉幅归档：<1% 收工 / 1~3% 进 §4.5 优化 / >3% 或行为异常进 §4.5 分锅

### §2.5 部署
- [ ] 转换工具 ↔ 板端 runtime 版本配套（R 路线一票否决级）
- [ ] 板端预处理与工具配置一致（U8 直喂 or 代码内 /255，二选一）
- [ ] 几何变换一致：训练 letterbox ↔ 板端 letterbox（或两端都用原生方形输入，不做缩放）
- [ ] labels 顺序与 names 核对过
- [ ] 后处理代码与输出形态匹配（A=只做NMS；B=官方demo的decode+NMS）
- [ ] conf/iou 阈值给了初值（待 §2.6 重标定）

### §2.6 验收
- [ ] 金标准图 FP32 vs INT8 四项对比：检出数量 / 框位置 / 类别 / 置信度差
- [ ] 阈值在**量化后模型**上重标定
- [ ] FPS 实测：纯推理耗时与整帧耗时分开测
- [ ] 烤机 ≥ 数小时：无内存泄漏、无热降频、无偶发崩溃

---

## §3 导出形态与后处理

### §3.1 三形态定义（v8/v11，nc=类别数）

| 形态 | 输出 shape | 图里含什么 | 板端后处理 | 使用方 |
|------|-----------|-----------|-----------|--------|
| **A** 解码单输出 | `[1, 4+nc, N]` | 解码后 xywh + 类别分 | 只需 NMS | J/C/X；R 可选 |
| **B** 三尺度特征 | 3× `[1, 64+nc, H, W]` | 未解码回归分布 + 类别分 | dist2bbox 解码 + NMS（官方 demo 提供） | M/R 及多数 NPU |
| **C** 端到端含NMS | 最终框 | 全部 | 无 | 仅 GPU/x86 可行；NPU 不支持 TopK/动态shape，**不推荐** |

> v5 差异：anchor-based，B 形态通道 = `3×(5+nc)`，A 形态输出 = `[1, 3×(5+nc), N]`。

### §3.2 平台 ↔ 形态适配

| 路线 | 形态 | 原因 |
|------|------|------|
| M | **B** | MaixPy nn.YOLO11 自带 decode，喂 A 直接报错 |
| R | **B**（zoo demo）或 A（自写后处理） | NPU 只跑纯卷积，decode/NMS 留在 CPU 侧 demo |
| J | **A** | GPU 无算子限制，ultralytics runtime 全自动 |
| C / X | **A** | 同上 |
| H/V/S/B | 按官方样例（多为 B 或"魔改头"） | 同 M/R 逻辑 |

### §3.3 形态 B 导出要点（细节见教程 §4.4）

- wrapper 截断检测头：`_, features = model(x)` 返回三尺度
- **输出名以导出后打印/graph 输出为准**，而且**代数不同层号就不同**：YOLO11 的 Detect 在第 23 层（`/model.23/Concat{,1,2}_output_0`），YOLOv8 在第 22 层（`/model.22/Concat_{1,2,3}_output_0`，注意从 `_1` 起）。模型规模 n/s/m/l/x **不改变**层号
- 配置文件、后处理代码中**逐字使用**这些名字（Pulsar2 `output_processors` 报错多半是抄错了）

### §3.4 形态 A 输出点数速算（自查导出是否正确）

| imgsz | A 形态输出 N | B 形态特征图 |
|-------|-------------|-------------|
| 320 | 2100 = 40²+20²+10² | 40/20/10 |
| 416 | 3549 = 52²+26²+13² | 52/26/13 |
| 480 | 4725 = 60²+30²+15² | 60/30/15 |
| 640 | 8400 = 80²+40²+20² | 80/40/20 |

### §3.5 自写后处理最小知识（形态 B，v8/v11 anchor-free）

```
每尺度特征 pred [1, 64+nc, H, W]（NCHW）→ NCHW→NHWC 后逐步：
① split：reg = 前 64 通道，cls = 后 nc 通道
② reg → [H, W, 4, 16] → softmax(-1) → 与 [0..15] 加权求和 → l,t,r,b
   （四个距离，单位=本尺度格宽）
③ anchor 点 = 格中心 (x+0.5, y+0.5)，乘 stride 换算到输入图像素：
   x1 = (x+0.5-l)·stride, y1 = (y+0.5-t)·stride, x2/y2 同理
④ cls 通道是 sigmoid **之前**的原始 logit（v8/v11 无 objectness；B 形态截断在激活函数前），
   后处理要自己 sigmoid——官方 demo 已内置，注意别重复做
⑤ 三尺度结果拼接 → conf 过滤 → NMS（iou_th≈0.45）
⑥ 坐标从输入图映射回原图：反 letterbox（减 pad、除 scale）
```

> 实际工程**优先改官方 demo**（rknn_model_zoo / ncnn examples / MaixPy）而不是从零实现——上面是让你读得懂 demo、改得动参数。

---

## §4 量化方法论（深潜）

### §4.1 INT8 量化原理速览

- 映射关系：`真实值 ≈ scale × (int8值 - zero_point)`；对称量化 zero_point=0，动态范围 `[-127s, +127s]`
- **权重量化基本无损**：训练完就固定，per-channel（每个输出通道一个 scale）精确定标
- **激活量化是误差大头**：分布随输入动态变化，长尾/离群点把 scale 撑大 → 99% 的值挤进少数档位
- **为什么检测头最敏感**：分类分支末端 Conv 直接输出"类别证据"，通道薄（nc 个）、无冗余层缓冲，微小噪声即可翻转类别排序；误差还会经 Concat/残差逐层放大
- 结构性结论：**深且宽的模型（s/m）比薄模型（n）更抗量化**；resize/上采样/depthwise 也常是敏感点

### §4.2 校准（定标）算法对比

| 算法 | 思想 | 速度 | 优点 | 缺点 | 各家叫法 |
|------|------|------|------|------|---------|
| MinMax | 直接取激活 min/max | ★ 最快 | 简单、无截断争议 | 一个离群点毁全局 scale | Pulsar2 `MinMax`；RKNN `normal`；ORT `MinMax` |
| Percentile | 按 99.9% 分位截断 | ★★ | 抗离群点 | 极值被截 | ORT `Percentile`；部分工具的分位选项 |
| KL / Entropy | 最小化量化前后分布的 KL 散度 | ★★★ | 整体分布保真，**业界默认推荐** | 需要足够校准图 | TRT `EntropyCalibrator2`；ncnn `kl`；RKNN `kl_divergence` |
| MMSE / MSE | 逐层最小化输出均方误差 | ★★★★ | 对小模型友好 | 转换时间翻倍级 | RKNN `mmse` |

**选型流程**：从工具默认值起步 → 掉 1~3% 时把 MinMax ↔ KL ↔ MMSE 各转一遍对比（成本只是几次转换）→ 场景含强光/红外等离群值多时优先 Percentile/KL → 换算法无效转 §4.3/§4.4。

### §4.3 校准集构建方法论

**原理**：工具把校准图喂进模型跑前向，收集每层激活分布 → 定 scale。所以校准集 = "激活分布的抽样"，**分布代表性 > 数量**。

| 维度 | 规则 |
|------|------|
| 数量 | 50 张是实测可用下限（M 路线实战即 50），100~300 常用，>500 收益递减 |
| 来源 | val 集 + 部署现场实拍；**禁止**与部署无关的网图/公开数据集 |
| 覆盖 | 每类 ≥3~5 张；光照/角度/距离多样；24h 部署就昼夜都放 |
| 场景 | 部署会遇到的极端场景（逆光/夜间/雨雾）**要包含**，不会出现的脏数据排除 |
| 标注 | **不需要标注**！只要图片本身 |
| 管理 | 固定随机种子、版本化——换校准集 = 换模型，必须重新验收 |

### §4.4 混合精度：各平台操作对照

| 路线 | 操作 | 要点 |
|------|------|------|
| M (Pulsar2) | config 顶层加 `"layer_configs": [{"layer_name": "...", "data_type": "FP32"}]` | layer_name 必须是 ONNX 确切节点名；实战保 cv3 末端三层（§7）；层号随代数变（YOLO11 是 `model.23`，YOLOv8 是 `model.22`） |
| R (RKNN) | hybrid quantization 两步法：step1 生成量化配置文件 → 把敏感层标记为 FP16 → step2 重编译 | 层名以 step1 输出的配置为准 |
| J (TensorRT) | builder API 设 per-layer precision，或 polygraphy 指定层精度 | 通常全网 FP16 已足够，先做 FP16 基线 |
| X (OpenVINO/NNCF) | `nncf.quantize(model, ds, ignored_scope=...)` 跳过敏感子图；另有 accuracy-aware 自动模式 | 兜底方案：整网 FP16 推理 |
| X (ORT) | `quantize_static(..., nodes_to_exclude=[...])` | 节点名用 Netron 查 |
| C (ncnn) | 无逐层开关；手段=换校准算法/扩校准集/整网 fp16 运行 | 真要混合就拆两个模型级联（少用） |
| H/V/S/B | H 用 mix 模式；其余按工具文档 | 以官方移植样例为准 |

**敏感层怎么选**：① 精度分析报告里相似度塌陷层（§4.6）；② 经验优先名单：分类分支末端 Conv（`cv3.x.2/Conv`）> 回归分支末端 > 最后一级 Concat 后首个 Conv > depthwise/resize 附近；**保层 ≤10 个**，保太多速度退化成 FP16 就失去意义。

### §4.5 掉精度排查决策树（先分锅，再动手）

```
量化后效果差
│
├─ 第0步 分锅①：FP32 ONNX 在电脑上跑同一批金标准图
│   ├─ ONNX 本身就不对 → 锅在【导出】：回查 opset/simplify/imgsz/形态
│   └─ ONNX 正常 ↓
├─ 第1步 分锅②：有条件时跑"量化模拟"输出对比板端实机输出
│   ├─ 只有板端不对 → 锅在【部署】：预处理四要素 / labels 顺序 /
│   │                  形态-后处理匹配 / runtime 版本（铁律 1·3·4）
│   └─ 模拟也差 → 锅在【量化】↓
├─ 第2步 按掉幅与症状分流
│   ├─ <1%            → 收工
│   ├─ 1~3%           → 换校准算法（§4.2）→ 重构校准集（§4.3）→ 混合精度（§4.4）
│   └─ >3% 或行为异常  → 对照 §4.8 症状表定位 → 混合精度保敏感层
│                       → 仍不行：n→s / 加大 imgsz 重训（小模型更量化敏感）
│                       → 最后手段 QAT（§4.7）
```

> 分锅的意义：三种锅（导出/部署/量化）的修法完全不同，不先定位就乱试，一晚上就没了。

### §4.6 精度分析工具速查

| 路线 | 工具/入口 | 看什么 |
|------|----------|--------|
| M | config `"precision_analysis": true`（EndToEnd） | 转换产物中的端到端误差报告 |
| R | `rknn.accuracy_analysis(inputs=[...], output_dir=...)` | snapshot 报告：每层量化 vs FP 的**余弦相似度**，盯 <0.9 的层 |
| J | polygraphy（`polygraphy debug` / `run --trt`）；`trtexec --dumpLayerInfo --profilingVerbosity=detailed` | 逐层输出对比、层精度分布 |
| X (OV) | NNCF 敏感度实验 / accuracy-aware 量化日志 | ignored_scope 候选清单 |
| X (ORT) | quantize_debug / QDQ 节点统计 | 量化节点插入点误差 |

**通用土法（报告看不懂时）——敏感层二分**：把可疑层集合一半标 FP16 重转 → 恢复则继续二分另一半，几次收敛到"真凶"。成本只是几次转换，简单粗暴有效。

### §4.7 QAT：什么时候上

| 问题 | 答案 |
|------|------|
| 原理 | 训练时插入伪量化节点：前向模拟 INT8 舍入，反向用直通估计器（STE），让权重主动适应量化噪声 |
| 何时上 | PTQ + 混合精度 + 换大模型后仍掉 >2~3%，且能接受 1~2 天重训 |
| 工具 | PyTorch 原生/torchao QAT；RKNN-Toolkit2 提供 QAT 导出支持；各 NPU 厂商新版工具链陆续支持"QAT 导出的 ONNX 直编译"（查你的版本文档） |
| 注意 | QAT 后编译/校准流程不变，仍要过一遍 §2.6 验收 |
| 结论 | **90% 的边缘部署用不到 QAT**，别一开始就上 |

### §4.8 症状速查表

| 症状 | 最可能原因 | 第一动作 |
|------|-----------|---------|
| **全识别成同一类** | 分类分支末端 Conv 被 INT8 压死 | 该层保 FP32（§4.4）；MaixCAM 实战见 §7 |
| 置信度集体跳水（0.9→0.3） | 校准分布 ≠ 部署分布 | 重构校准集（§4.3） |
| 置信度虚高、误框多 | 同上，或阈值未重标定 | 阈值重标定 + 校准集 |
| 漏检明显变多 | 阈值漂移 / 校准集缺该场景 | 降 conf 试探 + 补场景校准图 |
| 框位置偏移、抖动 | 回归分支量化敏感 | 回归末端层混合精度 |
| 整图完全无检出 | 形态与后处理不匹配 / names 顺序错 | 核对 §3.2 与 names |
| 只在部分场景崩 | 校准集缺该类场景 | 补图重转 |
| FP32 对、INT8 错且规律性强 | 预处理不一致（重复/缺失归一化、BGR、letterbox vs 拉伸） | 铁律 1 四要素自查 |
| 模型加载就失败 | 工具↔runtime 版本不配套 | 铁律 4 |

---

## §5 排障决策树

### §5.1 转换阶段失败（电脑上工具报错）

```
报错类型?
├─ "unsupported op / 算子不支持"
│   ├─ 降 opset（17→13 重导）
│   ├─ 升级工具链版本
│   ├─ 是不是 C 形态（图里含 NMS/TopK）？→ 换 A/B 形态
│   ├─ 先过 onnxsim 再转
│   └─ 特定算子：查官方移植样例有没有"魔改头"写法
├─ "shape / dimension mismatch"
│   ├─ 配置的 input_shapes 与 onnx 实际不符（Netron 核对）
│   ├─ 导出 imgsz ≠ 配置 imgsz
│   └─ dynamic/batch 问题 → 固定 batch=1 重导
├─ 转换机 "内存不足 / OOM"
│   ├─ 校准图先 resize 到输入尺寸再打包
│   ├─ 校准数量减半
│   └─ 换 16G+ 内存机器
└─ 校准阶段报错
    ├─ 打包格式不对（tar vs txt 清单 vs 目录，按 §1 表2）
    ├─ 路径在容器内不对（挂载点 /data 下再核对一遍）
    └─ 损坏/0字节/冷门格式图片 → 统一转 jpg
```

### §5.2 精度异常（能跑但结果不对）

**快速三查**（90% 命中）：① FP32 ONNX 在电脑上正常吗？② 预处理四要素（铁律1）？③ labels 顺序 + 形态-后处理匹配（铁律3）？
三查不中 → 进 **§4.5 排查树** 分锅定位 → 症状对照 **§4.8**。

### §5.3 性能异常（FPS 低）

```
先分锅：纯推理耗时 vs 整帧耗时
├─ 纯推理就慢
│   ├─ NPU 模型真的量化成 INT8 了吗（FP 模型在 NPU 上龟速）
│   ├─ imgsz 降一档（收益平方级）
│   ├─ 模型降档（s→n）
│   ├─ 平台特性没开：MaixPy dual_buff / RKNN 三核绑定 / TRT FP16 / ncnn 绑大核
│   └─ 输入通路：相机直出目标尺寸、zero-copy 喂 NPU
└─ 纯推理快、整帧慢
    ├─ 后处理在 Python 里跑？→ 换官方 C demo
    ├─ 显示/画框/编码占大头 → 异步化、降显示帧率
    ├─ CPU 被其他进程抢占（top 看看）
    └─ USB 相机取流慢 → 换 MIPI / 降取流分辨率
```

> FPS 测量规范：warmup 10 帧后取 ≥100 帧平均；分别报"纯推理 / 后处理 / 整帧"三个数。

### §5.4 运行时失败（板端加载/崩溃）

```
├─ 模型加载就失败
│   ├─ 工具↔runtime 版本不配套（第一嫌疑，铁律4）
│   ├─ 传输损坏：md5sum 两边对比
│   ├─ 目标芯片写错：target_hardware / --processor / --march 与实际芯片不符
│   └─ 板端内存不足：关掉其他大进程再试
├─ 推理时段错误
│   ├─ 输入 buffer 类型/尺寸不对（U8 直喂 vs float）
│   ├─ 输出读取越界：形态与后处理不匹配（A/B 搞反）
│   └─ 多线程并发调用非线程安全接口 → 加锁或单线程
└─ 间歇性 / 烤机后崩溃
    ├─ 供电不足（NPU 峰值电流，换电源线/减少外设）
    ├─ 过热降频 → 散热
    └─ 内存泄漏 → 长时间烤机 + 监控 top 复现
```

---

## §6 性能调优速查（按性价比排序）

| 序 | 手段 | 收益量级 | 成本 |
|----|------|---------|------|
| 1 | 降 imgsz（640→320） | **~4×**（平方级） | 重训一次 |
| 2 | 模型降档（s→n / 砍通道） | 2~4× | 重训 |
| 3 | 确认真在跑 INT8（NPU 全速的唯一形态） | 2~4× | 0 |
| 4 | 平台流水线特性：dual_buff / 多核绑定 / zero-copy / TRT FP16 | 1.2~2× | 改配置 |
| 5 | 输入通路：相机直出目标尺寸、免软件缩放 | 视情况 | 半天 |
| 6 | 后处理 C 化 / 独立线程 | 整帧 20~50% | 半天 |
| 7 | 剪枝 / 蒸馏 | 1.5~2× | 高（进阶，配合重训） |

> 决策提示：1~3 是"免费午餐"先用尽；4~6 是工程优化；7 是算法侧深水区。

---

## §7 已验证基线案例：MaixCAM2（路线 M）

> 完整实战记录见 `yolo-to-maixcam2-workflow.md`。此处是**参数快照**——换任何新平台时，用它当"应该长这样"的对照样板。

| 环节 | 实测参数 | 备注 |
|------|---------|------|
| 任务 | YOLO11n，9 类，imgsz=320 | 73 = 64 + 9 通道 |
| 训练 | epochs 200 / batch 16 / RTX 3050Ti 4GB 可训 | workers=1（Windows） |
| 导出 | 形态 B 三尺度 `[1,73,40,40]×3`，opset 19，onnxsim 简化 | 输出名 `/model.23/Concat*_output_*` |
| 量化 | Pulsar2（镜像 6.0）；`--target_hardware AX620E`；MinMax；校准 50 张 tar 包；precision_analysis=EndToEnd | 校准 mean 0 / std 255；芯片是 AX630C 但平台名写 AX620E |
| 部署 | `model_npu.axmodel`（NPU2 全速）+ `model_vnpu.axmodel`（NPU1，AI-ISP 用）+ `.mud` + MaixPy `nn.YOLO11(dual_buff=True)` | |
| 踩坑① | **全识别成同一类** → 分类层被 INT8 压死 → `layer_configs` 把 `cv3.x.2/Conv` 三层强制 FP32 解决 | 量化敏感层经典案例 |
| 踩坑② | `ai isp is on` 报错 → 夜视 AI-ISP 走 VNPU 通路 → 必须备 NPU1 编译的 model_vnpu | |
| 踩坑③ | 喂单输出 `[1,13,2100]` 报 `Tensors get tensor idx error` → nn.YOLO11 只吃形态 B | |

---

## §8 求助信息模板

发论坛 / 提 issue / 问 AI 之前，把这张表填全（附上**报错日志全文**和**一张能复现问题的图**）：

```
【训练】ultralytics x.x，YOLO11n，imgsz=320，nc=9，val mAP50-95=0.87
【导出】形态 B（三尺度），opset 19，输出名：/model.23/Concat_output_0 ...
        自检结果：onnx vs pt 误差 3e-5（通过）
【部署】路线 R（RK3588），RKNN-Toolkit2 2.x.x（x86 Ubuntu），板端 librknnrt 2.x.x
【量化】w8a8，normal(MinMax)，per-channel，校准 50 张（val 抽取，含全部类别）
【症状】板端所有目标识别成同一类；FP32 ONNX 电脑上正常；
        accuracy_analysis 显示 /model.23/cv3.0/cv3.0.2/Conv 相似度 0.71
【已试】换 KL 校准无改善；确认 labels 顺序与 names 一致
【自查】铁律1 预处理一致✓ / 铁律2 imgsz一致✓ / 铁律3 labels✓ / 铁律4 版本配套✓
```

> "自查"一行对应 §0.3 五条铁律——把这行写上，回答者能直接跳过 80% 的低级嫌疑。

---

## 全书完：三份文档的关系

```
新手教程（跟做主线，含路线卡）
   │ 遇到问题/查参数
   ▼
工程手册（本文件：决策树·速查表·量化方法论·排障）
   │ 路线 M 的出处
   ▼
yolo-to-maixcam2-workflow.md（已验证实战记录 / §7 基线快照）
```