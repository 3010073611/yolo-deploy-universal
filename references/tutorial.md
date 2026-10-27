# YOLO 训练→量化→部署 新手教程（全平台通用版）

> 这份教程带你从零走完 **数据准备 → 训练 → 导出 ONNX → 量化 → 平台转换 → 上板部署 → 验证调优** 的完整流程。
> 它不绑定任何一家硬件：MaixCAM、Rockchip、Jetson、树莓派/手机、x86 工控机、地平线、海思、全志、晶晨、算能……都只是最后"选一张路线卡"的差别。
>
> 配套文档：
> - 《YOLO 部署工程手册》（决策树 + 速查表 + 量化方法论 + 排障），教程里遇到问题时会指向它
> - `yolo-to-maixcam2-workflow.md`（MaixCAM2 已验证实战记录，本教程路线 M 的原始出处）
>
> 📌 注：`yolo-to-maixcam2-workflow.md` 是作者的本地实战记录，**未随本仓库发布**；其中可复用的结论已经并入 §6.1 与《工程手册》§7。

---

## 前言：这份文档怎么用

**适用范围**
- 训练框架：以 **ultralytics**（YOLOv5 / YOLOv8 / YOLO11）为主线。其他框架（YOLOX、PP-YOLOE 等）流程思想完全相同，只是导出和后处理细节有差异。
- 部署目标：**任何**推理硬件。你不需要全读，只需要走通用主线，最后读属于你硬件的那一张"路线卡"。

**阅读路径（新手照着走）**

```
第1章 选路线（确定你的硬件走哪条路线代号）
  ↓
第2章 数据准备 ─┐
第3章 训练      ├─ 通用主线，所有平台都一样
第4章 导出ONNX ─┤
第5章 量化准备 ─┘
  ↓
第6章 路线卡（只读你自己的那一张！）
  ↓
第7章 验证与调优
第8章 卡住了怎么办
```

**两个约定**
1. 命令默认 Windows PowerShell，Linux/macOS 差异会标注。
2. 教程教你"怎么做+为什么"；参数细节、对比表、排障树都在《工程手册》里，本文会说"详见手册 §x"。

---

## 路线代号总表

先混个眼熟，第1章会教你判断自己属于哪条路线。

> ⚠️ **注意：MaixCAM / MaixCAM-Pro（一代）不属于路线 M。** 它是算能 SG2002，转换走算能 TPU-MLIR（产物 `.cvimodel` + `.mud`），和路线 B 同一套工具链逻辑。本教程的路线 M 只针对 **MaixCAM2（AX630C）**。
> 参考：MaixCAM 用 [TPU-MLIR 转换](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html)，MaixCAM2 用 [Pulsar2 转换](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam2.html)。

| 代号 | 硬件类型 | 典型设备 | 转换工具 | 最终模型格式 |
|------|----------|----------|----------|--------------|
| **M** | 爱芯元智 NPU | **MaixCAM2**（AX630C 芯片；Pulsar2 里平台名要写 `AX620E`） | Pulsar2（Docker） | `.axmodel` + `.mud` |
| **R** | 瑞芯微 NPU | RK3588 / RK3566 / RK3568 / RK3576 / Luckfox RV1103/1106 | RKNN-Toolkit2 | `.rknn` |
| **J** | NVIDIA Jetson | Orin Nano / Orin NX / Xavier | 板上 TensorRT | `.engine` |
| **C** | 无 NPU 的 ARM 板 / 手机 | 树莓派、安卓手机、全志 CPU 板 | NCNN / MNN | `.param`+`.bin`（NCNN） |
| **X** | x86 PC / 服务器 / 工控机 | Intel/AMD CPU、Intel 核显 | OpenVINO / ONNX Runtime | `.xml`+`.bin` 或 `.onnx` |
| **H** | 地平线 BPU | RDK X3 / RDK X5（旭日系） | OpenExplorer（Docker） | `.bin` |
| **V** | 全志 / 晶晨 NPU | 全志 V853、Khadas VIM3（A311D） | 厂商 SDK 工具链 | `.nb` |
| **S** | 海思 | Hi3516DV500 / Hi3519DV500（新）；旧款 Hi3516/3519（NNIE） | SVP_NPU SDK（新）/ RuyiStudio（旧） | 厂商格式 |
| **B** | 算能 TPU | SE 系列盒子（BM1684 / BM1688） | TPU-MLIR（Docker） | `.bmodel` |

> **B / X / J 是"舒服路线"**：转换工具链友好、坑少。**M / R / H / V / S 是"NPU 路线"**：必须 INT8 量化，流程长一点，但这就是本教程的主战场。

---

# 第1章 选路线：先搞清楚你在哪，要去哪

## 1.1 全流程鸟瞰

不管什么硬件，YOLO 落地永远是这七步：

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────────┐   ┌─────────┐   ┌─────────┐
│ 0 选型    │ → │ 1 数据   │ → │ 2 训练   │ → │ 3 导出    │ → │ 4 量化+编译   │ → │ 5 上板   │ → │ 6 验证   │
│ 定硬件/尺寸│   │ 标注/划分 │   │ 得 best.pt│   │ 得 .onnx │   │ 得平台模型    │   │ 部署运行 │   │ 调优    │
└─────────┘   └─────────┘   └─────────┘   └──────────┘   └──────────────┘   └─────────┘   └─────────┘
     ↑                                                                                  │
     └────────────── 量化掉精度 / 帧率不达标 → 回头改尺寸、改模型、改量化 ←──────────────┘
```

两个最重要的认知，先刻在脑子里：

1. **硬件决定一切上游参数。** 输入尺寸（imgsz）、模型规模（n/s/m/l/x）、精度（INT8/FP16）都由你的目标硬件算力决定。**先选硬件，再训练**，否则训完发现板子跑不动，就得重训。
2. **部署三件套：预处理 + 模型文件 + 后处理。** 任何一个环节和训练时不一致（颜色序、归一化、输出解释方式），模型就会"变傻"。90% 的部署 bug 不在模型，在这三件套的对不上。

## 1.2 决策树：我的硬件走哪条路线？

```
Q1: 你的部署目标硬件是什么？
│
├─ 边缘 NPU 开发板 / 盒子（有专门 AI 加速芯片）
│    │
│    └─ Q2: 芯片厂商是谁？（看板子丝印 / 商品页参数表）
│         ├─ 瑞芯微 Rockchip（RK35xx、RV1103/1106）        → 路线 R（§6.2）
│         ├─ 爱芯元智 Axera（AX630C，如 MaixCAM2）          → 路线 M（§6.1）
│         ├─ 地平线 Horizon（RDK X3/X5，旭日系列）          → 路线 H（§6.6）
│         ├─ 全志 Allwinner（V853 等）                     → 路线 V（§6.7）
│         ├─ 晶晨 Amlogic（A311D，如 Khadas VIM3）         → 路线 V（§6.7）
│         ├─ 海思 Hisilicon
│         │    ├─ 新款 DV500 系列（支持 ONNX 直转）        → 路线 S（§6.8）
│         │    └─ 旧款 Hi3516/3519（NNIE，只认 Caffe）     → 路线 S（§6.8，避坑警告）
│         └─ 算能 Sophgo（BM1684/1688、以及 MaixCAM 一代的 SG2002）→ 路线 B（§6.9）
│
├─ NVIDIA Jetson 系列（Orin/Xavier）
│    └─ → 路线 J（§6.3）：FP16 起步，几乎无痛
│
├─ 没有 NPU 的 ARM 板 / 手机（树莓派、安卓、RK 板子跑 CPU）
│    └─ → 路线 C（§6.4）：NCNN / MNN
│
├─ x86 台式机 / 笔记本 / 工控机 / 服务器
│    └─ Q3: 主要用 CPU 还是显卡？
│         ├─ Intel CPU / 核显 / Intel 独显                → 路线 X（§6.5），首选 OpenVINO
│         └─ NVIDIA 显卡（非 Jetson）                     → 路线 J 同款思路：TensorRT/ONNX Runtime GPU
│
└─ 还没买硬件，只是想学习
     └─ → 路线 X（在自己的电脑上就能走完全流程），以后买了板子再换路线卡
```

选好路线后，先跳到你的路线卡（第6章）开头看一眼"**你需要准备的**"和"**产物**"，再回来从第2章开始干。这样你知道自己最终要交出什么。

## 1.3 三个跨平台通用概念（后面所有章节都建立在这上面）

### 概念一：部署三件套

任何平台推理 = **预处理 → 模型前向 → 后处理**，你必须在每个平台把这三样和训练时对齐：

| 环节 | 训练时（ultralytics） | 部署时怎么对齐 |
|------|----------------------|----------------|
| 颜色序 | **RGB** | OpenCV 读图默认是 BGR！要么代码里 `cvtColor` 转 RGB，要么在工具配置里声明 BGR 输入 |
| 布局 | CHW（通道在前） | NPU 工具常用 HWC（通道在后），工具配置里叫 `layout` / `NHWC`，一般工具自动处理，但要确认 |
| 几何变换 | **letterbox**（等比缩放 + 灰边填充成方形） | 板端直接 `cv2.resize(img, (320,320))` 是**拉伸变形**，和训练的 letterbox 不是一回事 → 框会整体偏移、小目标变形。要么板端照抄 letterbox 再反算坐标，要么用原生方形图（相机直出 320×320）避开缩放 |
| 归一化 | `x / 255`，mean=0, std=1 | **只能做一次！** 要么写进模型（NPU 工具配置 mean/std，板端直接喂 0~255 的 uint8 图），要么写在板端代码里。两边都做或都不做 = 白屏/乱识别 |

> 🔑 **铁律：归一化和颜色序，"恰好一次、训练推理一致"。** 记不住就抄这句话，排查 bug 时第一个检查它（颜色序 + 归一化 + 几何变换，一共三件事）。

> 📐 **为什么把 letterbox 单独拎出来讲？** ultralytics 训练/验证默认用 letterbox：等比缩放到 320 长边、剩余部分填灰边。而板上绝大多数示例代码都是 `cv2.resize(img, (320,320))` 直接拉伸。同一张 4:3 的图，两种方式喂给模型的内容完全不同——这是"FP32 电脑上好好的，上板精度就是差一截"最常见的原因之一，而它不在上面四要素的检查清单里。
> 判定方法：用一张 4:3 的图，电脑上跑 letterbox 版预处理，板上跑拉伸版，比较同一目标的框位置。差得多就是它。

### 概念二：ONNX 输出三形态

训练出的 `.pt` 导出成 `.onnx` 时，输出头有三种裁剪方式，**你的路线卡决定你要哪一种**：

| 形态 | 输出 | 说明 | 谁在用 |
|------|------|------|--------|
| **A：解码单输出** | `[1, 4+nc, 8400]`（640 输入；320 输入则是 `[1, 4+nc, 2100]`） | 模型输出解码后的框坐标+类别分数，**后处理只需写 NMS** | Jetson / x86 / NCNN 裸用（自己写后处理的人） |
| **B：三尺度特征图** | 3 个 `[1, 64+nc, H/8, W/8]`、`[1, 64+nc, H/16, W/16]`、`[1, 64+nc, H/32, W/32]` | 模型在检测头最后几个 Conv 后就截断，**解码和 NMS 都由部署框架的 C/C++ 代码做** | MaixPy（nn.YOLO11）、RKNN model zoo、各 NPU 官方 demo |
| **C：端到端含 NMS** | 直接输出最终框 | 最省事但 **NMS/TopK 算子绝大多数 NPU 工具链不支持**，动态 shape 也麻烦 | 基本只在 GPU/x86 上可用，不推荐依赖 |

> `nc` = 你的类别数；`64 = 4 × reg_max(16)` 是回归分支通道数（v8/v11 anchor-free 结构）。
> YOLOv5 是 anchor-based，形态 B 的通道数是 `3×(5+nc)`，思路相同。

**为什么会这样？** NPU 只擅长跑"纯卷积流"，像解码、NMS 这种带分支/排序/动态 shape 的逻辑编译不进去，所以 NPU 路线（M/R/H/V/S/B）普遍要求形态 B，把后处理留在 CPU 侧的官方 demo 代码里。GPU/x86 路线没有这个限制，形态 A 最省事。

### 概念三：量化三步曲

NPU 只认 **INT8**（自己电脑上的显卡/CPU 则 FP16/FP32 随意）。把 FP32 模型变 INT8 的过程叫**量化**，所有厂商工具链做法都一样，只是名字不同：

```
第1步 准备校准集（几十~几百张有代表性的图）
  ↓
第2步 定标：工具用校准集跑前向，统计每层激活值分布，算出每层的缩放系数 scale
  ↓
第3步 （可选）混合精度：发现某些层被 INT8 压坏了 → 把这几层单独设回 FP16/FP32
```

第5章会详细讲怎么准备校准集、掉精度了怎么办（深入的方法论在手册 §4）。

---

# 第2章 数据准备 —— 一切质量的上限

模型上限由数据决定，部署只能"不损失"它。这一章所有平台的读者都一样。

## 2.1 YOLO 数据格式（行业通用标准）

```
dataset/
├── images/
│   ├── train/          # 训练图
│   └── val/            # 验证图
├── labels/
│   ├── train/          # 与图片同名的一一对应 .txt
│   └── val/
└── dataset.yaml
```

每张图一个同名 `.txt`，每行一个目标：

```
class_id cx cy w h
```

- 四个坐标都是 **0~1 的归一化值**：中心点 (cx, cy) + 宽高 (w, h)，除以图片宽高
- `class_id` 从 0 开始

`dataset.yaml`：

```yaml
path: D:/datasets/mydataset    # 数据集根目录
train: images/train
val: images/val

names:
  0: cat
  1: dog
  2: person
```

> 🔑 **names 的顺序就是 class_id 的顺序**。部署后"类别显示错位"的病根 90% 在这里：训练时 names 顺序、部署时标签表顺序不一致。建议把这份 names 抄到一个固定的地方（部署时要用）。

## 2.2 数据集划分（分层思想）

val 集的作用是"客观打分"，所以 **每个类在 val 里都必须有样本**，否则该类的 mAP 无意义。

**直接用仓库里的脚本**（分层 + 容错 + 打印分布，不用自己抄）：

```powershell
python scripts/split_dataset.py --src raw --dst dataset --val-ratio 0.1
```

它比"listdir + shuffle 切一刀"多做三件必须做的事：① **分层划分**，每个类都保证有 val 样本；② **容忍没有 `.txt` 的背景图**（不崩、照样进 train）；③ **切完打印每类的 train/val 分布**，某类为 0 会当场告警。

划分完抽查两点：① 随便打开 val 的图和 txt 对一眼坐标对不对；② 训练一轮后看混淆矩阵，确认每个类都"被考到"。

## 2.3 数据坑清单

1. **class_id 必须从 0 连续**，和 `dataset.yaml` 的 names 一一对应
2. 路径避免**中文和空格**（Windows 下 dataloader 报奇怪的错，多半是它）
3. 每类 **≥100 张**是舒服的起点；更少也能训，但要配合数据增强和更多轮数
4. **场景多样性 > 数量**：不同光照、角度、距离、背景都要有，尤其是"部署现场会长什么样"
5. 小目标（成图后 <10 像素）检测很难：考虑加大 imgsz、或用裁图/局部训练
6. 🔑 **额外留 10~20 张部署现场实拍图，永远不进训练集** —— 它们有两个用途：上板前后对比效果的"金标准"、量化校准集的原料（第5章）
7. 适量放几张**没有目标的背景图**（空 txt 或不放 txt）进训练集，能明显减少误检

---

# 第3章 训练 —— 得到 best.pt

## 3.1 环境搭建

```powershell
# Windows PowerShell / CMD
python -m venv .venv
.venv\Scripts\activate

# 有 NVIDIA 显卡（按驱动支持的 CUDA 选，12.4 示例）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 没有显卡：
pip install torch torchvision

pip install ultralytics
```

验证：

```powershell
python -c "import torch; print(torch.cuda.is_available())"   # 有显卡应为 True
python -c "from ultralytics import YOLO; print('ok')"
```

> **没有显卡怎么办？** ① CPU 硬训：n 模型 + imgsz 320 + 小数据集可以忍；② 云 GPU：Colab 免费 T4、国内按小时租（AutoDL 等），训练完把 `best.pt` 下载回来，后面的转换都在自己电脑做（转换基本不需要 GPU）。

## 3.2 部署感知的超参数 —— 训练前先定死这两个

### imgsz（输入边长）

算力消耗 ∝ imgsz²：640 的计算量是 320 的 **4 倍**。它必须 **训练 = 导出 = 部署，三处一致**（32 的倍数）。

| 路线 | 建议 imgsz | 说明 |
|------|-----------|------|
| M（MaixCAM 系）、H（RDK X3）、R 之 RK3566/3568 | **320** | 主流边缘 NPU 甜点 |
| R 之 RK3588 / RK3576、B（BM1684）、V 之 A311D | 320~640 | 算力富余，可上 416/640 |
| R 之 Luckfox RV1103/1106、V 之 V853 | 160~320 | 极限小算力，n 模型+小尺寸 |
| J（Jetson Orin 系） | 416~640 | FP16 起步，够快 |
| C（树莓派/手机 CPU） | 320~416 | INT8 后再考虑加大 |
| X（x86 CPU） | 480~640 | INT8 + 现代 CPU 指令集可撑住 |
| 服务器 GPU | 640 | 标准尺寸 |

**新手策略**：先用 320 走通全流程，最终效果不够再加大重训。重训成本远低于部署后发现跑不动的返工成本。

> ⚠️ 320 是"先跑通"的保守起点，不是板子上限。比如 MaixCAM2 是 3.2 TOPS@INT8（AX630C），Sipeed 公开数据 YOLO11n@640 约 113 FPS——真做产品时该往上试，别被这张表钉死。

### 模型规模

| 规格 | 参数量（YOLO11） | 适用 |
|------|-----------------|------|
| n | ~2.6M | 边缘 NPU 的默认起点 |
| s | ~9.4M | RK3588；或 n 量化掉精度时的升级选项 |
| m / l / x | 更大 | 服务器、大算力盒子 |

> 经验：**n 比 s 对量化更敏感**（层薄、冗余少，INT8 误差没有缓冲空间）。先 n 跑通，如果量化后掉精度 >3%（第7章会测），换 s 重训通常立竿见影。

## 3.3 训练脚本

```python
# train.py
from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # 首次自动下载预训练权重
                              # 断点续训：YOLO("runs/detect/train/weights/last.pt")

model.train(
    data="dataset.yaml",
    epochs=200,
    imgsz=320,               # ← 3.2 定的值，之后导出/部署都用它
    batch=16,                # 显存不够就降 8/4；或 batch=-1 让它自动
    device=0,                # GPU；CPU 写 "cpu"
    workers=0,               # Windows 建议 0~2，避开多进程加载的坑
    patience=0,              # 0=跑满不早停；设 50 则 50 轮无提升自动停
    save=True,
    # ---- 数据增强：让模型见过"歪的、暗的、糊的"目标 ----
    hsv_h=0.05, hsv_s=0.9, hsv_v=0.8,    # 颜色/亮度扰动：适应光照与白平衡变化
    degrees=10,                            # 旋转 ±10°
    translate=0.2, scale=0.6,              # 平移、缩放（尺度泛化最重要的一个）
    shear=3, perspective=0.0005,
    mixup=0.1, copy_paste=0.1,             # 混合、贴图
)
```

> 增强不是越大越好：**固定机位/工业相机**（相机不动的产线、卡口）场景应大幅调小几何增强，模型学到的分布要贴近真实分布，而不是"万物皆可旋转"。

产物：`runs/detect/train/weights/best.pt`（val mAP 最高的那一轮）。

## 3.4 怎么判断"训好了"

1. 看 `runs/detect/train/results.csv` 的 `metrics/mAP50-95(B)`：曲线走平即收敛
2. 看 `confusion_matrix.png`：哪些类互相混淆？哪类大面积漏进 Background？
3. 粗略及格线：mAP50 ≥ 0.8、mAP50-95 ≥ 0.5（工业检测类任务要更严）
4. 过拟合信号：训练 loss 还在降，val mAP 却回落 → 下次给 `patience=30~50`
5. **肉眼看**：打开 `val_batch0_pred.jpg`（不是 val_batch_pred.jpg）；再用几张训练集外的现场图手动测：

```python
from ultralytics import YOLO
YOLO("runs/detect/train/weights/best.pt").predict("scene_test/1.jpg", conf=0.25, save=True)
```

记住此时的 val mAP 数字（比如 0.87）—— 它是第7章"量化有没有压坏模型"的**对比基线**。

---

# 第4章 导出 ONNX —— 通向所有平台的桥

## 4.1 为什么是 ONNX

`.pt` 是 PyTorch 的私有格式，只有 PyTorch 自己认识。**ONNX 是行业中间语言**——所有平台的转换工具（Pulsar2、RKNN-Toolkit2、TensorRT、NCNN、OpenVINO、TPU-MLIR……）都吃 ONNX：

```
best.pt ──导出──▶ best.onnx ──各家工具──▶ .axmodel / .rknn / .engine / .param / .xml / .bmodel ...
```

所以 ONNX 是"一次导出，到处转换"的分发格式。

## 4.2 第一步：确认你的路线要哪种形态（见第1章概念二）

| 路线 | 需要的形态 | 原因 |
|------|-----------|------|
| M（MaixPy） | **B 三尺度** | nn.YOLO11 框架自己解码，喂单输出会报错 |
| R（RKNN） | **B**（配官方 model zoo 后处理）或 A（自己写后处理） | 二选一：B 需要 decode+NMS，但能直接抄官方 demo 的现成代码；A 只剩 NMS，适合已有后处理框架的人 |
| J（Jetson） | **A**（或直接一键导 engine，跳过 ONNX） | GPU 无算子限制 |
| C（NCNN/MNN） | **A** | decode 已在图里，只剩 NMS |
| X（OpenVINO/ORT） | **A** | 同上 |
| H/V/S/B（国产 NPU） | 按官方 SDK 示例裁剪，常见 **B** 或"魔改头" | 和 M/R 同理：NPU 只跑纯卷积 |

> 拿不准就 A、B 都导出（每份几十 MB、几分钟的事），到第6章路线卡里自然知道用哪份。

## 4.3 形态 A：解码单输出（一行命令）

```python
from ultralytics import YOLO

YOLO("best.pt").export(format="onnx", imgsz=320, simplify=True, opset=13)
# 产物：runs/detect/train/weights/best.onnx
```

要点：
- `imgsz` **必须和训练一致**
- 默认 `dynamic=False, batch=1`：固定 shape 是边缘部署最稳的形态，别乱开 dynamic
- 输出 shape 应为 `[1, 4+nc, 2100]`（320 输入；640 输入则是 `[1, 4+nc, 8400]`）
- 如果你的路线是 J（Jetson），甚至可以跳过本章直接 `export(format="engine")`，见 §6.3

## 4.4 形态 B：三尺度特征图（脚本导出）

NPU 路线需要"把检测头最后的解码截掉，只留 3 个尺度的 Conv 输出"。用 wrapper 包一层再导：

```python
# export_3scale.py
import torch, onnx, onnxsim
from ultralytics import YOLO

PT, OUT, IMG = "best.pt", "best_3scale.onnx", 320

yolo = YOLO(PT)
det = yolo.model          # DetectionModel
det.eval()                # 必须 eval，否则 head 返回训练态结构

class ThreeScale(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m
    def forward(self, x):
        _, features = self.model(x)   # eval 态返回 (解码输出, [三尺度特征图])
        return features

dummy = torch.randn(1, 3, IMG, IMG)
torch.onnx.export(ThreeScale(det), dummy, OUT,
                  input_names=["images"],
                  opset_version=13)   # 工具链要求更高时再提到 17/19

# 简化图结构
ms, ok = onnxsim.simplify(OUT)
if ok:
    onnx.save(ms, OUT)

# ★ 打印真实的输出名和 shape —— 下一章配置文件全靠抄这里，别手打！
m = onnx.load(OUT)
for o in m.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(o.name, dims)
```

320 输入、v8/v11 的预期输出（nc 换成你的类别数）：

```
[1, 64+nc, 40, 40]   # 大目标尺度
[1, 64+nc, 20, 20]   # 中目标尺度
[1, 64+nc, 10, 10]   # 小目标尺度
```

> - YOLOv5 是 anchor-based，这里的通道数是 `3×(5+nc)`，流程完全一样
> - 输出名（形如 `/model.23/Concat_output_0`）由图结构决定，**以打印结果为准**。经验值：
>   - **YOLO11** → `/model.23/Concat_output_0`、`/model.23/Concat_1_output_0`、`/model.23/Concat_2_output_0`
>   - **YOLOv8** → `/model.22/Concat_1_output_0`、`/model.22/Concat_2_output_0`、`/model.22/Concat_3_output_0`（注意**从 `_1` 开始**，不是 `_0`）
>   - 变的是"代数"：v8 的 Detect 在第 22 层，v11 因为加了 C2PSA 变成第 23 层。**模型规模（n/s/m/l/x）不影响层号**；ultralytics 小版本升级也可能改名，所以永远以打印结果为准
> - 想可视化结构就装 [Netron](https://netron.app)（网页版直接拖文件进去），找最后三个 Concat 节点

## 4.5 自检：FP32 ONNX 和 .pt 必须对得上（★ 全流程最重要的 5 分钟）

量化、转换、上板排障的第一步永远是"确认 FP32 ONNX 本身没病"。现在花 5 分钟锁定它，以后能省几天冤枉路：

```python
# check_onnx.py —— 同一张真图，PyTorch 和 ONNX 各跑一遍，比输出
import numpy as np, cv2, torch, onnxruntime as ort
from ultralytics import YOLO

IMG, ONNX_PATH = 320, "best.onnx"      # 查形态 B 就改成 best_3scale.onnx

img = cv2.cvtColor(cv2.imread("val/1.jpg"), cv2.COLOR_BGR2RGB)   # 注意转 RGB！
x = cv2.resize(img, (IMG, IMG)).astype(np.float32).transpose(2, 0, 1)[None] / 255.0

# 参考答案：PyTorch
yolo = YOLO("best.pt"); yolo.model.eval()
with torch.no_grad():
    t_out = yolo.model(torch.from_numpy(x))     # (形态A输出, [三尺度])

# 学生答卷：ONNX
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
o_out = sess.run(None, {sess.get_inputs()[0].name: x})

t_a = t_out[0].numpy()                          # 形态 A 参考
print("形态A  max|pt-onnx| =", np.abs(t_a - o_out[0]).max())
# 查形态 B 则逐个比较：t_out[1][i] 对应 o_out[i]，i=0,1,2
for i, (tf, of) in enumerate(zip(t_out[1], o_out)):
    print(f"尺度{i}  max|pt-onnx| =", np.abs(tf.numpy() - of).max())
```

**判定：最大绝对误差 < 1e-3 即通过。** 不通过就别往下走——先排查 imgsz 是否和训练一致、预处理是不是 RGB+/255、ONNX 是否导错。

> 说明：这段脚本故意用**拉伸 resize**（不 letterbox），目的是让 PyTorch 和 ONNX 吃**完全相同的输入**，比的是"导出有没有导错"，不是"端到端精度"。真做端到端精度对比时，两端预处理必须完全一致（见 §1.3 四要素）。
> 本仓库 `scripts/check_onnx_vs_pt.py` 是这份脚本的加强版：自动识别形态 A/B、逐尺度输出误差、给出通过/不通过结论。

## 4.6 opset 怎么选

opset 是 ONNX 的"指令集版本"：越高算子越新，但老工具链可能不认。规则：

- 默认 **13**，跨工具链兼容性最好
- 报 `unsupported operator / Unsupported opset` → 降 opset 或升级工具链
- 路线 M（Pulsar2）实测 19 可用；越新的工具链支持的 opset 越高

## 4.7 到这里你手里有什么

| 物件 | 用途 |
|------|------|
| `best.pt` | 训练成果，永远备份 |
| `best.onnx` / `best_3scale.onnx` | 分发给任何平台的中间格式 |
| 记录下来的：输入名（`images`）、shape、**输出名列表** | 第6章配置文件和后处理代码要逐字抄它 |

---

# 第5章 量化入门 —— NPU 路线的必修课

> 路线 J / X 的读者也要读：FP16/FP32 虽然不强制 INT8，但"掉精度排查"的思路是通用的，而且你们迟早会遇到要上 INT8 提速的场景。

## 5.1 量化是什么，为什么 NPU 必须要

**INT8 量化 = 用 8 位整数（256 个档位）表示原本 32 位浮点的数。**

边缘 NPU 的硬件乘加单元就是按 INT8 设计的：跑 INT8 是全速，跑 FP32 要么不支持要么慢好几倍。所以对 NPU 路线（M/R/H/V/S/B），量化不是可选项，是**门票**。顺带的收益：模型体积缩到 1/4，内存带宽省 3/4，速度提升 2~4 倍。

原理一句话：找一个缩放系数 `scale`，让 `真实值 ≈ int8值 × scale`。整个模型几百层，每层都有自己的 scale——**定标（calibration）就是算出这些 scale 的过程**。

为什么会掉精度？INT8 只有 256 个档位，如果某层激活值分布很"长尾"（大部分值很小、极少数很大），统一用一个 scale 就会把绝大多数值压到几个档位里，细节全糊掉。掉精度的本质就是个别层被压得太狠——解法见 5.4 和手册 §4。

> 名词：**per-channel（逐通道）**= 卷积权重每个输出通道单独一个 scale，比一刀切的 per-tensor 精度高得多，现代工具链基本默认开启。看到配置里有 `quantized_method: channel` 这类字样，保持默认就好。

## 5.2 PTQ 还是 QAT？

| 方式 | 全称 | 做法 | 成本 | 何时用 |
|------|------|------|------|--------|
| **PTQ** | 训练后量化 | 拿训好的模型 + 校准集，工具几分钟搞定 | ★ | **99% 的人用这个，本教程只讲它** |
| QAT | 量化感知训练 | 训练时就模拟 INT8 舍入误差，重新训练 | ★★★★ | PTQ + 混合精度仍掉 >2~3% 时才考虑（手册 §4.7） |

## 5.3 校准集 —— 量化的"教材"

定标的原理：工具把校准集图片**喂进模型跑前向**，统计每层激活值的分布，据此算 scale。

所以校准集的作用是**告诉工具"真实数据的激活值长什么样"**。结论非常明确：

> 🔑 **分布代表性 > 数量。** 50 张贴近部署现场的图，胜过 500 张网上随便爬的图。

**选图四原则：**
1. 来自**真实部署场景**——部署在哪拍，校准图就该长什么样（同样的相机、光照、视角）
2. **覆盖所有类别**，别让某个类在校准集里失踪
3. 从 train/val 里抽即可，混入现场实拍图更好
4. 数量 50~500 张都有人用，**50 张是实测够用的下限**（MaixCAM 实战就是 50）

**不同工具要的"打包形态"不同**，常见三种：`.tar` 包（Pulsar2）、`.txt` 路径清单（RKNN）、文件夹/生成器（OpenVINO/ORT）。一个脚本把前两种一次备好：

```python
# make_calib.py —— 从 val 抽 N 张，同时产出 tar 包和 txt 清单
import os, random, tarfile

SRC, N = "dataset/images/val", 50
random.seed(42)
imgs = [f for f in os.listdir(SRC) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
pick = random.sample(imgs, min(N, len(imgs)))

with tarfile.open("calibration.tar", "w") as tar:
    for f in pick:
        tar.add(os.path.join(SRC, f), arcname=f)

with open("dataset.txt", "w") as fp:
    for f in pick:
        fp.write(os.path.abspath(os.path.join(SRC, f)) + "\n")

print(f"已打包 {len(pick)} 张 → calibration.tar / dataset.txt")
```

> 注意：工具在校准时会按你配置的预处理（mean/std/颜色序）处理这些图——**配置必须和部署时一致**，否则统计出来的 scale 就是错的。这又回到第1章铁律。

## 5.4 精度预期：量化后掉多少算正常

| 量化后 mAP 掉幅 | 判定 | 动作 |
|----------------|------|------|
| < 1% | 完美 | 直接收工 |
| 1% ~ 3% | 正常范围 | 可先接受；想榨精度 → 手册 §4.5 排查树 |
| > 3% | 有问题 | 按第7章流程分锅定位 |
| 行为异常（全识别成一类 / 置信度集体跳水 / 满屏乱框） | 有大问题 | 几乎一定是敏感层被压死或预处理不一致 → §6.x 你路线卡的"坑"一节 + 手册 §4.8 症状表 |

**打一剂预防针**：检测头的**分类分支末端 Conv**（如 `/model.23/cv3.x/cv3.x.2/Conv`；YOLOv8 则是 `/model.22/cv3.x/cv3.x.2/Conv`）是全模型量化敏感度最高的地方之一——它直接决定"这是什么类"，层数又薄。MaixCAM 实战就栽在这里过：INT8 后所有目标被识别成同一类，解法是把这几层强制回 FP32（混合精度），见 §6.1 实战记录。记住这个位置，遇到"全识别成一类"先想到它。

另外：INT8 后模型的置信度整体会漂移，**conf 阈值要在量化后的模型上重新标定**（第7章会做）。

---

# 第6章 路线卡 —— 翻到属于你的平台，照着做到底

## 6.0 先看这里：NPU 路线的通用套路

每张路线卡固定五段：**环境准备 → 转换+量化 → 上板 → 运行 → 该路线的坑**。

先剧透一个行业规律，看完你就不会觉得各家的流程面目全非了：

> **厂商量化工具链基本都用 Docker 镜像发布**（Pulsar2、OpenExplorer、TPU-MLIR……），因为依赖重、环境老，容器一次打包到处跑。所以 NPU 路线的第一天几乎都是：装 Docker → load 厂商镜像。

**Docker 通用三步**（路线 M/H/B/V 都会用到）：

1. 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)（Windows 会自动启用 WSL2，需管理员权限）
2. 从厂商渠道下载镜像 tar 包，加载：`docker load -i 厂商镜像.tar.gz`
3. 运行时把工作目录挂载进去：
   ```powershell
   # Windows：务必用 PowerShell 执行（Git Bash 会乱转换路径参数导致挂载失败）
   docker run --rm -v D:\yolo_work:/data 镜像名 -c "容器内执行的转换命令"
   ```

另一个通用规律：**转换/量化在电脑上做，推理在板子上做**，两边各自有软件包，且**版本必须配套**（路线 R 尤其严格）。

---

## 6.1 路线 M：MaixCAM2（爱芯元智 AX630C）

> ⭐ **已验证路线**——本卡内容来自一次完整跑通的实战记录（`yolo-to-maixcam2-workflow.md`），并结合 Sipeed 官方转换文档校正过，遇到问题优先相信这里。

**你需要准备**：`best_3scale.onnx`（形态 B，§4.4）、`calibration.tar`（§5.3）、Docker Desktop
**最终产物**：`model_npu.axmodel` + `model.mud` → MaixPy 板上运行

### 第一步：环境

1. 装 Docker Desktop（见 6.0）
2. 从 HuggingFace `AXERA-TECH/Pulsar2` 下载 Pulsar2 镜像 tar.gz
3. `docker load -i pulsar2_v6.0.tar.gz`（文件名/镜像标签以你实际下到的版本为准，`docker images` 里看到什么就写什么）

> ⚠️ **最容易搞错的一点：芯片型号 ≠ Pulsar2 平台名。**
> - **MaixCAM2**：芯片是 **AX630C**（3.2 TOPS@INT8），但 Pulsar2 里 `--target_hardware` 要写 **`AX620E`**。Sipeed 官方文档原文：「MaixCAM2 对应 Pulsar2 文档中的 AX620E 平台……固定写 AX620E」。
> - **MaixCAM / MaixCAM-Pro（一代）**：芯片是算能 **SG2002**，走的是算能 TPU-MLIR（产物 `.cvimodel`），**和 Pulsar2 完全不是一个工具链**，请去看 Sipeed 的 MaixCAM 转换文档，别用本卡。
> - 参考：[MaixCAM2 转换文档](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam2.html)、[MaixCAM 转换文档](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html)

### 第二步：写配置文件 config.json

把 §4.4 打印出来的输出名，逐字抄进 `output_processors`：

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU2",
  "quant": {
    "input_configs": [{
      "tensor_name": "images",
      "calibration_dataset": "datasets/train.tar",
      "calibration_size": 50,
      "calibration_mean": [0, 0, 0],
      "calibration_std": [255.0, 255.0, 255.0]
    }],
    "calibration_method": "MinMax",
    "precision_analysis": true,
    "precision_analysis_method": "EndToEnd"
  },
  "input_processors": [{
    "tensor_name": "images",
    "tensor_format": "RGB",
    "tensor_layout": "NCHW",
    "src_format": "RGB",
    "src_dtype": "U8",
    "src_layout": "NHWC",
    "csc_mode": "NoCSC"
  }],
  "output_processors": [
    {"tensor_name": "/model.23/Concat_output_0", "dst_perm": [0, 2, 3, 1]},
    {"tensor_name": "/model.23/Concat_1_output_0", "dst_perm": [0, 2, 3, 1]},
    {"tensor_name": "/model.23/Concat_2_output_0", "dst_perm": [0, 2, 3, 1]}
  ],
  "compiler": { "check": 3, "check_mode": "CheckOutput", "check_cosine_simularity": 0.9 }
}
```

配置怎么读（对照第1章的预处理四要素）：

- `calibration_mean/std`：把 `/255` 归一化**内嵌进模型**，板端代码直接喂 0~255 的图
- `src_dtype: U8` + `src_layout: NHWC`：告诉编译器板端喂的是"通道在后"的 8 位图
- `dst_perm: [0,2,3,1]`：把输出从 NCHW 排成 NHWC，方便 MaixPy 后处理读
- `precision_analysis`：让工具自己报告量化前后每层误差（第7章排查用）
- **故意不写 `input_shapes`**：输入尺寸直接从 ONNX 读。手写一遍就等于多了一个可能写错的数字（shape mismatch 的经典来源）；只有多输入/动态 shape 的模型才需要显式指定
- `tensor_format` / `src_format` 都写 **RGB**，配 `csc_mode: NoCSC`（不做颜色空间转换）：和训练时的 RGB 一致，也和 MUD 里的 `input_type = rgb` 一致
  - ⚠️ 老攻略里常见 `BGR/BGR` 的写法。那只在"训练就是 BGR、板端也喂 BGR"时成立；写反了模型看到的是 R/B 互换，精度会**悄悄**掉，不报错
- `tensor_layout: NCHW` 是模型内部布局，`src_layout: NHWC` 是板端喂进来的内存布局——**这两个不是一回事**，别写反
- `compiler.check: 3` + `check_mode: CheckOutput`：让编译器拿 NPU 输出和 ONNX 输出对比，余弦相似度低于 0.9 会告警（官方推荐值，比原来的 `check: 2` 更容易抓出量化坏层）

### 第三步：Docker 里转换

```powershell
# 推荐用法：先起交互容器（Sipeed 官方文档就是这个用法，不依赖镜像 entrypoint 怎么定义）
docker run -it --rm -v D:\yolo_work:/data -w /data pulsar2:6.0

# 进入容器后，在 /data 目录下执行（路径写相对路径，跨平台最省心）
pulsar2 build --target_hardware AX620E --input ./best_3scale.onnx --config ./config.json --output_dir ./out_npu
```

> 注意两件事：
> 1. `--target_hardware` **固定 `AX620E`**（MaixCAM2 的芯片是 AX630C，但 Pulsar2 平台名就叫 AX620E）。
> 2. 不要写成 `docker run ... 镜像名 -c "pulsar2 build ..."`——`-c` 是 bash 的参数，镜像入口如果不是 bash 就会把整串当成命令名直接报错。要一条命令跑完就用 `docker run ... 镜像名 sh -c "pulsar2 build ..."`。

产物：`out_npu/compiled.axmodel` → 改名 `model_npu.axmodel`。

**还要一份 NPU1 模式**（AI-ISP 兼容用）：把 config.json 里 `npu_mode` 改成 `"NPU1"` 再跑一次，产物改名 `model_vnpu.axmodel`。原因见下面坑 ②。

### 第四步：写 MUD 描述文件

`model.mud`——MaixPy 靠它认识模型：

```ini
[basic]
type = axmodel
model_npu = model_npu.axmodel
model_vnpu = model_vnpu.axmodel

[extra]
model_type = yolo11
type = detector
input_type = rgb
labels = 你的标签1, 你的标签2, 你的标签3

input_cache = true
output_cache = true
input_cache_flush = false
output_cache_inval = true

mean = 0,0,0
scale = 0.00392156862745098,0.00392156862745098,0.00392156862745098   # = 1/255
```

> ★ `labels` 的顺序必须 = `dataset.yaml` 的 `names` 顺序，数量必须 = nc。这一条用 `scripts/lint_deploy_config.py --mud` 一条命令就能核对，不用肉眼数。

> **关于 MUD 里的 `mean` / `scale`**：归一化**已经编译进 axmodel 了**（就是上面 config.json 的 `calibration_std=255`）。这里再写一遍只是给人看的记录，MaixPy 不会再加一遍——Sipeed 官方文档的原话是「实际上在转模型时，预处理已经集成在模型中，这里写出来只是为了方便查看」。所以看到两处都有 `/255` 别慌，归一化仍然只做了一次。

### 第五步：板上运行代码

```python
# main.py（MaixPy）
from maix import camera, display, image, nn

detector = nn.YOLO11(model="/root/sy/model.mud", dual_buff=True)   # dual_buff: 前处理/推理流水线并行，提速
cam = camera.Camera(width=320, height=320)
disp = display.Display()

while True:
    img = cam.read()
    if img is None:
        continue
    objs = detector.detect(img, conf_th=0.5, iou_th=0.45)
    for obj in objs:
        lab = f"{detector.labels[obj.class_id]}: {obj.score:.2f}"
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED, thickness=2)
        img.draw_string(obj.x, obj.y - 8, lab, color=image.COLOR_GREEN, scale=1, thickness=1)
    disp.show(img)
```

### 第六步：部署

```
本机目录 sy/
├── main.py
├── model.mud
├── model_npu.axmodel
└── model_vnpu.axmodel
```

```powershell
scp -r sy/ root@192.168.x.x:/root/      # 板子 IP 按实际
# 板上运行：python /root/sy/main.py
```

### 该路线的坑（实战亲历）

| 症状 | 原因 | 解法 |
|------|------|------|
| nn.YOLO11 加载报 `Tensors get tensor idx error` | 喂了形态 A 单输出 `[1,13,2100]` 的模型 | **必须**用形态 B 三尺度（§4.4）转出来的 axmodel |
| `ai isp is on` 报错 | 夜视 AI-ISP 开着时走的是 VNPU 通路 | MUD 里 `model_vnpu` 必须配 NPU1 编译的模型；NPU2 编译的不能当 vnpu 用 |
| **所有目标识别成同一类** | INT8 把分类分支末端压死 | config.json 加 `layer_configs`，把 cv3 末端 3 层 Conv 强制 FP32（见下） |
| layer_configs 报错找不到层 | 层名写错 | `layer_name` 必须是 ONNX 里的确切节点名（Netron 查，或 §4.4 打印的输出名附近） |
| Docker 挂载路径失败 | Git Bash 转换路径 | 用 PowerShell 执行 docker 命令 |

分类层保精度的三行配置（加进 config.json 顶层）：

```json
"layer_configs": [
  {"layer_name": "/model.23/cv3.0/cv3.0.2/Conv", "data_type": "FP32"},
  {"layer_name": "/model.23/cv3.1/cv3.1.2/Conv", "data_type": "FP32"},
  {"layer_name": "/model.23/cv3.2/cv3.2.2/Conv", "data_type": "FP32"}
]
```

---

## 6.2 路线 R：Rockchip RKNN（RK3588 / RK3566 / RK3568 / RK3576 / Luckfox RV1103/1106）

**你需要准备**：`best_3scale.onnx` 或 `best.onnx`、`dataset.txt`（§5.3）、一台 x86 电脑（转换机）+ 你的板子（刷好官方系统）
**最终产物**：`best.rknn` → 板上 `rknn-toolkit-lite2`（Python）或 C API 运行

### 第一步：转换机装 RKNN-Toolkit2

1. 到 GitHub `rockchip-linux/rknn-toolkit2` 仓库，按 README 安装（Linux x86_64 原生最顺，Windows / WSL 也可用于转换）
2. 先装 `requirements`，再 `pip install` 对应平台的 wheel 包

> 🔑 **本路线第一铁律：Toolkit2 版本 ↔ 板端 librknnrt.so 版本必须配套。** 版本错位会得到段错误、"rknn init failed" 等神秘错误。板子刷官方系统、电脑装同版本工具，最省心。新版 ultralytics 也支持 `format='rknn'` 一键导出，但建议先用下面的脚本手动走一遍，理解每步在干嘛。

### 第二步：转换 + 量化

```python
# convert_rknn.py
from rknn.api import RKNN

rknn = RKNN(verbose=True)

rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],      # 把 /255 内嵌进模型 → 板端直接喂 0~255 的 uint8 图
    target_platform='rk3588',          # ★ 按你的芯片改：rk3566 / rk3568 / rk3576 / rv1103 / rv1106 ...
    quantized_dtype='w8a8',            # 权重8bit+激活8bit（标准INT8）
    quantized_algorithm='normal',      # normal=MinMax 定标；掉精度可换 'kl_divergence' 或 'mmse'（慢但准）
    quantized_method='channel',        # 权重 per-channel，保持默认
    optimization_level=3,
)

rknn.load_onnx('best_3scale.onnx')     # 形态 B；若自己写后处理可用形态 A
rknn.build(do_quantization=True, dataset='dataset.txt')   # §5.3 的校准清单
rknn.export_rknn('best.rknn')
print('OK → best.rknn')
```

### 第三步：转换机上先做精度自查

```python
# 必须接在上面 build() 之后、同一个 rknn 对象上调用（新开进程/新建对象会报模型未构建）
rknn.accuracy_analysis(inputs=['dataset/images/val/1.jpg'], output_dir='./snapshot')
```

`accuracy_analysis` 用的是已经 `load_onnx + build` 过的模型，**不能单独跑**（这是很多人第一次用就卡住的点）。打开 `snapshot` 里的报告：**每层给一个量化前后输出的余弦相似度**（越接近 1 越好）。哪层相似度明显塌下去（<0.9），它就是敏感层——第7章排查和混合精度都用得上这个信息。

### 第四步：上板运行

```python
# 板端 Python 最小推理（pip install rknn-toolkit-lite2，版本与转换工具配套）
from rknnlite.api import RKNNLite
import numpy as np, cv2

rknn = RKNNLite()
rknn.load_rknn('best.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)   # 仅 RK3588：CORE_0/1/2 或 CORE_AUTO 三核自动分配

img = cv2.cvtColor(cv2.imread('test.jpg'), cv2.COLOR_BGR2RGB)   # ★ BGR→RGB
img = cv2.resize(img, (320, 320))
outs = rknn.inference(inputs=[np.expand_dims(img, 0)])          # 3 个尺度输出
```

**后处理别从零写**：拿官方 **rknn_model_zoo**（GitHub `airockchip/rknn_model_zoo`）里 yolov8 / yolov11 的 demo，它的 C/Python 代码自带 dist2bbox 解码 + NMS，把你的类别数和置信度阈值改了就能用；要 C/S 的话也是在这个 demo 上改。

### 该路线的坑

1. **版本不配套**是新手第一大坑（见第一步铁律），症状千奇百怪，先查版本再查别的
2. **颜色序**：OpenCV 读图是 BGR，模型训练用 RGB——校准图和板端图都要转成 RGB 再喂
3. 量化掉精度 → 混合精度：Toolkit2 的 **hybrid quantization 两步法**（step1 生成配置 → 在配置里把敏感层标回 FP16 → step2 重编译），操作见手册 §4.4
4. **Luckfox（RV1103/1106）**：同一套工具链，`target_platform` 写对即可；算力很小，imgsz 建议 160~320、模型用 n
5. RK3588 想吃满 3 个 NPU 核：单实例 `CORE_AUTO` 或跑多实例；单模型默认只占 1 核，觉得"3588 也就这点速度"多半是这原因

---

## 6.3 路线 J：NVIDIA Jetson（Orin / Xavier，TensorRT）

**你需要准备**：板子（刷好 JetPack）、板内能联网
**最终产物**：`best.engine`（在板上生成，**只能在同型号+同版本环境的板上跑**）

> Jetson 是"舒服路线"：TensorRT 对 PyTorch 生態兼容极好，ultralytics 一条命令导出+部署，FP16 几乎零精度损失。**先 FP16 跑通，速度不够再上 INT8。**

### 第一步：板上环境

JetPack 自带 TensorRT。装 PyTorch 要用 **NVIDIA 的 Jetson 专用 wheel**（普通 pip 装的没有 GPU 支持），到 NVIDIA Jetson PyTorch 页面按 JetPack 版本选；然后 `pip install ultralytics`。

### 第二步：一键导出 engine

```bash
# imgsz 必须 = 训练时的值（本教程示例是 320，见 §3.3）。
# 你要用 640 也可以，但训练/导出/部署必须全程 640 —— 别在这里悄悄改成 640。

# FP16（推荐起步，精度几乎无损）
yolo export model=best.pt format=engine imgsz=320 half=True

# INT8（自动用 data 指定数据集抽样校准）
yolo export model=best.pt format=engine imgsz=320 int8=True data=dataset.yaml
```

也可走标准 ONNX 路线（更透明）：

```bash
# 先按第4章导 best.onnx，再：
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --fp16 --saveEngine=best.engine
```

### 第三步：运行（后处理全自动）

```python
from ultralytics import YOLO

model = YOLO("best.engine")            # engine/onnx/pt 都能直接加载
results = model.predict("test.jpg", conf=0.25, save=True)
```

ultralytics 的 runtime 自带解码 + NMS，不用写一行后处理。

### 该路线的坑

1. **engine 不可移植**：绑定 GPU 架构 + TensorRT 版本。要在哪块板跑，就在哪块板导；团队分发分 `.pt`/`.onnx`，各自本地导 engine
2. INT8 掉精度：用 **polygraphy**（NVIDIA 官方调试工具）逐层对比定位敏感层，再决定混合精度（手册 §4.6）
3. Orin 系列需要 JetPack 5/6（TensorRT 8.5+）；旧 Xavier/Nano 的 JetPack 上 opset 别开太高
4. 多路视频流 + 高吞吐的终极形态是 DeepStream（进阶话题，先把单路跑通）

---

## 6.4 路线 C：NCNN / MNN（无 NPU 的 ARM 板 / 手机）

**你需要准备**：`best.pt`、校准图片文件夹（量化才需要）
**最终产物**：NCNN：`.param` + `.bin`；MNN：`.mnn`

> 适合：树莓派、安卓手机/盒子、任何"只有 CPU"的板子。没有 NPU 的硬性 INT8 要求，量化是"可选加速项"。

### NCNN（推荐，生态最大）

**第一步：一键导出**（ultralytics 内部走 ONNX → pnnx → ncnn）：

```bash
yolo export model=best.pt format=ncnn imgsz=320
# 产物：best_ncnn_model/ 目录，里面三个文件
#   model.ncnn.param
#   model.ncnn.bin
#   metadata.yaml        ← 记录 imgsz / names / 输出是不是解码后的
# 输出是形态 A（解码后），NMS 由运行时代码做
```

**第二步（可选）：INT8 量化**——用 ncnn 官方工具链两连击。注意文件名是 `model.ncnn.param` / `model.ncnn.bin`，**不是** `model.param` / `model.bin`：

```bash
# ① 先写 imagelist.txt：每行一个校准图路径（不需要标签）
#    例：
#    calib/0001.jpg
#    calib/0002.jpg

# ② 生成校准表（method=kl 是推荐的定标算法；shape 是 [w,h,c]）
ncnn2table model.ncnn.param model.ncnn.bin imagelist.txt yolo.table \
  mean=[0,0,0] norm=[0.00392,0.00392,0.00392] shape=[320,320,3] pixel=RGB method=kl thread=8

# ③ 应用校准表生成 INT8 模型
ncnn2int8 model.ncnn.param model.ncnn.bin model-int8.ncnn.param model-int8.ncnn.bin yolo.table
```

> 🔑 `mean/norm/pixel` 三个参数 **必须和你推理代码里的预处理一字不差**（下同），否则量化统计的分布和实际推理不一致，精度直接崩。

**第三步：运行**——ncnn 的 `examples/` 里有 yolov8/yolo11 的 C++ demo，把类别数、输入尺寸、归一化三处改了就能用；安卓走 JNI；Python 有 pybind 封装。

### MNN（备选）

**最省事**：新版 ultralytics 支持 `yolo export model=best.pt format=mnn imgsz=320`（内部走 ONNX → MNN），产物是 `best_mnn_model/model.mnn`。

**手动转换**（要控制参数时）：

```bash
# -f 指定输入格式，--MNNModel 指定输出文件；缺任何一个都会报参数错误
MNNConvert -f ONNX --modelFile best.onnx --MNNModel best.mnn --fp16
```

**INT8 量化**：用 MNN 仓库自带的 `tools/quantization`（没有叫 "MNN-Quant" 的独立工具），先按仓库里的 `calibration.json` 样例写好校准图目录和预处理参数，再跑：

```bash
# 在 MNN 源码仓库根目录
python3 tools/quantization/quantized.py --config calibration.json
```

> MNN 的量化脚本对配置格式比较挑、改版也勤，**直接照搬仓库里的样例改路径**，比照着博客手写靠谱。

### 该路线的坑

1. **校准参数与推理预处理不一致**是本路线第一大坑（见上 🔑）
2. ARM CPU 的 INT8 提速依赖芯片的 **dotprod 指令**（近几年的 Cortex-A 系都有；很老的可能无提升）
3. 手机上记得把推理线程**绑到大核**（taskset / API 亲和性设置），否则调度器把活派给小核
4. 预期管理：中端手机 + n 模型 + 320 输入，单帧几十毫秒量级；想要实时多路就换 NPU 路线

---

## 6.5 路线 X：x86 PC / 服务器（OpenVINO / ONNX Runtime）

**你需要准备**：`best.onnx`（形态 A）、校准图片文件夹
**最终产物**：`best_int8.xml/.bin`（OpenVINO）或 `best_int8.onnx`（ORT）

> 选型：**Intel CPU / 核显 / Intel 独显 → OpenVINO（首选）**；AMD CPU 或想跨平台 → ONNX Runtime。两者 FP32 模型都能直接跑，量化只是提速手段。

### OpenVINO 路线

**最省事**（新版 ultralytics 内置 NNCF 量化）：

```bash
yolo export model=best.pt format=openvino int8=True data=dataset.yaml
```

**手动 NNCF 量化**（想控制校准集时用）：

```python
# quantize_ov.py
import glob, cv2, numpy as np, openvino as ov, nncf

model = ov.Core().read_model("best.onnx")      # OV 可直接读 onnx

def calib_loader():
    for f in sorted(glob.glob("calib/*.jpg"))[:200]:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
        yield img.transpose(2, 0, 1)[None]

quantized = nncf.quantize(model, nncf.Dataset(calib_loader()))
ov.save_model(quantized, "best_int8.xml")
```

**运行**（后处理全自动，和跑 .pt 一模一样的体验）：

```python
from ultralytics import YOLO
results = YOLO("best_int8.xml").predict("test.jpg", conf=0.25, save=True)
```

### ONNX Runtime 路线

```python
# quantize_ort.py
import glob, cv2, numpy as np
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      CalibrationMethod, QuantFormat, QuantType)

class Reader(CalibrationDataReader):
    def __init__(self):
        self.files = sorted(glob.glob("calib/*.jpg"))[:200]
        self.i = 0
    def get_next(self):
        if self.i >= len(self.files):
            return None
        img = cv2.cvtColor(cv2.imread(self.files[self.i]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
        self.i += 1
        return {"images": img.transpose(2, 0, 1)[None]}

quantize_static("best.onnx", "best_int8.onnx", Reader(),
               calibration_method=CalibrationMethod.MinMax,
               quant_format=QuantFormat.QDQ,
               per_channel=True,
               activation_type=QuantType.QUInt8,
               weight_type=QuantType.QInt8)
```

运行同样是 `YOLO("best_int8.onnx")` 一行。

### 该路线的坑

1. **x86 上 INT8 不保证快**：提速靠 CPU 的 VNNI/AMX 指令（近几年的 Intel 都有）。老机器上 INT8 可能原地踏步甚至更慢——先跑 FP32 基准，量化后对比再决定用哪个
2. 形态 A 的输出含义不变，量化不影响后处理代码
3. OpenVINO 迭代快，直接 `pip install openvino nncf` 装新版即可；老教程里的老工具名（mo/pot）已废弃

---

## 6.6 路线 H：地平线 RDK X3 / X5（OpenExplorer 工具链）

**你需要准备**：`best.onnx`、校准图、Docker
**最终产物**：`.bin`（hbmodel）→ RDK 板上运行

- **工具**：OpenExplorer（Docker 镜像，地平线开发者社区 `developer.horizon.cc` 注册下载），也是 6.0 的 Docker 套路
- **转换**：`hb_mapper makertbin` 把 ONNX 转成 `.bin`，输入尺寸/校准路径写在随行的 yaml 配置里，转换时自动做 PTQ 定标
- **混合精度**：转换模式可选，把敏感层留 fp16（对应工具里的 mix/混合模式）
- **板端**：RDK 系统自带运行时——命令行用 `hrt_model_exec` 看模型信息/跑单帧，程序里用 C API 或 ROS 的 dnn node
- **官方样例**：地平线开发者社区/官方仓库有 yolov5、yolov8 的完整移植示例，**照着样例改**，别从零配置

**该路线的坑**：① 检测头可能要按官方样例微调结构（BPU 对部分算子支持有限，样例里怎么改就怎么抄）；② SDK、文档、镜像都在开发者社区账号体系内，先注册再干活。

---

## 6.7 路线 V：全志 V853 / 晶晨 A311D（Verisilicon 系 NPU）

**你需要准备**：`best.onnx`、校准图、厂商 SDK / Docker 镜像
**最终产物**：`.nb` 模型 → 板上厂商 runtime 运行

这两家的 NPU 都源自 Verisilicon 的 IP，工具链思路一致：**ONNX → 带校准的转换 → `.nb` 模型**，全流程在厂商提供的工具容器里完成。

- **全志 V853**：约 1 TOPS 级小算力。工具链在 Tina/Longan SDK 里（全志开发者社区文档），板端用 AWNN runtime。imgsz 建议 160~320、模型 n。
- **晶晨 A311D**（Khadas VIM3 等）：5 TOPS。用 Amlogic NPU SDK（Khadas 文档提供 Docker 镜像），转换工具为 ovxlib 系。

**该路线的坑**：① 文档分散在厂商 wiki 和 SDK 压缩包里，搜"模型转换 + 量化"章节照做；② demo 的后处理代码按 SDK 示例改；③ 校准时的预处理参数与板端运行时配置保持一致（老规矩，第1章铁律）。

---

## 6.8 路线 S：海思（新旧两代，天差地别）

**先分清你的芯片属于哪一代**——这决定了这是"普通路线"还是"天坑路线"：

| 世代 | 芯片 | 工具链 | 评价 |
|------|------|--------|------|
| 新：SVP_NPU | Hi3516DV500 / Hi3519DV500 等 | SDK 内 ONNX 直转工具 | 流程与主流 NPU 一致，正常走 |
| 旧：NNIE | Hi3516CV300/DV300、Hi3519AV100 等 | RuyiStudio + NNIE mapper，**只支持 Caffe** | ONNX→Caffe→.wk，YOLO 头要手工拆解排布，成本极高 |

> ⚠️ **避坑结论：新项目直接选 DV500 系列。** 旧 NNIE 只在维护存量产品时才碰，且优先找现成的移植样例抄，不要试图自己从 ONNX 硬啃。

**该路线的坑**：海思资料基本都在封闭 SDK 和代理商渠道里，拿到配套 SDK 版本才有对应文档；先跑通 SDK 自带 demo，再换自己的模型。

---

## 6.9 路线 B：算能 SE 系列（BM1684 / BM1684X / BM1688，TPU-MLIR）

**你需要准备**：`best.onnx`、校准图文件夹、Docker
**最终产物**：`.bmodel` → 板上 sophon-sail / bmrt 运行

工具链是 **TPU-MLIR**（GitHub `sophgo` 组织下载 Docker 镜像），三步曲结构非常清晰：

```bash
# ① ONNX → MLIR 中间表示
#    ★ ONNX 必须用 --model_def（--model_file 是给 Caffe 权重用的，写错会报找不到文件）
#    ★ 这一步没有 --processor 参数——处理器只在 model_deploy 里指定
#    ★ 示例输入 320，与 §3.3 训练一致；大算力盒子想用 640 就全程 640
model_transform.py --model_name yolo --model_def best.onnx \
  --input_shapes "[[1,3,320,320]]" --mean "0,0,0" --scale "0.00392,0.00392,0.00392" \
  --keep_aspect_ratio --pixel_format rgb \
  --output_names "/model.23/Concat_output_0,/model.23/Concat_1_output_0,/model.23/Concat_2_output_0"
#    用形态 B 时写上面这三个输出名（YOLO11；v8 是 /model.22/Concat_1..3_output_0）；
#    用形态 A 就删掉 --output_names，让它自己取最终输出

# ② 校准：对校准图跑统计，生成定标表
run_calibration.py yolo.mlir --dataset ../calib_images --input_num 100 -o yolo-calib.table

# ③ 部署编译：吃下定标表，吐出 INT8 bmodel
#    --processor 按芯片填，且只在这里填：bm1684 / bm1684x / bm1688
model_deploy.py --mlir yolo.mlir --quantize INT8 --calibration_table yolo-calib.table \
  --processor bm1684x --model yolo_int8.bmodel
# （参数细节以你版本的 TPU-MLIR 文档为准，官方还有 yolov5/yolov8 移植样例可抄）
```

板端推理用 **sophon-sail**（Python/C++ 接口，体验接近 onnxruntime）或 bmrt C API。

**该路线的坑**：① `--processor` 与芯片必须匹配，三种芯片的 bmodel 互不通用；② 官方移植样例（sophgo 仓库 samples）优先照抄；③ SE 盒子注意散热和功耗墙，长时间满载会降频。

---

## 6.10 第6章小结：所有 NPU 路线其实是同一条路线

走到这里你会发现，九张卡的骨架完全一致：

```
ONNX → (写配置: 输入shape/预处理/校准集) → (Docker里) 量化+编译 → 厂商格式模型
     → 板端 runtime + 官方 demo 后处理 → 跑起来 → 和 FP32 基线对比
```

差异只在：**工具叫什么名字、配置文件什么格式、混合精度怎么开**。所以换平台时，你真正要读的只有厂商的"模型转换"一节文档——概念全部都在本教程里了。

---

# 第7章 验证与调优 —— 证明量化没把模型搞坏

部署跑起来 ≠ 部署成功。这一章把"模型在板上到底行不行"变成可量化的事实。

## 7.1 上板验证三板斧

核心思想：**把量化模型和 FP32 基线放在同一张考卷上对比**。基线分数在第3章末尾已经记下了（比如 val mAP50-95 = 0.87）。

**第一板斧：金标准图对比（定性，5 分钟，必做）**

拿第2章留的 10~20 张现场实拍图（从未参与训练），同一张图分别让电脑上的 FP32 模型和板端量化模型推理，肉眼对比四件事：

| 检查项 | 健康表现 | 异常信号 |
|--------|---------|---------|
| 检出数量 | 基本一致 | 漏检明显增多 / 凭空多框 |
| 框的位置 | 贴合度相同 | 框整体偏移、抖动 |
| 类别 | 一致 | **全识别成同一类**（→ §5.4 预防针） |
| 置信度 | 差距 < 0.1 | 集体跳水（0.9 → 0.3）或虚高 |

**第二板斧：逐层精度分析（定位用）**

| 路线 | 工具 | 产出 |
|------|------|------|
| M | Pulsar2 `precision_analysis: true`（config 里已开） | 量化前后端到端误差报告 |
| R | `rknn.accuracy_analysis(...)` → snapshot 报告 | 每层余弦相似度 |
| J | polygraphy / trtexec 逐层 dump | 逐层输出对比 |
| X | NNCF / ORT 量化 debug 输出 | 敏感度信息 |

看什么：**哪一层的相似度明显塌陷（<0.9）**，它就是敏感层——这是 7.3 混合精度要"保"的对象。

**第三板斧：全量 mAP 对比（定量，可选，正式交付建议做）**

严谨做法（值得花这半天，正式交付必做）：

1. **板端**写个小脚本遍历 val 集，把每张图的检测结果写成 YOLO txt（`class cx cy w h conf`，坐标按原图尺寸归一化）
2. **电脑上**用同一套评测代码（比如 ultralytics 的 `DetectionValidator`，或 pycocotools）算 mAP，与 FP32 的 0.87 对比
3. 判定标准同手册 §0.2：**掉幅 < 3%**

只做前两板斧也能干活，但那时你只有"看起来差不多"的定性结论，没有能写进验收单的数字。

## 7.2 阈值重标定（量化后必做，常被忽略）

INT8 后置信度分布整体漂移，FP32 上调好的 `conf=0.25` 拿到板上可能满屏误框、也可能大面积漏检。**阈值必须在量化后的模型上重调**——注意是量化模型，不是 `best.pt`：

```python
# 情况一：形态 A（单输出）的量化模型 —— ultralytics 能直接加载 ONNX/OpenVINO 后端并跑 val
from ultralytics import YOLO

m = YOLO("best_int8.onnx")        # 或 best_int8.xml；★ 不要写成 best.pt
for th in (0.25, 0.35, 0.5, 0.6):
    r = m.val(data="dataset.yaml", imgsz=320, conf=th)   # imgsz 必须和训练一致
    print(f"conf={th}  P={r.box.mp:.3f}  R={r.box.mr:.3f}  mAP50={r.box.map50:.3f}")
```

```python
# 情况二：形态 B（三尺度）的 NPU 模型 —— 电脑上跑不了 val，只能在板上扫
# 板上改 conf_th，各跑一遍固定的一批金标准图，记录「检出数 / 误框数 / 漏检数」，
# 挑误检与漏检代价最平衡的那个阈值，写死进代码
```

> ⚠️ 拿 `best.pt` 扫出来的阈值只是 FP32 的参考值。INT8 模型的置信度整体偏低或偏高，直接套用是"板上满屏乱框"的常见原因。

选择原则是业务问题：**漏检代价高 → 阈值调低；误检代价高 → 阈值调高**。`iou_th`（NMS 用，默认 0.45）一般不用动。

## 7.3 掉精度的处理阶梯（从便宜到贵，逐级做）

| 级 | 手段 | 成本 | 说明 |
|----|------|------|------|
| 1 | **换校准算法**：MinMax → KL/Entropy（或反向） | 改一个词重转 | 各工具一个参数的事（手册 §4.2） |
| 2 | **换/补校准集**：更贴部署场景、覆盖所有类 | 半天 | §5.3 四原则重选 |
| 3 | **混合精度**：把第二板斧找到的敏感层强制 FP16/FP32 | 半天 | 各平台操作对照表：手册 §4.4 |
| 4 | **升级模型/尺寸**：n→s 或 320→416 重训 | 1~2 天 | 小模型量化敏感（§3.2） |
| 5 | **QAT** 量化感知训练 | 最后手段 | 手册 §4.7 |

每做完一级，回到 7.1 重新验证，达标即停。

## 7.4 FPS 不达标怎么办

先分锅：测**纯推理耗时**和**整帧耗时**。新手常常发现推理只占 30ms，剩下 50ms 花在取流、格式转换、画框、显示上——那优化方向完全不同。

按性价比排序的提速手段：

1. **减小 imgsz**（收益平方级）—— 640→320 直接 4 倍
2. 换更小模型（n 已经是最小就砍类别数/减通道，进阶）
3. 用平台流水线特性：MaixPy `dual_buff`、RKNN 多核绑定/zero-copy、Jetson 开 FP16+PDL
4. 检查输入通路：相机直出目标尺寸（免缩放）、零拷贝喂 NPU
5. 剪枝/蒸馏——进阶话题，先把上面用尽

---

# 第8章 卡住了怎么办

## 8.1 先自查四件事（90% 的部署 bug 在这里）

按顺序问自己三遍：

1. **预处理四问**：颜色序是 RGB 吗？归一化恰好做了一次吗？输入尺寸和训练一致吗？**几何变换一致吗（letterbox 还是拉伸）**？
2. **输出解释**：后处理代码对应的输出形态对吗（A/B 搞反是经典事故）？labels 顺序 = dataset.yaml 的 names 吗？
3. **版本配套**：转换工具和板端 runtime 是配套版本吗（路线 R 尤其）？

## 8.2 按症状跳转手册

| 症状 | 去处 |
|------|------|
| 转换阶段报"算子不支持"/转换失败 | 手册 §5.1 |
| 板端加载模型失败/崩溃 | 手册 §5.4 |
| 精度崩 / 全识别成一类 / 乱框 | 手册 §5.2 + §4.5 排查树 |
| FPS 明显低于预期 | 手册 §5.3 |
| 输出 shape 对不上 | 回 §4.4：把打印出的输出名/shape 逐字抄进配置 |

## 8.3 求助信息模板

**只维护一份**，在《工程手册》§8：把「训练 / 导出 / 部署 / 量化 / 症状 / 已试 / 自查」七行填全，附**完整报错日志**和**一张能复现的图**。

> 「自查」那行对应手册 §0.3 的五条铁律——写上它，回答者能直接跳过 80% 的低级嫌疑。本教程不再复制一份模板，避免两边版本不一致。

---

# 附录A：全平台速查表

**已合并到《工程手册》表1（部署基础）与表2（量化验证）**，本教程不再维护第二份。

需要一眼扫全部路线时去查手册；需要跟做时留在第6章的路线卡。两张表若有出入，**以手册为准**（单一事实源）。

# 附录B：常用入口

- ultralytics 文档：`docs.ultralytics.com`
- Netron（看 ONNX 结构）：`netron.app`
- Pulsar2 镜像：HuggingFace `AXERA-TECH/Pulsar2`（Sipeed 文档也给了 ModelScope 入口）；MaixPy 文档：`wiki.sipeed.com/maixpy`
  - MaixCAM2（Pulsar2，目标平台写 `AX620E`）：`wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam2.html`
  - MaixCAM 一代（算能 TPU-MLIR）：`wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html`
- RKNN 工具：GitHub `rockchip-linux/rknn-toolkit2`；后处理样例：`airockchip/rknn_model_zoo`
- 地平线开发者社区：`developer.horizon.cc`
- TPU-MLIR：GitHub `sophgo` 组织
- Docker Desktop：`docker.com/products/docker-desktop`

# 附录C：三份文档的关系

```
本教程（新手跟做） ──遇到问题──▶ 工程手册（决策树/速查表/排障/量化方法论）
       │
       └──路线 M 的出处──▶ yolo-to-maixcam2-workflow.md（已验证实战记录）
```

---

**🎉 走到这里，你应该已经：训出 best.pt → 导出验证过 ONNX → 量化成平台模型 → 板上实时跑 → 用数据证明它没坏。**
**换任何新硬件时，回来翻第1章决策树和第6章路线卡就够了。**
