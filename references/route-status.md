# 路线可信度清单

> 为什么要这张表：一份"全平台"资料最容易出问题的地方，是把**作者没跑过**的路线写得和跑过的一样笃定。这里给每条路线打标签，请按标签取用。
>
> 图例：
> - ✅ **实测**：作者完整跑通过（训练→转换→上板→验收）
> - 🟡 **文档对齐**：命令/参数逐条对照官方文档核对过，但未在本资料作者手上跑通
> - 🔴 **仅索引**：只有二手资料与官方入口，**当作"去哪查"用，不要当操作手册**

| 路线 | 硬件 | 标签 | 依据 / 权威入口 | 已知坑（必读） |
|---|---|---|---|---|
| **M** | MaixCAM2（AX630C） | ✅ 实测 + 🟡 官方校正 | [Sipeed MaixCAM2 转换](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam2.html) | `--target_hardware` 写 **AX620E**；必须形态 B；AI-ISP 需另编 NPU1 的 `model_vnpu` |
| **R** | RK3588/3566/3568/3576/RV1103/1106 | 🟡 文档对齐 | [RKNN-Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) · [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) | Toolkit2 ↔ 板端 `librknnrt.so` 版本必须配套；`accuracy_analysis` 不能单独跑 |
| **J** | Jetson Orin / Xavier | 🟡 文档对齐 | [Ultralytics TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) · NVIDIA DeepStream | engine 绑定 GPU 架构 + TRT 版本，不可跨板分发 |
| **C** | 树莓派 / 安卓 / 无 NPU ARM | 🟡 文档对齐 | [Ultralytics NCNN](https://docs.ultralytics.com/integrations/ncnn/) · `Tencent/ncnn` `examples/` | 产物叫 `model.ncnn.param/.bin`；校准表参数必须与推理预处理逐字一致 |
| **X** | x86 CPU / Intel 核显 | 🟡 文档对齐 | [Ultralytics OpenVINO](https://docs.ultralytics.com/integrations/openvino/) · `openvinotoolkit/nncf` | INT8 提速依赖 VNNI/AMX，老 CPU 可能更慢 |
| **H** | 地平线 RDK X3 / X5 | 🔴 仅索引 | [Horizon 开发者社区](https://developer.horizon.cc)（需注册）· `horizonrobotics` on GitHub | 检测头常需按官方 yolov5/v8 样例微调；SDK 在账号体系内 |
| **V** | 全志 V853 / 晶晨 A311D | 🔴 仅索引 | 全志开发者社区 · Khadas Docs（Amlogic NPU SDK） | 文档散在 SDK 压缩包与 wiki；demo 后处理按 SDK 改 |
| **S** | 海思 DV500（新）/ Hi3516-3519 NNIE（旧） | 🔴 仅索引 | 海思 SDK / 代理商；NNIE 用 RuyiStudio | 旧 NNIE 只吃 Caffe，YOLO 头要手工拆，成本极高；新项目直接选 DV500 |
| **B** | 算能 SE 盒子（BM1684/1684X/1688）、**MaixCAM 一代（SG2002）** | 🟡 文档对齐 | [Sophgo TPU-MLIR](https://github.com/sophgo/tpu-mlir) · [MaixCAM 转换](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html) | `--processor` 三种芯片互不通用；MaixCAM 一代产物是 `.cvimodel`，不是 `.bmodel` |

## 一个容易混的点：芯片型号 ≠ 工具链平台名

| 设备 | 芯片 | 工具链 | 平台/目标名 |
|---|---|---|---|
| MaixCAM / MaixCAM-Pro | 算能 **SG2002** | 算能 TPU-MLIR | `cvimodel` |
| MaixCAM2 | 爱芯 **AX630C** | Pulsar2 | **`AX620E`**（固定） |
| M5Stack Module LLM 等同芯设备 | 爱芯 AX630C | Pulsar2 | 按厂商文档，通常同为 AX620E |

> 这条如果搞错，现象是"模型编译成功了但板子加载失败"，很容易怀疑到别的地方去。

## 想给这张表升级标签？

把"你把哪条路线跑通了"的证据补进来（版本号 + 完整命令 + 成功日志 + 一个跑通的检测结果），我们就把 🔴 → 🟡 → ✅ 往上提。标签只升不降，除非有人复现失败。
