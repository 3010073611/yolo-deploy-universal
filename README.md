# YOLO 训练 → 量化 → 部署（全平台通用）

一份**不绑定任何硬件**的 YOLO 落地资料：从数据准备、训练、导出 ONNX，到 INT8 量化、上板部署、精度验收与排障。同时是给 AI Agent 用的 **DSH Skill**——人直接读也完全够用。

覆盖 9 条路线：MaixCAM2（爱芯 AX630C）、Rockchip RKNN、Jetson、ARM CPU（NCNN/MNN）、x86（OpenVINO/ORT）、地平线、全志/晶晨、海思、算能 TPU。

---

## 这份资料解决什么问题

边缘部署的坑很少在"模型"，而在**四件事对不上**：

1. 预处理（颜色序 / 归一化 / 布局 / letterbox）训练与推理不一致
2. `imgsz` 训练、导出、部署三处不同
3. `labels` 顺序与 `dataset.yaml` 的 `names` 错位
4. 转换工具与板端 runtime 版本不配套

所以这里的组织方式是：**先讲清概念 → 给可执行的路线卡 → 用脚本把"抄错名字/写错顺序"这类错误机器化地拦掉**。

## 目录结构

```
.
├── SKILL.md                  # Agent 入口：铁律、路线速选、文档地图（人也可以先读这个）
├── references/
│   ├── tutorial.md           # 新手跟做主线（数据→训练→导出→量化→九张路线卡→验收）
│   ├── handbook.md           # 工程手册：决策树 / 速查表 / 量化方法论 / 排障树 / 求助模板
│   ├── route-status.md       # 每条路线的「已验证 / 仅二手资料」标签 + 权威文档入口
│   └── changelog.md          # 相对原始文档改了什么、依据是什么
├── scripts/
│   ├── lint_deploy_config.py # ★ 用 ONNX 真实输入/输出名核对 config.json / .mud / dataset.yaml
│   ├── check_onnx_vs_pt.py   # ★ 导出自检：.pt 与 .onnx 同图输出误差 < 1e-3
│   └── split_dataset.py      # 分层划分（每类都进 val）+ 支持无 txt 的背景图
└── assets/
    ├── pulsar2-maixcam2.json # Pulsar2 配置模板（MaixCAM2 / AX620E）
    ├── rknn-convert.py       # RKNN-Toolkit2 转换 + 校准脚本模板
    └── model.mud             # MaixPy MUD 描述文件模板
```

## 怎么读（按你的角色）

| 你是 | 读这个 |
|---|---|
| 第一次把 YOLO 放到板子上 | `references/tutorial.md`，按第 1→8 章顺序走，只读你那一条路线卡 |
| 已经上板，但精度/FPS 不对 | `references/handbook.md` §4（量化）、§5（排障树） |
| 要选硬件 | `SKILL.md` 的路线速选表 + `references/route-status.md` |
| 想快速判断"我的配置有没有写错" | 直接跑 `scripts/lint_deploy_config.py` |

## 作为 DSH Skill 安装

把这个仓库放到 DSH 的 skill 根目录即可被自动发现（`SKILL.md` 会被解析，正文每次加载都重新读取，改完不用重启）：

```powershell
# 方式 A：直接放进用户级 skill 目录（所有项目可用）
git clone https://github.com/3010073611/yolo-deploy-universal.git "$env:USERPROFILE\.dsh\skills\yolo-deploy"

# 方式 B：仓库放在别处，用目录联接指过去（Windows 不需要管理员权限）
New-Item -ItemType Junction -Path "$env:USERPROFILE\.dsh\skills\yolo-deploy" `
  -Target "D:\code\yolo-deploy-universal"
```

> 也支持项目级安装：`<仓库根>/.dsh/skills/yolo-deploy/`。

## 仓库地址与日常更新

公开仓库：<https://github.com/3010073611/yolo-deploy-universal>

```powershell
cd $env:USERPROFILE\.dsh\skills\yolo-deploy
git add -A
git commit -m "docs: 修正 XXX"
git push
```

> 从零重建远端时：在 GitHub 建**空仓库**（不要勾 README/gitignore/license），然后
> `git remote add origin https://github.com/<账号>/<仓库>.git && git push -u origin main`。

## 快速开始（脚本）

```powershell
# 0) 依赖（按需）
pip install onnx onnxruntime numpy pyyaml
# 自检脚本还需要: pip install ultralytics torch opencv-python

# 1) 分层划分数据集（每类都保证进 val，背景图不会让脚本崩）
python scripts/split_dataset.py --src raw --dst dataset --val-ratio 0.1

# 2) 导出后立刻自检：ONNX 和 .pt 必须对得上（< 1e-3）
python scripts/check_onnx_vs_pt.py --pt best.pt --onnx best_3scale.onnx --image scene/1.jpg --imgsz 320

# 3) 转换前先校验配置：输出名、层号、labels 顺序一次查完
python scripts/lint_deploy_config.py --onnx best_3scale.onnx ^
    --config assets/pulsar2-maixcam2.json --mud assets/model.mud --data dataset.yaml
```

## 验证程度（重要）

这份资料的**可靠性是不均匀的**，请按 `references/route-status.md` 的标签取用：

- **M（MaixCAM2）**：作者实跑过 + Sipeed 官方文档校正
- **R / J / C / X**：通用做法，命令按官方文档核对过，但未逐条实测
- **H / V / S**：只有二手资料与官方入口，**当作索引，别当手册**
- **B（TPU-MLIR）**：命令按官方文档校正过，未实测

遇到与本资料和厂商官方文档冲突的地方，**以厂商文档为准**，并欢迎提 issue 修正。

## 相对原始版本的修订

本仓库源自两份个人笔记，开源前做了一次系统性校对，修掉了 20 余处会让人"照着做失败"的问题（MaixCAM 一代工具链误判、`--target_hardware` 写错、TPU-MLIR 参数名、ncnn/MNN 产物名、阈值在错误模型上标定、letterbox 缺失等）。逐条列出改动与依据见 [`references/changelog.md`](references/changelog.md)。

## 贡献

- 报错/纠错：请附上**芯片型号 + 工具链版本 + 完整报错日志 + ONNX 输入输出名和 shape**（模板见 `references/handbook.md` §8）
- 新增平台：优先补一张路线卡 + 一段 `route-status.md` 条目，注明「已验证」还是「资料整理」
- 提交前请跑一遍 `scripts/lint_deploy_config.py`，确保示例配置自洽

## 致谢

- [Ultralytics](https://docs.ultralytics.com)（训练与导出）
- [Sipeed MaixPy](https://wiki.sipeed.com/maixpy)（MaixCAM / MaixCAM2 转换文档）
- [Rockchip RKNN-Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) / [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)
- [AXERA Pulsar2](https://huggingface.co/AXERA-TECH) / [Sophgo TPU-MLIR](https://github.com/sophgo/tpu-mlir) / [Horizon OpenExplorer](https://developer.horizon.cc)

## License

[MIT](LICENSE)
