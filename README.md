# TRELLIS.2 非官方推理复现与工程实践

基于 Microsoft **[TRELLIS.2](https://github.com/microsoft/TRELLIS.2)** 的预训练模型，将单张图片转换为带 PBR 材质的 3D GLB 资产，并提供后处理、运行信息记录和实验性刚体仿真脚本。

原论文：[Native and Compact Structured Latents for 3D Generation](https://arxiv.org/abs/2512.14692)。核心模型、表示方法和训练成果属于原作者。本仓库整理推理与工程流程，不包含论文训练复现或全套指标复现，也不主张新的模型方法。

## 已实现与验证范围

| 内容 | 范围 |
|---|---|
| `generate.py` | 预训练推理、GLB 导出、metadata、可选原始几何保存；完整生成仍需真实 GPU 验收 |
| `download_weights.py` | 显式下载模型、记录 HF revision、仅修改本地路径并保留原配置 |
| `post_process.py` | 几何平滑、颜色统计校正；保留 metallic/roughness，强度零不改色 |
| `experiment.py` | 实验性统计函数，读取失败明确报错；**不是批量实验入口** |
| `physics_sim.py` / `sim_pipeline.py` | 实验性整物体刚体下落：单凸包 / 包围盒碰撞近似；没有骨架或活动关节 |
| `tests/` | CPU / mock 回归测试；不验证真实模型生成质量或 PyBullet 运动精度 |

当前仓库没有附上可核验的完整真实推理日志或生成案例。若添加复现结果，请同时提供实际硬件、版本、输入来源、seed、配置和运行日志。测试通过不能代替模型推理复现。

## 环境与版本

官方当前说明：仅在 **Linux** 上测试，NVIDIA GPU 至少 **24GB 显存**，验证过 A100 / H100；推荐 CUDA Toolkit 12.4，官方新环境使用 torch 2.6.0 / CUDA 12.4。Windows 原生运行、本机 8GB 显存以及各分辨率的具体峰值未在这里验收，`low_vram` 不是低显存可运行保证。

本仓库安装入口固定官方源码 commit：`75fbf0183001ed9876c8dbb35de6b68552ee08bd`。这固定核心源码，不代表所有第三方依赖已完整锁定。成功安装后建议保存 `python -m pip freeze > environment-lock.txt`，同时记录 CUDA Toolkit、驱动与子模块版本。

## 安装（Linux，具备 conda 和 CUDA 开发工具）

```bash
git clone https://github.com/EatWhiteRice/trellis2-3d-generator.git
cd trellis2-3d-generator
bash setup.sh --new-env
conda activate trellis2
```

脚本递归克隆固定版本，调用官方 `setup.sh` 安装 CUDA 扩展，再安装额外依赖。官方源码不支持这里旧版使用的 `pip install -e .` 安装方式；单独安装 `requirements.txt` 也不能替代官方 CUDA 扩展安装。

已有环境可以省略 `--new-env`。已有官方 checkout 可设置 `TRELLIS2_ROOT`；如果它的 commit 与默认固定版本不同，脚本会停止。检查版本后可通过 `TRELLIS2_REF=<完整commit>` 明确采用该版本，并自行验证兼容性。脚本不会静默覆盖另一版本的 checkout。

## 显式下载模型

```bash
python download_weights.py
```

预计约 20GB，实际容量取决于版本；默认缓存为 `~/.cache/trellis2/`，可用 `TRELLIS2_CACHE` 改位置。安装和测试不会调用下载脚本。

TRELLIS.2 与结构解码器使用 Microsoft 官方 HF 模型仓库；下载时将 revision 解析为确切 commit，并写入 `pipeline/download_manifest.json`。重跑相同 checkpoint 可设置 `TRELLIS2_MODEL_REVISION` 和 `TRELLIS_IMAGE_REVISION` 为记录中的 SHA。`pipeline.official.json` 保留未经路径修改的配置，采样器和归一化数值不重新手写。

本脚本沿用原有 ModelScope DINOv3 镜像路径；该镜像的 revision 尚未自动固定，应额外记录实际版本/文件哈希，或自行配置获授权的官方 DINOv3 本地模型。模型授权以原模型条款为准，镜像不代表可免除原条款。

RMBG-2.0 需要先在 [HF 模型页](https://huggingface.co/briaai/RMBG-2.0) 获得访问授权并配置 `HF_TOKEN`。未成功下载时，预处理仍可能尝试访问 gated 上游模型。首次真实推理不能据此保证完全离线。需要镜像时由使用者显式设置 `HF_ENDPOINT`，脚本不强制更改端点。

## 生成与记录

先将你有权使用的图片放入本地目录（仓库不附 `examples/chair.jpg`）：

```bash
python generate.py examples/your_image.png -o outputs/case001.glb --seed 42 --steps 12 --save-raw
```

输出：

- `case001.glb`：经过 simplify / remesh / 材质导出的资产。
- `case001.metadata.json`：输入/配置哈希、源码 commit、git dirty、seed、步骤、设备、torch/CUDA、分阶段耗时和峰值显存。
- `case001.raw_geometry.npz`：仅在 `--save-raw` 时保存，包含处理前 vertices/faces；**不是完整 SLAT、纹理或第一阶段 checkpoint**。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `-o, --output` | GLB 输出路径 | `output.glb` |
| `-r, --resolution` | 512 / 1024 直接管线 | 512 |
| `--steps` | 各生成阶段的采样步数，必须大于零 | 12 |
| `--seed` | 全局随机 seed；不保证跨设备/版本数值完全一致 | 42 |
| `--trellis-root` | 官方 checkout 路径，也可设 `TRELLIS2_ROOT` | 自动查找 |
| `--save-raw` | 保存处理前几何，可能增加磁盘和 CPU 内存消耗 | 关闭 |

旧的 `--no-pbr` 从未实际生效，现已移除。脚本保留输入 alpha；元数据计时通过 CUDA 同步区分生成和导出。峰值显存覆盖重置后的生成/导出阶段，不包括此前的模型加载峰值，原始几何保存耗时不计入这两个阶段。

## 可选后处理与仿真

```bash
python post_process.py outputs/case001.glb --calibrate --original examples/your_image.png --color-strength 0.3
python -m pip install -r requirements-sim.txt
python physics_sim.py outputs/case001.glb --output outputs/drop.glb
```

颜色校正是通道均值/标准差匹配，不能保证真实材质还原；几何平滑会改变细节。比较模型几何时请使用 raw dump 和处理后资产分别评价。仿真默认将标准 GLB 的 Y-up 转成 Z-up，导出时转回标准 Y-up；非标准 Z-up 输入可用 `--input-up z`。模型归一化到指定大小，质量固定为 0.15kg；碰撞使用简化近似。它不证明机械臂关节、人体骨架或可编辑部件正确。

`pipeline_post.py` 保留为旧 CLI 入口，调用同一后处理实现。

## CPU 回归测试

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python generate.py --help
```

测试使用原创小网格、临时图片和 mock 外部接口；不安装 torch/CUDA 扩展，不加载模型，不下载权重。覆盖评测错误处理、配置精度、CLI、颜色/PBR 保持、场景部件与 Z-up 预处理、碰撞参数。

## 下一步研究接口

当前包装脚本仍直接运行完整 pipeline。要检查“连接处问题是否在 SLAT 前出现”，需进一步按实际 TRELLIS.2 版本导出 `sample_sparse_structure` 的 coords，记录粗格子分辨率并对照 raw / export 几何。这里没有实现第一阶段独立加载或结构停止模式；跳过 SLAT 也不自动等于低显存能运行。

## 许可与来源

本仓库包装代码采用 [MIT](LICENSE)。官方示例相关部分保留 Microsoft 版权；核心模型、官方代码及其他依赖的来源与独立条款见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
