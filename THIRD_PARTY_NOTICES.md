# 来源与独立许可

- 原论文与模型方法：[Native and Compact Structured Latents for 3D Generation](https://arxiv.org/abs/2512.14692)，Microsoft TRELLIS.2 作者团队。本仓库不主张这些方法的原创权。
- 官方代码和示例：[microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)，[MIT](https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE)，Copyright (c) Microsoft Corporation。官方示例衍生的推理与导出部分保留其许可声明于根目录 LICENSE。
- 模型下载默认仓库：[microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)、[microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large)。以各模型卡条款为准。
- 图像条件模型：[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)。当前下载脚本沿用第三方 ModelScope 镜像，仍应遵守原模型许可和访问条件，并记录具体镜像版本。
- 背景移除：[briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)。访问、使用和再分发条件独立于本仓库 MIT；运行前取得所需授权。
- 渲染与 CUDA 扩展等依赖（包括 nvdiffrast、nvdiffrec、CuMesh、FlexGEMM、o-voxel）、Python 库和 PyBullet 遵循各自许可；本仓库 MIT 不为第三方软件或模型重新授权。

输入图片、纹理与生成案例的来源/授权需由添加者另行注明。测试中的临时图片与小网格均由程序构造，不从网络采集。
