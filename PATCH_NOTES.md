# 维护修复（2026-10-03）

## 文档与安装

- 明确非官方预训练推理复现的范围、真实 GPU 验收状态及模型/脚本来源。
- 克隆链接与署名更新为 EatWhiteRice；添加 MIT 和第三方说明。
- 安装包装器调用固定版本的官方 setup.sh，递归获取子模块；移除不成立的 pip install -e . 和未经配置的 attention 自动回退说明。
- CPU 测试和可选 PyBullet 安装分别列出。当前官方 README 的 Linux / 至少 24GB 要求替代未经验证的显存断言。

## 行为修复

- generate.py：先解析 CLI，再查模型和导入 torch；--help 不需要 GPU。移除未生效的 --no-pbr；保留 alpha；CUDA 同步计时；记录 metadata；可选保存 simplify/remesh 前几何。保持完整 pipeline 的模型/采样入口，不加入训练或新生成方法。
- 下载配置：不再重写归一化/采样器数值；保留官方 config，只本地化路径。HF 下载解析并记录 revision；DINO 镜像版本仍待固定。下载前缺依赖明确报错，不在异常分支偷偷 pip 安装。
- 评测：读取纹理失败明确抛出 ValueError，不返回零误差或假统计。experiment.py 是统计函数，批量入口尚未实现。
- 后处理：颜色强度零不改颜色；保留 metallic/roughness；pipeline_post.py 转为同实现的旧入口。
- 仿真：场景全部部件和变换进入预处理；GLB Y-up 转为仿真 Z-up、导出转回，非标准输入可覆盖；physics_sim.py 明确单凸包近似并传入面索引，移除未实际使用的 VHACD；sim_pipeline.py 仍是盒子近似，碰撞和显示框中心与网格匹配。

## 验证与剩余风险

12 项 CPU/mock 回归测试通过，并完成独立代码复核。测试不依赖真实模型、torch 或 CUDA，和真实推理严格区分；后者没有在这次维护中执行。官方 CUDA 安装仅核对入口和语法，没有编译验收。真实 GLB 导出、设备相关确定性和 PyBullet 运动精度需要对应环境验证。metadata 的数值和大小描述以 README 为准，不能从 CPU/mock 结果推断模型质量。

结构停止/保存、完整 SLAT dump、分阶段重采样和真实关节失败预测均未实现。
