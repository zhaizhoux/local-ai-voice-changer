# Local AI Voice Changer

Windows 本地 RVC 实时变声器，提供中文控制面板和虚拟声卡输出。

麦克风 → RVC 音色转换 → CABLE Input → 聊天或游戏中的 CABLE Output。

## 当前版本

`v0.1.0-beta.1` 是源码预览版，需要自行配置兼容的 Applio 引擎、Python 环境和音色模型。
本机验证过 GPU 推理、虚拟声卡传输和短时麦克风启停。音色自然度仍需针对输入声音调节。
开发期间使用的模型已训练 100 轮。后续 120 轮训练因独显不可用而未完成。
此发布包提供控制面板源码，音色权重、训练语料和驱动由使用者另行准备。

## 功能

- 选择本地音色、麦克风和输出设备。
- 调节半音、分块大小、检索比例、输出增益和静音门限。
- 显示输入电平、单块推理耗时和丢块数量。
- 播放预缓冲、有限长度音频队列和设备名称重新匹配。
- 停止或关闭后释放音频设备。
- 提供启动和桌面快捷方式脚本。

## 安装

1. 准备 Windows 10/11、Python 3.12，以及可运行 `rvc.realtime.core.VoiceChanger` 的 [Applio](https://github.com/IAHispano/Applio) 环境。实时使用推荐兼容的 NVIDIA CUDA 显卡。
2. 按 Applio 自身说明安装引擎依赖。控制面板还使用 `numpy`、`sounddevice` 和 Python 自带的 `tkinter`。仅安装本项目的 `requirements.txt` 无法替代引擎安装。
3. 在引擎目录准备 `rvc/models/embedders/contentvec/pytorch_model.bin`、对应 `config.json` 和 `rvc/models/predictors/rmvpe.pt`。
4. 将你的 RVC 推理模型放入本项目的 `models` 目录。
5. 复制 `runtime.example.json` 为 `runtime.json`，填写引擎目录和该环境的 `pythonw.exe`。配置支持相对项目目录的路径，也支持绝对路径。`RVC_HOME` 环境变量可覆盖 `engine_dir`。
6. 从 [VB-Audio 官网](https://vb-audio.com/Cable/) 安装 VB-CABLE，按厂商说明完成安装与重启。
7. 右键用 PowerShell 运行 `启动变声器.ps1`。需要桌面入口时运行 `创建桌面快捷方式.ps1`。

引擎接口参考版本为 `67a38f6abab3e90a55efa534ea3185a222c102f0`。引擎版本与接口变化可能影响兼容性，完整安装过程尚未在另一台干净电脑验证。

## 使用

输入选择实体麦克风，输出选择 `CABLE Input (VB-Audio Virtual Cable)`，优先使用 WASAPI。
点击“开始变声”，等待模型加载和预热，再在目标软件中选择 `CABLE Output` 作为麦克风。
目标软件的播放设备选择耳机或实体扬声器。耳机试听可避免扬声器回授。

音调从 `0` 开始，根据原声与目标音色调整。分块从 `200 ms` 开始，卡顿时尝试 `300` 或 `400 ms`。
该数值表示分块大小，实际端到端延迟还包含模型推理、预缓冲和音频设备延迟。
静音门限默认 `-60 dB`。轻声被切掉时降低门限。参数修改后停止并重新开始。

## 数据与日志

模型推理在本机执行。麦克风音频只在处理队列中传递，程序不会保存麦克风录音。
`runtime.json` 保存环境路径，`settings.json` 保存界面设置，日志写入项目目录。
这些运行时文件已加入 `.gitignore`。

## 验证范围

开发机上已验证 RVC 模型能输出有限、非静音音频，VB-CABLE 能传输测试信号。
调参对比使用既有测试语音完成，效果需要结合实际麦克风声音试听。
CPU 可以运行推理，开发机上的 CPU 耗时曾超过 200 ms 分块预算。
完整的长时间实时稳定性和跨机器安装仍需进一步验证。

## 许可证

本项目代码使用 MIT 许可证。Applio、PyTorch、模型和 VB-CABLE 分别遵循各自许可证。
第三方项目链接与来源记录见 `THIRD_PARTY_NOTICES.md`。
