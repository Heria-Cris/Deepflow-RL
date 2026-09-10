# DeepFlow RL

面向弱网络环境的端云协同大语言模型推理研究项目。

DeepFlow RL 将投机推理扩展为端云协同机制：边缘端的轻量草稿模型生成候选 Token，
仅向云端发送 Token ID；云端目标模型进行验证。该机制以轻量语义载荷替代传统分层
推理中的中间激活值传输，并通过微批流水线减少高时延网络下的等待开销。PPO 控制器
根据带宽、延迟、Prompt 长度和历史吞吐量，自适应选择协议模式、投机深度与微批大小。

## 仓库内容

- `core/`：设备、链路和 LLaMA 风格模型的解析式规格。
- `engine/`：Roofline 风格物理模型、统一仿真器、投机和流水线兼容接口。
- `rl/`：基于 Gymnasium 的 DeepFlow 强化学习环境。
- `configs/`：设备、网络、模型和显存安全系数的标准实验配置。
- `models/`：训练好的 PPO 模型和检查点；二进制模型由 Git LFS 管理。
- `CHART/`：论文图表生成脚本及其输出。
- `paper/`：投稿论文的 LaTeX 源文件、参考文献和所用图表。
- `run_paper_experiments.py`：统一论文实验与基线、PPO 对比入口。
- `analyze_memory_feasibility.py`：可行域、OOM 构成和显存压力分析入口。

## 研究范围

本仓库使用统一的解析式/离散事件仿真后端，对计算、网络、投机验证、流水线和显存
可行性进行一致建模。结果用于比较不同策略在相同假设下的相对性能；它不是实际端云
设备部署性能的直接测量。

## 快速开始

```powershell
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = '1'
python test_phase1.py
python test_phase2.py
python test_phase3.py
python test_phase4.py
python run_paper_experiments.py
python analyze_memory_feasibility.py
```

详细的研究背景、代码约束、实验语义、审稿修订流程与提交规则见
[`AGENTS.md`](AGENTS.md)。

## 版本管理

每次有效修改后均应检查差异、完成相应验证，并使用清晰中文提交信息进行本地完整
提交。训练模型、检查点和归一化文件只保留在本地完整版本中；远程发布仅推送不含
`models/` 的代码发布分支或副本。请勿提交访问令牌、密钥、环境变量文件、Python
缓存或 IDE 本地配置。
