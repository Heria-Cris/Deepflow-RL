# DeepFlow RL 项目上下文（供 Agent 使用）

## 项目用途与目录关系

本工作区服务于已投稿的系统论文 **《DeepFlow RL: Token-Based Speculative
Inference for Adaptive Edge-Cloud Collaborative LLM Serving under Weak
Network》**。后续工作将围绕审稿意见开展修订，必须始终保持论文、仿真器、
实验、图表和结论之间的一致性。

项目由两个并列目录组成；目前两处目录均没有可用的 Git 提交历史：

- 论文源文件：`D:\study\项目组\deepflow_rl\FGCS-main.tex`
- 实验代码：`E:\workspace\python\DeepFlow-RL`
- 论文使用的图表：`D:\study\项目组\deepflow_rl\fig*.pdf` 与 `fig*.png`
- 代码侧图表生成脚本和中间图：本仓库的 `CHART\` 目录。

修改论断、公式或实验数值前，必须先阅读 `FGCS-main.tex` 与相关实现。应将
论文源文件和本目录代码视为事实来源；已生成的 PDF/PNG 是输出，不能单独作为
证据。

## 研究问题与核心贡献

本研究关注弱网络条件下的端云协同 LLM 推理。传统的分层切分方法需要传输中间
激活值，在低带宽下会被通信开销主导。DeepFlow RL 使用部署在边缘端的草稿模型和
部署在云端的目标模型：边缘端发送投机生成的 **Token ID**，云端进行验证。微批
流水线将边缘草稿、网络传输和云端验证相互重叠。PPO 再根据吞吐、通信和显存
可行性自适应选择联合动作。

因此，本工作的核心比较不只是“在哪里切分”，更是“传输什么”：

- `P = 0`：DeepFlow 的 Token-ID 传输模式。
- `P > 0`：基于激活值传输的解析式传统切分基线；边缘端执行目标模型的前 `P`
  层，再发送其输出激活值。
- 策略还会选择投机深度 `K` 与微批大小 `MB`。

论文的主要结论都是在 **统一的解析式/离散事件仿真器** 上获得的比较性结论。本
仓库并未实现或基准测试真实的多设备 LLM 推理。不得将报告的 `tok/s` 表述为实际
部署测量性能，也不得把仿真参数默认为硬件实测结果。

## 论文结构

`FGCS-main.tex` 是单文件的 Elsevier 论文：

- 第 1--3 节：研究动机、相关工作、系统模型、Token/激活值载荷、Roofline 风格
  延迟模型、流水线完工时间与显存模型。
- 第 4 节：DeepFlow RL 架构、Token 投机协议、微批流水线调度器、PPO 控制器和
  端到端工作流。
- 第 5 节：仿真设置、基线、PPO 配置、性能、策略行为与资源感知分析。
- 代表性主场景为 `1 Mbps`、`50 ms`、提示词长度 `512`；论文当前报告 Best
  Static DeepFlow 与 PPO 均为 39.53 tok/s，最佳可行本地基线为 3.59 tok/s，云端
  独立执行为 23.05 tok/s，传统切分为 10.55 tok/s。11.01x 提升是相对于本地基线。
  修改图表或数值结论前必须重新计算确认。

主要实验以 Jetson AGX Orin 64GB 为边缘端、A100-PCIE-80GB 为云端，以
TinyLlama-1.1B 为草稿模型、Llama-2-7B 为目标模型。这些设备参数及模型行为均
为配置化的解析式参数，而不是运行时自动探测或测得的结果。

## 代码架构与实现约束

`engine/simulator.py` 是唯一的仿真事实来源。强化学习环境、独立验证脚本和基线
枚举都必须继续通过 `DeepFlowSimulator`，不得复制或分叉出另一套公式。

- `core/hardware.py`：校准后的 `Device` 与 `NetworkLink` 模型；有效算力和有效
  显存带宽通过配置的效率系数得到。
- `core/model_spec.py`：LLaMA 风格层的参数量、FLOPs、激活值和 KV Cache 大小公式。
- `engine/physics.py`：Roofline 风格单层延迟，取计算时间与权重显存访问时间的
  最大值。
- `engine/simulator.py`：阶段成本、投机有效产出、显存可行性与三资源离散事件
  流水线时间线。
- `engine/speculative.py`、`engine/pipeline.py`：兼容层；均委托给
  `DeepFlowSimulator`。
- `rl/envs/flow_env.py`：Gymnasium 环境与奖励函数。
- `train_phase5.py`：PPO 训练与检查点保存。
- `eval_phase5.py`：输出代表性场景下的确定性 PPO 决策。
- `run_paper_experiments.py`：论文基线搜索、PPO 对比、带宽扫描、策略行为与可行
  Oracle 诊断。
- `analyze_memory_feasibility.py`：动作空间枚举、可行比例、OOM 构成、显存比例和
  Prompt 长度分析。
- `CHART\Chart_*.py`：图表生成脚本。向论文目录复制图表前，必须先确认脚本输入
  数值的来源。

### 场景、动作与指标语义

- 观测状态：`[bandwidth_mbps, latency_ms, prompt_len, last_throughput]`。
- PPO 内部动作顺序为 `[mb_idx, k_idx, partition_point]`，与论文的 `(P, K, M)`
  记号顺序不同。
- `MB` 候选为 `[1, 2, 4, 8, 16, 32]`；`K` 候选为 `[0, 1, 3, 5, 7, 10]`；`P`
  取值为 `0..32`。总批大小固定为 32 时，动作空间共
  `6 * 6 * 33 = 1188` 个动作。
- 只有同一仿真器计算的边缘端与云端峰值显存均不超过配置预算，动作才可行。不
  可行动作的吞吐为零，并会标注 `edge`、`cloud` 或 `both` OOM。
- 单微批阶段成本由边缘端草稿和可选目标模型前缀、一次网络传输、云端验证构成。
  同一资源在时间线上串行，不同资源可在多个微批之间重叠。
- 每条序列的期望有效 Token 数为 `1 + sum(alpha_j)`；默认
  `alpha_j = 0.85 * 0.85^(j-1)`。这是用于系统层分析的校准投机产出模型，并非
  对真实草稿模型/目标模型逐 Token 轨迹的回放。
- 显存模型包含模型权重、草稿/前缀/后缀 KV Cache、激活值/工作区、框架开销、
  通信缓冲区与安全系数。标准参数位于 `configs/devices_paper.json`。

## PPO 与模型产物约束

训练使用 8 个采用域随机化的并行环境。带宽在 `[0.5, 100] Mbps` 上按对数均匀
采样，延迟在 `[5, 120] ms` 上均匀采样，Prompt 长度从 `128..8192` 中采样。
`train_phase5.py` 当前使用双层、每层 256 单元的 actor/critic，学习率 `2e-4`、
`n_steps=512`、批大小 `256` 和 350,000 个训练步数。状态和奖励通过
`VecNormalize` 归一化。

确定性评估脚本会在 `models/ppo_deepflow/best_model/` 中的模型与归一化文件均存在
时优先加载该目录，否则退回 `final_model.zip` 与 `vec_normalize.pkl`。模型必须始终
与其对应的 `VecNormalize` 统计文件配套使用。若重新训练导致环境、动作映射、
奖励、配置或归一化发生变化，必须重新生成实验结果并记录其来源。

## 标准实验流程

在 `E:\workspace\python\DeepFlow-RL` 目录运行：

```powershell
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = '1'
python test_phase1.py
python test_phase2.py
python test_phase3.py
python test_phase4.py
python eval_phase5.py
python run_paper_experiments.py
python analyze_memory_feasibility.py
```

Windows 的 GBK 终端下，现有中文和 emoji 控制台输出需要设置 `PYTHONUTF8=1`。截至
2026 年 9 月 10 日，本工作区已成功运行 Phase 1--3；当前解释器缺少 `gymnasium`，
因此 Phase 4 及依赖 PPO 的命令尚未在此环境中重跑，需先安装上述依赖。

## 版本管理与远程同步

本目录是本地完整 Git 仓库，远程仓库为
`https://github.com/Heria-Cris/Deepflow-RL`。论文文件位于仓库内的 `paper/`
目录；不要再将论文修改只留在外部工作目录。

版本管理采用“本地完整提交、远程代码发布”的双层规则。Git 不能从同一个提交中只
推送部分文件，因此绝不能把含训练模型的本地完整分支直接推送到远程代码仓库。

每次完成任何项目文件改动后，必须先完成本地完整提交：

1. 检查 `git status` 与 `git diff`，确认包含本次改动及相关生成物。
2. 运行与改动范围相匹配的验证，并在提交说明或工作记录中注明未运行的验证及原因。
3. 使用清晰、具体的中文提交信息执行 `git add --all`、`git commit`。本地提交应
   包含代码、论文、配置、实验脚本、图表、日志、模型产物及审稿回复材料。

远程发布时，只能推送不含 `models/` 训练模型目录的独立代码发布分支或代码发布
副本。发布内容可包括源码、论文、配置、实验脚本、必要图表和文档，但不得上传模型
权重、检查点、归一化产物或其他训练二进制文件。完成代码发布后，须确认远程分支不
包含 `models/`，并使用清晰中文提交描述。

模型的 `.zip`、`.pkl` 和 `.npz` 文件可继续由 Git LFS 为本地完整版本管理，但不得
使用 `git push --no-verify` 规避 LFS 上传：该做法会在远程留下失效的 LFS 指针，既
不是完整模型发布，也不是干净的代码发布。不得提交密码、访问令牌、私钥、环境变量
文件、Python 缓存或 IDE 本地配置。提交前不要覆盖、删除或回退其他人的未提交改动。

## 审稿意见修订规则

1. 改动前先归类审稿意见：澄清、仿真假设/校准、缺少基线或消融、可复现性问题，
   或需要收缩的结论。
2. 对每一项改变的公式或假设，从 `FGCS-main.tex` 追踪到本仓库的实现，再重生成
   受影响的表格和图。不允许为迎合论文文字而手工修改作图数据。
3. 保持可比性：除非论文明确说明理由，所有基线应使用同一个仿真器、吞吐定义、
   总批大小与显存可行性规则。
4. 对评估范围保持准确表述：这是经过校准的统一仿真研究。若审稿人要求部署验证，
   应清晰说明限制，而不是扩大现有结论。
5. 每一项数值修订都要在审稿记录或回复信中保存具体配置、脚本、模型/检查点、
   归一化文件、场景列表和输出路径。

## 需要特别谨慎的事项

- 论文系统模型将 `tau` 称为往返时延（round-trip latency），但代码的
  `NetworkLink.estimate_comm_time()` 会在每次传输上增加一个配置的链路延迟，论文
  其他段落也有将 50 ms 叫作单程延迟的表述。涉及延迟的审稿意见必须先统一术语与
  实现语义，不能未经验证地自行解释。
- 草稿模型与目标模型的解析配置 `max_position_embeddings` 分别为 2048 和 4096，
  但仿真器并不强制限制，且分析 Prompt 可达 8192。若该点受到审稿人关注，应一并
  说明或修订抽象、实验和结论，不能只修改文字。
- `P=0, K=0` 在代码中作为 cloud-only 基线。当前仿真器仍通过 `max(1, K)` 施加
  最小单 Token 传输载荷；未核对和修改该约定前，不得把它称为严格的零通信云端
  执行。
- 经过塑形的奖励函数（shaped reward）对 `P=0` 有很小奖励、对非零分区有很小惩罚。不得宣称 PPO 完全
  没有策略偏好；准确表述应为加入了轻量的结构先验。
