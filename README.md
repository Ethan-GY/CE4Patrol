# CE4Patrol: 具身智能安防巡逻的上下文增强评估框架

## 1. 背景

随着具身智能从实验室加速走向产业落地，四足机器人（机器狗）凭借其卓越的地形适应性与自主移动能力，已成为安防巡逻领域最具规模化潜力的应用载体。与传统定点巡检不同，动态巡逻的核心诉求是“在持续移动中，对任意潜在威胁进行实时、准确、可解释的异常判断”。这一任务对智能体的环境理解深度、上下文推理能力与决策鲁棒性提出了前所未有的挑战。

## 2. 核心问题：“薛定谔的异常”

尽管大型视觉语言模型（VLM）在通用场景下表现出色，但在严肃的工业安防场景中，它们存在一个根本性瓶颈：VLMs 擅长识别“视觉异常”，却对“逻辑异常”束手无策。我们将其定义为 **“薛定谔的异常（Schrödinger’s Anomaly）”**——即视觉表征完全正常，但在特定上下文逻辑下应被判定为异常的状态。

这类异常在安防巡逻中普遍存在，且危害巨大，具体表现为三类核心问题：

1.  **预训练偏见冲突 (Domain-Knowledge Conflict)**：VLMs 从互联网学习的“常识”与工业安全规范相悖。
    *   **例子**：模型将“绿色设备灯”普遍理解为正常，但设备手册规定“仅红色为正常”，导致致命误判。

2.  **时空悖论 (Inadequate Dynamic Context Modeling)**：同一视觉状态在不同时间/地点应有不同判定。
    *   **例子**：“人员出现在机房”在白天是正常维护，半夜则是入侵事件。静态图像模型无法建模此类动态上下文。

3.  **语义模糊下的决策瘫痪 (Lack of Fine-Grained Semantic Understanding)**：面对远处模糊或不确定的目标，VLMs 难以做出有效决策。
    *   **例子**：面对疑似烟头的小黑点，VLMs 要么沉默（漏报），要么武断分类（误报），缺乏在不确定性下推荐风险规避措施的能力。

## 3. CE4Patrol 框架

为解决上述挑战，我们提出了 **CE4Patrol (Context-Enhanced Evaluation for Patrol)** 框架。该框架通过向 VLM 注入多层次的上下文信息，并结合思维链（Chain-of-Thought）推理，显著提升模型在复杂安防场景下的异常检测与决策能力。

其核心思想是将上下文分解为三个层次，并在推理时动态注入：

-   **S (Spatiotemporal Context) 时空上下文**：提供时间、地点、历史状态等信息，解决“时空悖论”问题。
-   **R (Rules & Domain Knowledge) 规则与领域知识**：注入特定场景的安全规范、设备手册等，解决“预训练偏见冲突”问题。
-   **D (Decision & Action) 决策与行动指南**：提供标准操作流程（SOP）和风险应对策略，解决“决策瘫痪”问题。

通过在 `main.py` 中配置不同的 `ExperimentConfig`，可以对不同层次上下文注入的效果进行消融实验，从而量化评估各部分对模型性能的贡献。

## 4. 项目结构
├── .env                  # 环境变量配置 (需手动创建)
├── README.md             # 项目说明
├── ce4patrol/            # 核心代码模块
│   ├── analysis/         # 实验结果分析与可视化
│   ├── context_loader.py # 数据集与上下文加载器
│   ├── data_models.py    # 数据模型定义
│   ├── evaluation/       # 评估指标与CRS计算器
│   ├── prompt_generator.py # 上下文增强的Prompt生成器
│   └── vlm_caller.py     # VLM模型调用接口
├── data/                 # 数据文件
│   ├── dataset.json      # 实验用的数据集
│   └── images/           # 场景图片
├── main.py               # 主实验运行脚本
├── mock_runner.py        # 使用模拟VLM响应的快速演示脚本
└── requirements.txt      # Python依赖

## 5. 快速开始

### 5.1. 安装依赖

```bash
pip install -r requirements.txt
```

### 5.2. 配置环境变量

在项目根目录下创建 `.env` 文件，并填入您所使用的 VLM 的 API 信息：
CE4_API_KEY="your_api_key"
CE4_API_BASE="your_api_base_url"
CE4_MODEL="your_model_name"

### 5.3. 运行完整实验

执行 `main.py` 脚本，将运行所有预定义的消融实验，并对 `data/dataset.json` 中的每个案例进行评估。

```bash
python main.py
```

实验结果将保存在 `results/` 目录下，包括：
-   `experiment_outputs.json`: 所有实验的原始输出和评估指标。
-   `analysis_report.md`: 对比分析报告。
-   各种可视化图表 (`.png` 格式)，如雷达图、热力图等。

### 5.4. 运行模拟演示

如果您想快速了解项目流程而无需配置和调用 VLM API，可以运行 `mock_runner.py`。该脚本使用预定义的模拟 VLM 响应来展示完整的实验、评估和分析流程。

```bash
python mock_runner.py
```

## 6. 实验与评估

本项目通过一系列消融实验来验证 CE4Patrol 框架的有效性。实验配置定义在 `main.py` 的 `ABLATIONS` 列表中，包括：

-   **CE4_FULL**: 包含 S、R、D 三层上下文并启用思维链（CoT）。
-   **NoCoT**: 包含三层上下文但禁用思维链。
-   **SR_only**, **S_only**, **R_only**, **D_only**: 只包含部分上下文层。
-   **None**: 不包含任何额外上下文（作为基线）。

我们引入了 **CRS (Comprehensive Reliability Score)** 综合可靠性评分，从准确性、逻辑一致性、行动可靠性和置信度四个维度全面评估模型在安防任务中的表现。
