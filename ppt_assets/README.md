# PPT1 可视化图（自动生成）

本目录用于存放 **PPT1（10页）** 每页的主图（16:9 PNG），适合直接拖进 PPT 作为“架构/流程/算法/演示”可视化。

## 一键生成

在项目根目录运行：

```powershell
python scripts/generate_ppt_figures.py
```

会在本目录生成 `slide01_*.png` ～ `slide10_*.png`。

## 每页对应的图片

1. 系统架构总览 → `slide01_architecture_overview.png`
2. 数据源 & 构建产物 → `slide02_inputs_outputs.png`
3. Ingestion 流程（Build） → `slide03_build_pipeline.png`
4. 图索引 + 社区报告 → `slide04_graph_and_community.png`
5. LangGraph 链路图 → `slide05_langgraph_flow.png`
6. Hybrid 检索（vector + graph） → `slide06_retrieve_hybrid.png`
7. 融合与可解释性（引用/路径） → `slide07_fusion_citations.png`
8. 生成模式（offline/OpenAI/Ollama） → `slide08_generation_modes.png`
9. 功能操作演示（序列图） → `slide09_demo_sequence.png`
10. 评估 + 优缺点 + 下一步（占位示例） → `slide10_evaluation_summary.png`

## 说明

- `slide10_evaluation_summary.png` 里的柱状图是 **示例占位**，你可以把 notebook 的真实指标替换进脚本里再重生成。
- 如果你最终把 PPT 压缩到 8 页，建议合并：
  - `slide03` + `slide04`（Build & Graph）
  - `slide06` + `slide07`（Retrieval & Fusion）

