"""
Generate PPT1 visualization figures for this repo (GraphRAG MVP).

Outputs:
- ppt_assets/slide01_architecture_overview.png
- ...

Design goals:
- Pure Python (matplotlib + networkx), no external binaries (Graphviz) required.
- 16:9 canvas, PPT-friendly resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import networkx as nx


# -----------------------------
# Theme / helpers
# -----------------------------


@dataclass(frozen=True)
class Theme:
    bg: str = "#FFFFFF"
    text: str = "#0F172A"  # slate-900
    muted: str = "#475569"  # slate-600
    border: str = "#CBD5E1"  # slate-300

    blue: str = "#2563EB"
    blue_50: str = "#DBEAFE"
    green: str = "#16A34A"
    green_50: str = "#DCFCE7"
    amber: str = "#D97706"
    amber_50: str = "#FEF3C7"
    purple: str = "#7C3AED"
    purple_50: str = "#EDE9FE"
    red: str = "#DC2626"
    red_50: str = "#FEE2E2"


THEME = Theme()


def _set_fonts() -> None:
    # Best-effort Chinese font fallback on Windows/macOS/Linux.
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def new_canvas(*, w: float = 13.333, h: float = 7.5, dpi: int = 150):
    fig = plt.figure(figsize=(w, h), dpi=dpi, facecolor=THEME.bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(
    ax,
    *,
    xy: Tuple[float, float],
    wh: Tuple[float, float],
    title: str,
    lines: Iterable[str] = (),
    fc: str = "#FFFFFF",
    ec: str = THEME.border,
    lw: float = 1.5,
    title_color: str = THEME.text,
    text_color: str = THEME.muted,
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)

    ax.text(
        x + 0.02 * w,
        y + h - 0.22 * h,
        title,
        ha="left",
        va="center",
        fontsize=18,
        color=title_color,
        fontweight="bold",
    )

    if lines:
        text = "\n".join([str(s) for s in lines])
        ax.text(
            x + 0.02 * w,
            y + h - 0.42 * h,
            text,
            ha="left",
            va="top",
            fontsize=13,
            color=text_color,
            linespacing=1.3,
        )

    return patch


def arrow(ax, *, start: Tuple[float, float], end: Tuple[float, float], color: str = THEME.border, lw: float = 2.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arr)
    return arr


def label(ax, *, xy: Tuple[float, float], text: str, size: int = 12, color: str = THEME.muted, ha: str = "left"):
    ax.text(xy[0], xy[1], text, ha=ha, va="center", fontsize=size, color=color)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# -----------------------------
# Slides
# -----------------------------


def slide01_architecture(out_dir: Path) -> Path:
    fig, ax = new_canvas()

    box(
        ax,
        xy=(0.04, 0.62),
        wh=(0.22, 0.30),
        title="Data Sources",
        lines=["CSV（课程表）", "Markdown（手册）", "PDF（pypdf文本抽取）"],
        fc=THEME.blue_50,
        ec=THEME.blue,
    )
    box(
        ax,
        xy=(0.30, 0.62),
        wh=(0.22, 0.30),
        title="Ingestion (Build)",
        lines=["load → split → embed", "Chroma向量库", "NetworkX先修图"],
        fc=THEME.green_50,
        ec=THEME.green,
    )
    box(
        ax,
        xy=(0.56, 0.62),
        wh=(0.22, 0.30),
        title="Index Artifacts",
        lines=["chunks.jsonl", "chroma/ (persist)", "graph.json.gz", "manifest/entities"],
        fc=THEME.purple_50,
        ec=THEME.purple,
    )
    box(
        ax,
        xy=(0.80, 0.62),
        wh=(0.16, 0.30),
        title="FastAPI",
        lines=["/health", "/qa"],
        fc="#F8FAFC",
        ec=THEME.border,
    )

    box(
        ax,
        xy=(0.62, 0.18),
        wh=(0.18, 0.28),
        title="LangGraph Agent",
        lines=["route→retrieve→fuse", "→generate", "hybrid: vector + graph", "citations"],
        fc=THEME.amber_50,
        ec=THEME.amber,
    )
    box(
        ax,
        xy=(0.82, 0.18),
        wh=(0.14, 0.28),
        title="Streamlit UI",
        lines=["调用后端", "展示答案/引用/路径"],
        fc="#F1F5F9",
        ec=THEME.border,
    )

    arrow(ax, start=(0.26, 0.77), end=(0.30, 0.77), color=THEME.border)
    arrow(ax, start=(0.52, 0.77), end=(0.56, 0.77), color=THEME.border)
    arrow(ax, start=(0.78, 0.77), end=(0.80, 0.77), color=THEME.border)

    # FastAPI loads artifacts and calls agent
    arrow(ax, start=(0.86, 0.62), end=(0.72, 0.46), color=THEME.border)
    arrow(ax, start=(0.72, 0.46), end=(0.72, 0.62), color=THEME.border)

    # UI calls FastAPI
    arrow(ax, start=(0.89, 0.32), end=(0.89, 0.62), color=THEME.border)
    arrow(ax, start=(0.91, 0.62), end=(0.91, 0.32), color=THEME.border)
    label(ax, xy=(0.92, 0.49), text="HTTP", size=11)

    # Artifacts used by agent
    arrow(ax, start=(0.67, 0.62), end=(0.67, 0.46), color=THEME.border)
    label(ax, xy=(0.685, 0.53), text="load", size=11)

    label(ax, xy=(0.04, 0.05), text="GraphRAG MVP (offline-first)", size=12, color=THEME.muted)

    path = out_dir / "slide01_architecture_overview.png"
    save(fig, path)
    return path


def slide02_inputs_outputs(out_dir: Path) -> Path:
    fig, ax = new_canvas()

    box(
        ax,
        xy=(0.06, 0.18),
        wh=(0.38, 0.68),
        title="输入（Sources）",
        lines=[
            "CSV: 课程行（course_code / prerequisite / year / program…）",
            "Markdown: 手册文本",
            "PDF: 文本抽取（无OCR）",
            "",
            "目录：data/sources/",
        ],
        fc=THEME.blue_50,
        ec=THEME.blue,
    )

    box(
        ax,
        xy=(0.56, 0.18),
        wh=(0.38, 0.68),
        title="输出（Index Artifacts）",
        lines=[
            "chunks.jsonl  (chunk文本 + metadata)",
            "chroma/       (向量库持久化)",
            "graph.json.gz (先修图 + 证据chunk_id)",
            "entities.json (实体/别名索引)",
            "community_reports.json (社区摘要)",
            "manifest.json (配置与可复现信息)",
            "",
            "目录：data/index/",
        ],
        fc=THEME.purple_50,
        ec=THEME.purple,
    )

    arrow(ax, start=(0.44, 0.52), end=(0.56, 0.52), color=THEME.border)
    label(ax, xy=(0.47, 0.56), text="build_index", size=12)

    path = out_dir / "slide02_inputs_outputs.png"
    save(fig, path)
    return path


def slide03_build_pipeline(out_dir: Path) -> Path:
    fig, ax = new_canvas()

    steps = [
        ("scan", ["scan_source_files"], THEME.blue_50, THEME.blue),
        ("load", ["load_sources"], THEME.green_50, THEME.green),
        ("split", ["split_documents", "chunk_size/overlap"], THEME.amber_50, THEME.amber),
        ("embed", ["HuggingFace", "fallback: hash"], THEME.purple_50, THEME.purple),
        ("persist", ["Chroma + JSON/GZ"], "#F1F5F9", THEME.border),
    ]

    x0 = 0.05
    y = 0.55
    w = 0.18
    h = 0.22
    gap = 0.03

    for i, (t, lines, fc, ec) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, xy=(x, y), wh=(w, h), title=t, lines=lines, fc=fc, ec=ec)
        if i < len(steps) - 1:
            arrow(ax, start=(x + w, y + h / 2), end=(x + w + gap, y + h / 2), color=THEME.border)

    # Outputs breakdown
    box(
        ax,
        xy=(0.08, 0.16),
        wh=(0.84, 0.26),
        title="落盘产物（便于 Build/Serve 解耦）",
        lines=[
            "- chunks.jsonl：检索证据与引用基础",
            "- chroma/：向量检索",
            "- graph.json.gz：先修关系图（含证据chunk_id）",
            "- entities/community/manifest：实体链接、全局证据、可复现",
        ],
        fc="#FFFFFF",
        ec=THEME.border,
    )

    path = out_dir / "slide03_build_pipeline.png"
    save(fig, path)
    return path


def slide04_graph_and_community(out_dir: Path) -> Path:
    fig, ax = new_canvas()

    # Draw a small prerequisite graph with two "communities"
    g = nx.DiGraph()
    edges = [
        ("AI101", "ML201"),
        ("ML201", "DL301"),
        ("ML201", "NLP301"),
        ("MATH101", "ML201"),
        ("STAT101", "ML201"),
        ("DS101", "DS201"),
        ("DS201", "ML201"),
        ("DS201", "VIS301"),
    ]
    g.add_edges_from(edges)

    pos = nx.spring_layout(g, seed=7)

    # Community coloring (manual for stability)
    comm_a = {"AI101", "ML201", "DL301", "NLP301", "MATH101", "STAT101", "DS201"}
    comm_b = {"DS101", "VIS301"}

    colors = []
    for n in g.nodes():
        if n in comm_b:
            colors.append(THEME.purple)
        else:
            colors.append(THEME.blue)

    # Left: graph canvas region
    ax_graph = fig.add_axes([0.06, 0.16, 0.58, 0.74])
    ax_graph.axis("off")
    ax_graph.set_facecolor("#FFFFFF")
    nx.draw_networkx_edges(g, pos, ax=ax_graph, edge_color=THEME.muted, arrows=True, arrowsize=18, width=2.0)
    nx.draw_networkx_nodes(g, pos, ax=ax_graph, node_color=colors, node_size=1200, edgecolors="#FFFFFF", linewidths=2)
    nx.draw_networkx_labels(g, pos, ax=ax_graph, font_size=12, font_color="#FFFFFF", font_weight="bold")

    # Right: explanation panel
    box(
        ax,
        xy=(0.68, 0.54),
        wh=(0.28, 0.36),
        title="先修图（NetworkX）",
        lines=[
            "节点：course_code",
            "边：PREREQUISITE_OF",
            "边属性：confidence + evidence_chunk_ids",
            "支持：k-hop 邻域 / 最短路径",
        ],
        fc=THEME.green_50,
        ec=THEME.green,
    )
    box(
        ax,
        xy=(0.68, 0.16),
        wh=(0.28, 0.32),
        title="社区报告（Community Reports）",
        lines=[
            "对图做社区划分（离线）",
            "生成“全局摘要”伪chunk",
            "查询时作为 global evidence",
        ],
        fc=THEME.amber_50,
        ec=THEME.amber,
    )

    label(ax, xy=(0.06, 0.08), text="示意图：颜色代表不同社区（真实项目会从 graph.json.gz 构建）", size=11)

    path = out_dir / "slide04_graph_and_community.png"
    save(fig, path)
    return path


def slide05_langgraph_flow(out_dir: Path) -> Path:
    fig, ax = new_canvas()

    nodes = [
        ("route", ["实体链接", "hybrid / vector_only"]),
        ("retrieve", ["Chroma top_k", "Graph k-hop/paths"]),
        ("fuse", ["去重+重排", "evidence_pack + citations"]),
        ("generate", ["offline / OpenAI / Ollama", "引用约束"]),
    ]

    x0 = 0.08
    y0 = 0.58
    w = 0.18
    h = 0.26
    gap = 0.05
    fcs = [THEME.blue_50, THEME.green_50, THEME.amber_50, THEME.purple_50]
    ecs = [THEME.blue, THEME.green, THEME.amber, THEME.purple]

    for i, (name, lines) in enumerate(nodes):
        x = x0 + i * (w + gap)
        box(ax, xy=(x, y0), wh=(w, h), title=name, lines=lines, fc=fcs[i], ec=ecs[i])
        if i < len(nodes) - 1:
            arrow(ax, start=(x + w, y0 + h / 2), end=(x + w + gap, y0 + h / 2), color=THEME.border)

    box(
        ax,
        xy=(0.12, 0.18),
        wh=(0.76, 0.24),
        title="State（GraphRAGState）",
        lines=[
            "query, top_k, k_hop, route",
            "vector_hits, graph_paths, graph_hits",
            "evidence_pack, answer, citations, debug(latency_ms…)",
        ],
        fc="#F8FAFC",
        ec=THEME.border,
    )

    path = out_dir / "slide05_langgraph_flow.png"
    save(fig, path)
    return path


def slide06_retrieve_hybrid(out_dir: Path) -> Path:
    fig, ax = new_canvas()

    box(
        ax,
        xy=(0.06, 0.62),
        wh=(0.26, 0.28),
        title="Query",
        lines=["用户问题", "抽课程码 / 别名匹配"],
        fc="#F8FAFC",
        ec=THEME.border,
    )

    box(
        ax,
        xy=(0.38, 0.64),
        wh=(0.24, 0.26),
        title="Vector Retrieval",
        lines=["Chroma similarity_search", "top_k 文本chunk"],
        fc=THEME.blue_50,
        ec=THEME.blue,
    )
    box(
        ax,
        xy=(0.38, 0.30),
        wh=(0.24, 0.26),
        title="Graph Retrieval",
        lines=["k-hop 邻域", "paths + edge/node证据", "community reports"],
        fc=THEME.green_50,
        ec=THEME.green,
    )

    box(
        ax,
        xy=(0.70, 0.46),
        wh=(0.24, 0.28),
        title="Candidates",
        lines=["vector_hits + graph_hits", "graph_paths"],
        fc=THEME.amber_50,
        ec=THEME.amber,
    )

    arrow(ax, start=(0.32, 0.76), end=(0.38, 0.76), color=THEME.border)
    arrow(ax, start=(0.32, 0.76), end=(0.38, 0.42), color=THEME.border)
    arrow(ax, start=(0.62, 0.77), end=(0.70, 0.60), color=THEME.border)
    arrow(ax, start=(0.62, 0.43), end=(0.70, 0.57), color=THEME.border)

    label(ax, xy=(0.06, 0.08), text="hybrid：并行取证（向量相似 + 图关系扩展）", size=12)

    path = out_dir / "slide06_retrieve_hybrid.png"
    save(fig, path)
    return path


def slide07_fusion_citations(out_dir: Path) -> Path:
    fig, ax = new_canvas()

    box(
        ax,
        xy=(0.06, 0.58),
        wh=(0.36, 0.34),
        title="Fusion / Re-rank",
        lines=[
            "去重：按 chunk_id 合并",
            "补齐：graph-only 证据也加入",
            "",
            "简单打分：",
            "final_score = α·vector_score + β·graph_bonus",
        ],
        fc=THEME.amber_50,
        ec=THEME.amber,
    )

    box(
        ax,
        xy=(0.52, 0.58),
        wh=(0.42, 0.34),
        title="evidence_pack (Top8)",
        lines=[
            "[pdf:...::p12::c3] …",
            "[csv:courses_mock.csv::row7] …",
            "[community:c0] …",
            "…",
        ],
        fc="#F8FAFC",
        ec=THEME.border,
    )

    box(
        ax,
        xy=(0.06, 0.18),
        wh=(0.88, 0.28),
        title="Explainability（可解释性）",
        lines=[
            "返回 citations：source / page_or_row / section / chunk_id",
            "返回 graph_paths：先修链路步骤 + 每条边的 evidence_chunk_ids",
            "debug：latency_ms / evidence_counts（便于演示与诊断）",
        ],
        fc=THEME.green_50,
        ec=THEME.green,
    )

    path = out_dir / "slide07_fusion_citations.png"
    save(fig, path)
    return path


def slide08_generation_modes(out_dir: Path) -> Path:
    fig, ax = new_canvas()

    box(
        ax,
        xy=(0.06, 0.62),
        wh=(0.26, 0.28),
        title="Offline (default)",
        lines=["确定性摘要", "无外部API", "可复现"],
        fc=THEME.blue_50,
        ec=THEME.blue,
    )
    box(
        ax,
        xy=(0.37, 0.62),
        wh=(0.26, 0.28),
        title="OpenAI (optional)",
        lines=["ChatOpenAI", "更流畅回答", "失败自动回退"],
        fc=THEME.green_50,
        ec=THEME.green,
    )
    box(
        ax,
        xy=(0.68, 0.62),
        wh=(0.26, 0.28),
        title="Ollama (optional)",
        lines=["本地模型", "需运行ollama服务", "失败自动回退"],
        fc=THEME.purple_50,
        ec=THEME.purple,
    )

    box(
        ax,
        xy=(0.10, 0.18),
        wh=(0.84, 0.34),
        title="统一约束：Citations 必须来自 evidence_pack",
        lines=[
            "1) 清理模型输出里的非法引用",
            "2) 追加规范化的 “Citations: <chunk_id,...>” 行",
            "3) 确保答案可追溯、可解释",
        ],
        fc=THEME.amber_50,
        ec=THEME.amber,
    )

    path = out_dir / "slide08_generation_modes.png"
    save(fig, path)
    return path


def slide09_demo_sequence(out_dir: Path) -> Path:
    fig, ax = new_canvas()

    lanes = [
        ("User", 0.08),
        ("Streamlit", 0.30),
        ("FastAPI", 0.52),
        ("Agent", 0.74),
        ("Chroma/Graph\nArtifacts", 0.92),
    ]

    # lifelines
    for name, x in lanes:
        ax.plot([x, x], [0.16, 0.90], color=THEME.border, linewidth=2)
        ax.text(x, 0.93, name, ha="center", va="center", fontsize=13, color=THEME.text, fontweight="bold")

    def msg(y: float, src: float, dst: float, text: str):
        arrow(ax, start=(src, y), end=(dst, y), color=THEME.muted, lw=2.0)
        ax.text((src + dst) / 2, y + 0.02, text, ha="center", va="bottom", fontsize=11, color=THEME.muted)

    msg(0.84, 0.08, 0.30, "输入Query + 参数(top_k,k_hop)")
    msg(0.76, 0.30, 0.52, "POST /qa")
    msg(0.68, 0.52, 0.74, "LangGraph invoke")
    msg(0.60, 0.74, 0.92, "load artifacts + retrieve")
    msg(0.52, 0.92, 0.74, "vector_hits + graph_hits + paths")
    msg(0.44, 0.74, 0.52, "answer + citations + debug")
    msg(0.36, 0.52, 0.30, "JSON response")
    msg(0.28, 0.30, 0.08, "展示：Answer/Citations/Paths/Debug")

    path = out_dir / "slide09_demo_sequence.png"
    save(fig, path)
    return path


def slide10_evaluation_summary(out_dir: Path) -> Path:
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150, facecolor=THEME.bg)

    # Left: bar chart placeholder
    ax1 = fig.add_axes([0.06, 0.18, 0.52, 0.70])
    ax1.set_facecolor("#FFFFFF")
    metrics = ["Hit@k", "Validity", "Latency(ms)↓"]
    vector_only = [0.45, 0.62, 180]
    hybrid = [0.62, 0.78, 210]
    x = range(len(metrics))
    ax1.bar([i - 0.18 for i in x], vector_only, width=0.36, label="vector_only", color=THEME.blue)
    ax1.bar([i + 0.18 for i in x], hybrid, width=0.36, label="hybrid", color=THEME.green)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(metrics, fontsize=12)
    ax1.tick_params(axis="y", labelsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", frameon=False)
    ax1.set_title("Evaluation (示例占位，可替换为 notebook 结果)", fontsize=14, color=THEME.text)

    # Right: pros/cons/next
    ax2 = fig.add_axes([0.62, 0.18, 0.34, 0.70])
    ax2.axis("off")
    ax2.set_facecolor("#FFFFFF")

    # Three stacked boxes
    box(ax2, xy=(0.00, 0.68), wh=(1.00, 0.30), title="优点", lines=["离线优先可运行", "图+向量互补", "可解释引用（chunk_id）"], fc=THEME.green_50, ec=THEME.green)
    box(
        ax2,
        xy=(0.00, 0.34),
        wh=(1.00, 0.30),
        title="不足",
        lines=["PDF无OCR", "融合/排序较简单", "图谱关系目前偏“先修课”"],
        fc=THEME.red_50,
        ec=THEME.red,
    )
    box(ax2, xy=(0.00, 0.00), wh=(1.00, 0.30), title="下一步", lines=["OCR/更强实体链接", "Reranker/学习排序", "更丰富图谱与监控"], fc=THEME.blue_50, ec=THEME.blue)

    path = out_dir / "slide10_evaluation_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def generate_all(out_dir: Path) -> list[Path]:
    _set_fonts()
    out: list[Path] = []
    out.append(slide01_architecture(out_dir))
    out.append(slide02_inputs_outputs(out_dir))
    out.append(slide03_build_pipeline(out_dir))
    out.append(slide04_graph_and_community(out_dir))
    out.append(slide05_langgraph_flow(out_dir))
    out.append(slide06_retrieve_hybrid(out_dir))
    out.append(slide07_fusion_citations(out_dir))
    out.append(slide08_generation_modes(out_dir))
    out.append(slide09_demo_sequence(out_dir))
    out.append(slide10_evaluation_summary(out_dir))
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "ppt_assets"
    paths = generate_all(out_dir)
    print("Generated PPT figures:")
    for p in paths:
        print("-", p.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

