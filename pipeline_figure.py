from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
PDF_OUTPUT = OUTPUT_DIR / "pipeline_figure.pdf"
PNG_OUTPUT = OUTPUT_DIR / "pipeline_figure.png"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    fill: str = "0.96",
    fontsize: float = 10,
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="0.15",
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="0.1",
        linespacing=1.3,
    )


def add_down_arrow(ax, x: float, y_top: float, y_bottom: float) -> None:
    ax.annotate(
        "",
        xy=(x, y_bottom),
        xytext=(x, y_top),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.2", shrinkA=6, shrinkB=6),
    )


def add_cross(ax, x: float, y: float, size: float = 0.04, color: str = "#b22222") -> None:
    ax.add_line(Line2D([x - size, x + size], [y - size, y + size], lw=2.0, color=color))
    ax.add_line(Line2D([x - size, x + size], [y + size, y - size], lw=2.0, color=color))


def draw_panel(
    ax,
    *,
    title: str,
    separator_text: str,
    final_text: str,
    leakage_caption: str,
    dynamic: bool,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel = patches.FancyBboxPatch(
        (0.02, 0.03),
        0.96,
        0.94,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="0.35",
        facecolor="white",
    )
    ax.add_patch(panel)

    ax.text(0.5, 0.93, title, ha="center", va="center", fontsize=11, fontweight="bold", color="0.05")

    box_width = 0.52
    box_height = 0.12
    box_x = 0.24
    positions = [0.76, 0.56, 0.36, 0.16]

    add_box(ax, box_x, positions[0], box_width, box_height, "[System Prompt]", fill="0.94", fontsize=10)
    add_box(ax, box_x, positions[1], box_width, box_height, separator_text, fill="0.98", fontsize=8.5)
    add_box(ax, box_x, positions[2], box_width, box_height, "[User Input]", fill="0.94", fontsize=10)
    add_box(ax, box_x, positions[3], box_width, box_height, final_text, fill="0.98", fontsize=9.2)

    center_x = box_x + box_width / 2
    add_down_arrow(ax, center_x, positions[0], positions[1] + box_height)
    add_down_arrow(ax, center_x, positions[1], positions[2] + box_height)
    add_down_arrow(ax, center_x, positions[2], positions[3] + box_height)

    leak_y = positions[1] + box_height / 2
    final_y = positions[3] + box_height / 2
    annotation_x = 0.8

    if dynamic:
        add_cross(ax, center_x, 0.11)
        ax.text(
            0.72,
            0.07,
            leakage_caption,
            ha="center",
            va="bottom",
            color="0.1",
            linespacing=1.2,
        )
    else:
        ax.annotate(
            "",
            xy=(0.83, final_y - 0.01),
            xytext=(center_x + 0.13, leak_y - 0.01),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5,
                color="#b22222",
                linestyle=(0, (4, 3)),
                connectionstyle="arc3,rad=-0.35",
            ),
        )
        ax.text(
            0.74,
            0.07,
            leakage_caption,
            ha="center",
            va="bottom",
            color="0.1",
            linespacing=1.2,
        )


def build_figure() -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), constrained_layout=True)

    draw_panel(
        axes[0],
        title="Static PPA",
        separator_text="SELECT from fixed separator pool\n(e.g. ====BEGIN-A1B2C3D4====)",
        final_text="SAME separator reused next request",
        leakage_caption="Leakage blast radius:\nall future requests",
        dynamic=False,
    )
    draw_panel(
        axes[1],
        title="Dynamic PPA (ours)",
        separator_text=(
            "GENERATE\n"
            "SHA-256(timestamp || session_id || nonce)[:24]\n"
            "(e.g. ====BEGIN-f3a8c2e1d9b7====)"
        ),
        final_text="New unique separator each request",
        leakage_caption="Leakage blast radius:\nsingle request only",
        dynamic=True,
    )

    fig.suptitle(
        "Static vs. Dynamic PPA: Assembly Pipeline and Leakage Blast Radius",
        fontsize=12,
        y=1.02,
    )
    return fig


def save_figure() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    figure.savefig(PDF_OUTPUT)
    figure.savefig(PNG_OUTPUT, dpi=300)
    plt.close(figure)
    return PDF_OUTPUT, PNG_OUTPUT


def main() -> None:
    pdf_path, png_path = save_figure()
    print(f"Saved publication figure to {pdf_path}")
    print(f"Saved publication figure to {png_path}")


if __name__ == "__main__":
    main()