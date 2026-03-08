"""
WaveDL GUI — Interactive Data Plots Card

Displays six research-grade visualisations using Seaborn + matplotlib
embedded in FigureCanvasQTAgg widgets with NavigationToolbar.

**Lazy rendering**: only the currently selected tab's plot is generated.
Previous plots are closed to free memory. This prevents the OOM / memory
leak that occurred when all six figures were created simultaneously.
"""

from __future__ import annotations

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    IndeterminateProgressRing,
    SegmentedWidget,
    SimpleCardWidget,
    isDarkTheme,
    setFont,
)

from ..common.theme_colors import muted_text_color as _muted_text_color
from ..service.data_plots_worker import DataBundle, DataPlotsWorker


# ── Theme helpers ────────────────────────────────────────────────────────────

_PALETTE_LIGHT = [
    "#3b82f6",
    "#8b5cf6",
    "#06b6d4",
    "#f59e0b",
    "#ef4444",
    "#10b981",
    "#f97316",
    "#ec4899",
]
_PALETTE_DARK = [
    "#60a5fa",
    "#a78bfa",
    "#22d3ee",
    "#fbbf24",
    "#f87171",
    "#34d399",
    "#fb923c",
    "#f472b6",
]


def _sns_style(dark: bool) -> dict:
    """Return seaborn rc params matching the app theme."""
    if dark:
        return {
            "axes.facecolor": "#1e1e1e",
            "figure.facecolor": "#1e1e1e",
            "text.color": "#d4d4d4",
            "axes.labelcolor": "#d4d4d4",
            "xtick.color": "#a0a0a0",
            "ytick.color": "#a0a0a0",
            "axes.edgecolor": "#3a3a3a",
            "grid.color": "#2a2a2a",
        }
    return {
        "axes.facecolor": "#fafafa",
        "figure.facecolor": "#ffffff",
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.edgecolor": "#cccccc",
        "grid.color": "#eeeeee",
    }


def _apply_theme(dark: bool):
    """Configure seaborn + matplotlib for the current app theme."""
    style = "darkgrid" if dark else "whitegrid"
    palette = _PALETTE_DARK if dark else _PALETTE_LIGHT
    sns.set_theme(style=style, palette=palette, rc=_sns_style(dark))


# ── Individual plot builders (each returns a Figure) ─────────────────────────


def _plot_input_distribution(b: DataBundle, dark: bool) -> Figure:
    palette = _PALETTE_DARK if dark else _PALETTE_LIGHT
    indices = b.input_feat_indices
    n = len(indices)
    cols = min(n, 4)
    rows = max(1, (n + cols - 1) // cols)

    fig = Figure(figsize=(3.2 * cols, 2.6 * rows), dpi=100)
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for i, feat_idx in enumerate(indices):
        ax = fig.add_subplot(rows, cols, i + 1)
        data = b.x_flat[:, feat_idx]
        sns.histplot(
            data,
            bins=50,
            kde=True,
            color=palette[i % len(palette)],
            alpha=0.7,
            edgecolor="none",
            ax=ax,
            stat="count",
        )
        ax.set_title(f"Feature {feat_idx}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("Input Feature Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _plot_output_distribution(b: DataBundle, dark: bool) -> Figure:
    palette = _PALETTE_DARK if dark else _PALETTE_LIGHT
    n_out = min(b.y_flat.shape[1], 8)
    cols = min(n_out, 4)
    rows = max(1, (n_out + cols - 1) // cols)

    fig = Figure(figsize=(3.2 * cols, 2.6 * rows), dpi=100)
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for i in range(n_out):
        ax = fig.add_subplot(rows, cols, i + 1)
        data = b.y_flat[:, i]
        sns.histplot(
            data,
            bins=50,
            kde=True,
            color=palette[i % len(palette)],
            alpha=0.7,
            edgecolor="none",
            ax=ax,
            stat="count",
        )
        ax.set_title(f"Output {i}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("Output Target Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _plot_sample_preview(b: DataBundle, dark: bool) -> Figure:
    palette = _PALETTE_DARK if dark else _PALETTE_LIGHT
    X = b.x_raw
    n_preview = min(5, X.shape[0])

    if X.ndim == 2:
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        for i in range(n_preview):
            ax.plot(
                X[i],
                color=palette[i % len(palette)],
                alpha=0.85,
                linewidth=1.5,
                label=f"Sample {i}",
            )
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Value")
        ax.set_title("Sample Preview (First 5 Samples)", fontweight="bold")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    elif X.ndim == 3:
        fig = Figure(figsize=(3 * n_preview, 3), dpi=100)
        for i in range(n_preview):
            ax = fig.add_subplot(1, n_preview, i + 1)
            ax.imshow(X[i].T, aspect="auto", cmap="viridis")
            ax.set_title(f"Sample {i}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel("Channel", fontsize=8)
    elif X.ndim >= 4:
        fig = Figure(figsize=(3 * n_preview, 3), dpi=100)
        for i in range(n_preview):
            ax = fig.add_subplot(1, n_preview, i + 1)
            img = X[i]
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.moveaxis(img, 0, -1)
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            ax.imshow(img, cmap="viridis" if img.ndim == 2 else None)
            ax.set_title(f"Sample {i}", fontsize=9, fontweight="bold")
            ax.axis("off")
    else:
        fig = Figure(figsize=(8, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Cannot preview this data shape",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    fig.suptitle("Sample Preview", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _plot_correlation_heatmap(b: DataBundle, dark: bool) -> Figure:
    corr = b.corr_matrix
    if corr is None:
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Only 1 output — correlation\nrequires ≥ 2 outputs",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Output Correlation", fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        return fig

    n = corr.shape[0]
    fig = Figure(figsize=(max(6, n * 0.35), max(5, n * 0.3)), dpi=100)
    ax = fig.add_subplot(111)
    cmap = "coolwarm" if dark else "RdBu_r"
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Output Correlation Heatmap", fontweight="bold", fontsize=12)
    ax.set_xlabel("Output Index", fontsize=9)
    ax.set_ylabel("Output Index", fontsize=9)
    fig.tight_layout()
    return fig


def _plot_input_statistics(b: DataBundle, dark: bool) -> Figure:
    palette = _PALETTE_DARK if dark else _PALETTE_LIGHT
    n = len(b.input_means)

    fig = Figure(figsize=(max(8, n * 0.25), 4), dpi=100)
    ax = fig.add_subplot(111)
    x_pos = np.arange(n)
    ax.bar(
        x_pos,
        b.input_means,
        yerr=b.input_stds,
        color=palette[0],
        alpha=0.7,
        capsize=2,
        edgecolor="none",
        error_kw={"elinewidth": 1, "capthick": 1, "alpha": 0.5},
    )
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Value")
    ax.set_title("Input Feature Statistics (Mean ± Std)", fontweight="bold")

    if n <= 20:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(b.stat_feat_indices, fontsize=8)
    else:
        ax.set_xticks(x_pos[:: max(1, n // 10)])

    fig.tight_layout()
    return fig


def _plot_input_vs_output(b: DataBundle, dark: bool) -> Figure:
    palette = _PALETTE_DARK if dark else _PALETTE_LIGHT
    indices = b.scatter_feat_indices
    n_show = len(indices)
    y_target = b.y_flat[:, 0]

    cols = min(n_show, 3)
    rows = max(1, (n_show + cols - 1) // cols)
    fig = Figure(figsize=(3.5 * cols, 3.2 * rows), dpi=100)
    fig.subplots_adjust(hspace=0.45, wspace=0.4)

    for i, feat_idx in enumerate(indices):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.scatterplot(
            x=b.x_flat[:, feat_idx],
            y=y_target,
            s=6,
            alpha=0.45,
            color=palette[i % len(palette)],
            edgecolor="none",
            ax=ax,
        )
        ax.set_xlabel(f"Input[{feat_idx}]", fontsize=9)
        ax.set_ylabel("Output[0]", fontsize=9)
        ax.set_title(f"Feature {feat_idx} vs Output", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    fig.suptitle("Input vs Output Scatter", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Plot builder dispatch ────────────────────────────────────────────────────

_BUILDERS: dict[str, callable] = {
    "Input Distribution": _plot_input_distribution,
    "Output Distribution": _plot_output_distribution,
    "Sample Preview": _plot_sample_preview,
    "Correlation Heatmap": _plot_correlation_heatmap,
    "Input Statistics": _plot_input_statistics,
    "Input vs Output": _plot_input_vs_output,
}


# ── Compact toolbar subclass ─────────────────────────────────────────────────


class _CompactToolbar(NavigationToolbar2QT):
    """NavigationToolbar with a smaller height and no coordinate display."""

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.setFixedHeight(32)
        for action in self.actions():
            if action.text() == "":
                action.setVisible(False)


# ── Main card ────────────────────────────────────────────────────────────────


class DataPlotsCard(SimpleCardWidget):
    """Tabbed card with interactive data visualisation plots.

    Only the currently selected tab's plot is rendered. Switching tabs
    disposes the previous figure and creates a new one on demand,
    keeping memory usage low.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: DataPlotsWorker | None = None
        self._bundle: DataBundle | None = None
        self._current_plot_name: str | None = None
        self._init_ui()
        self.hide()

    # ── UI ────────────────────────────────────────────────────────

    def _init_ui(self):
        self.setBorderRadius(10)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QVBoxLayout()
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(0)
        root.addLayout(content)

        # Section title
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self._title_label = CaptionLabel("DATA VISUALISATION", self)
        self._title_label.setTextColor(_muted_text_color(), _muted_text_color())
        setFont(self._title_label, 10, QFont.Weight.Bold)
        title_row.addWidget(self._title_label)
        title_row.addStretch()
        content.addLayout(title_row)

        content.addSpacing(12)

        # Tab bar
        self._pivot = SegmentedWidget(self)
        content.addWidget(self._pivot)

        content.addSpacing(8)

        # Container for the active plot (canvas + toolbar)
        self._plot_container = QVBoxLayout()
        self._plot_container.setContentsMargins(0, 0, 0, 0)
        self._plot_container.setSpacing(0)
        content.addLayout(self._plot_container)

        # Spinner page (shown while loading data)
        self._spinner_widget = QWidget(self)
        spinner_layout = QVBoxLayout(self._spinner_widget)
        spinner_layout.setAlignment(Qt.AlignCenter)
        self._spinner = IndeterminateProgressRing(self._spinner_widget)
        self._spinner.setFixedSize(36, 36)
        spinner_layout.addWidget(self._spinner, 0, Qt.AlignCenter)

        self._spinner_caption = CaptionLabel("Loading data…", self._spinner_widget)
        self._spinner_caption.setTextColor(_muted_text_color(), _muted_text_color())
        setFont(self._spinner_caption, 11)
        spinner_layout.addWidget(self._spinner_caption, 0, Qt.AlignCenter)

        self._spinner_widget.setMinimumHeight(360)
        self._plot_container.addWidget(self._spinner_widget)

        # Track the current canvas and toolbar for cleanup
        self._current_canvas: FigureCanvasQTAgg | None = None
        self._current_toolbar: _CompactToolbar | None = None

    # ── Layout helpers ────────────────────────────────────────────

    def _showCard(self):
        self.setMinimumHeight(520)
        self.show()
        QTimer.singleShot(0, self._updateParentLayout)

    def _updateParentLayout(self):
        parent = self.parentWidget()
        if parent is not None:
            parent.adjustSize()
            parent.updateGeometry()

    # ── Public API ────────────────────────────────────────────────

    def set_data_path(self, path: str):
        """Trigger data loading for the given file."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self._dispose_current_plot()
        self._bundle = None
        self._current_plot_name = None
        self._spinner_widget.show()
        self._pivot.hide()
        self._showCard()

        self._worker = DataPlotsWorker(path, parent=self)
        self._worker.dataReady.connect(self._on_data_ready)
        self._worker.errorOccurred.connect(self._on_error)
        self._worker.start()

    def clear(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        self._dispose_current_plot()
        self._bundle = None
        self._current_plot_name = None
        self.setMinimumHeight(0)
        self.setFixedHeight(0)
        self.hide()
        QTimer.singleShot(0, self._updateParentLayout)

    # ── Slots ─────────────────────────────────────────────────────

    def _on_data_ready(self, bundle: DataBundle):
        """Data arrived — set up tabs and render the first plot."""
        self._bundle = bundle
        self._spinner_widget.hide()

        # Build pivot tabs (no plots rendered yet)
        self._pivot.clear()
        for name in DataPlotsWorker.PLOT_NAMES:
            if name not in _BUILDERS:
                continue
            self._pivot.addItem(
                routeKey=name,
                text=name,
                onClick=lambda checked=False, n=name: self._show_plot(n),
            )

        self._pivot.show()

        # Render the first plot
        first = DataPlotsWorker.PLOT_NAMES[0]
        self._pivot.setCurrentItem(first)
        self._show_plot(first)

    def _on_error(self, error: str):
        self._dispose_current_plot()
        self._spinner_widget.hide()

        err_label = CaptionLabel(f"Could not generate plots: {error}", self)
        err_label.setWordWrap(True)
        err_label.setTextColor(_muted_text_color(), _muted_text_color())
        err_label.setMinimumHeight(360)
        err_label.setAlignment(Qt.AlignCenter)
        setFont(err_label, 11)
        self._plot_container.addWidget(err_label)

    # ── Lazy plot rendering ───────────────────────────────────────

    def _show_plot(self, name: str):
        """Render a single plot on demand, disposing the previous one."""
        if self._bundle is None:
            return
        if name == self._current_plot_name:
            return  # Already showing this plot

        # Dispose old figure to free memory
        self._dispose_current_plot()

        # Apply theme and build the figure
        dark = isDarkTheme()
        _apply_theme(dark)
        builder = _BUILDERS.get(name)
        if builder is None:
            return

        fig = builder(self._bundle, dark)

        # Create canvas + toolbar
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(360)
        toolbar = _CompactToolbar(canvas, self)

        self._plot_container.addWidget(toolbar)
        self._plot_container.addWidget(canvas, 1)

        self._current_canvas = canvas
        self._current_toolbar = toolbar
        self._current_plot_name = name

    def _dispose_current_plot(self):
        """Close the current matplotlib figure and remove canvas/toolbar."""
        if self._current_canvas is not None:
            # Close the matplotlib figure to free memory
            fig = self._current_canvas.figure
            if fig is not None:
                import matplotlib.pyplot as plt

                plt.close(fig)

            self._plot_container.removeWidget(self._current_canvas)
            self._current_canvas.deleteLater()
            self._current_canvas = None

        if self._current_toolbar is not None:
            self._plot_container.removeWidget(self._current_toolbar)
            self._current_toolbar.deleteLater()
            self._current_toolbar = None

        self._current_plot_name = None

    # ── Custom painting ───────────────────────────────────────────

    def _normalBackgroundColor(self):
        return QColor(255, 255, 255, 13 if isDarkTheme() else 170)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.borderRadius

        p.setBrush(self._normalBackgroundColor())
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), r, r)

        accent_rect = self.rect().adjusted(1, 1, -1, 0)
        accent_rect.setHeight(4)
        path = QPainterPath()
        path.addRoundedRect(
            accent_rect.x(), accent_rect.y(), accent_rect.width(), r * 2, r, r
        )
        clip_rect = QPainterPath()
        clip_rect.addRect(
            accent_rect.x(),
            accent_rect.y(),
            accent_rect.width(),
            accent_rect.height(),
        )
        path = path.intersected(clip_rect)
        p.setBrush(QColor("#8b5cf6") if not isDarkTheme() else QColor("#a78bfa"))
        p.drawPath(path)

        p.end()
