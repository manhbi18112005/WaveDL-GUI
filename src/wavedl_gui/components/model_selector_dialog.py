"""
WaveDL GUI - Model Selector Dialog & Shared Browser Panel

Provides:
    ModelCard            – compact card for a single model
    ModelDetailPanel     – right-hand detail view
    CategoryFilterBar    – horizontal pill buttons for category filtering
    ModelBrowserPanel    – reusable left+right model browser (used by wizard AND dialog)
    ModelSelectorDialog  – full-screen dialog wrapping the browser panel
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    DropDownPushButton,
    FlowLayout,
    InfoBadge,
    MaskDialogBase,
    PillPushButton,
    RoundMenu,
    ScrollArea,
    SearchLineEdit,
    SimpleCardWidget,
    StrongBodyLabel,
    SubtitleLabel,
    TitleLabel,
    isDarkTheme,
    setFont,
)

from ..common.constants.models import MODEL_CATEGORIES, MODEL_INFO


# =============================================================================
# Model Card Widget (left panel list item)
# =============================================================================
class ModelCard(SimpleCardWidget):
    """A compact card representing a single model in the list."""

    clicked = Signal(str)  # model_key

    def __init__(self, model_key: str, parent=None):
        super().__init__(parent)
        self._key = model_key
        self._selected = False
        self._info = MODEL_INFO.get(model_key, {})

        self.setFixedHeight(72)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(3)

        # Top row: name + pretrained badge
        topRow = QHBoxLayout()
        topRow.setSpacing(8)

        self.nameLabel = StrongBodyLabel(self._info.get("display_name", model_key))
        setFont(self.nameLabel, 13, QFont.Weight.DemiBold)
        topRow.addWidget(self.nameLabel)

        if self._info.get("is_pretrained"):
            badge = InfoBadge.custom(
                "Pretrained",
                QColor(52, 199, 89),
                QColor(48, 209, 88),
                self,
            )
            topRow.addWidget(badge)

        topRow.addStretch()

        # Dims pills
        for dim in self._info.get("supported_dims", []):
            dimLabel = QLabel(f"{dim}D")
            dimLabel.setFixedHeight(18)
            dimLabel.setStyleSheet(
                "QLabel {"
                "  background: rgba(120,120,128,0.12);"
                "  color: rgba(142,142,147,1);"
                "  border-radius: 4px;"
                "  padding: 0px 6px;"
                "  font-size: 10px;"
                "  font-weight: 500;"
                "}"
            )
            topRow.addWidget(dimLabel)

        layout.addLayout(topRow)

        # Short description
        self.descLabel = CaptionLabel(
            self._info.get("short_description", "No description available")
        )
        self.descLabel.setWordWrap(False)
        self.descLabel.setTextColor(QColor(142, 142, 147), QColor(142, 142, 147))
        layout.addWidget(self.descLabel)

    @property
    def model_key(self) -> str:
        return self._key

    def setSelected(self, selected: bool):
        self._selected = selected
        self.update()

    def isSelected(self) -> bool:
        return self._selected

    def mouseReleaseEvent(self, event):
        QWidget.mouseReleaseEvent(self, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._key)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._selected:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Accent border
            accent = QColor(0, 159, 170) if not isDarkTheme() else QColor(0, 186, 199)
            pen = QPen(accent, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            path = QPainterPath()
            path.addRoundedRect(1, 1, self.width() - 2, self.height() - 2, 8, 8)
            painter.drawPath(path)
            painter.end()


# =============================================================================
# Detail Panel (right pane)
# =============================================================================
class ModelDetailPanel(QFrame):
    """Right panel showing detailed info about the selected model."""

    _BULLET_STYLE = (
        "color: rgba(142,142,147,1); font-size: 12px; padding: 0; margin: 0;"
    )
    _TAG_LIGHT = (
        "background: rgba(0,159,170,0.08); color: rgba(0,159,170,1);"
        " border-radius: 4px; padding: 2px 8px; font-size: 10px; font-weight: 500;"
    )
    _TAG_DARK = (
        "background: rgba(0,186,199,0.12); color: rgba(0,186,199,1);"
        " border-radius: 4px; padding: 2px 8px; font-size: 10px; font-weight: 500;"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modelDetailPanel")
        self.setMinimumWidth(340)

        outerLayout = QVBoxLayout(self)
        outerLayout.setContentsMargins(0, 0, 0, 0)
        outerLayout.setSpacing(0)

        # Scrollable content
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        self._scroll.viewport().setStyleSheet("background: transparent;")
        outerLayout.addWidget(self._scroll)

        self._content = QWidget()
        self._content.setStyleSheet("background: transparent;")
        self._scroll.setWidget(self._content)

        layout = QVBoxLayout(self._content)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(0)

        # ── Model name ──
        self.titleLabel = TitleLabel("Select a model")
        setFont(self.titleLabel, 22, QFont.Weight.Bold)
        layout.addWidget(self.titleLabel)
        layout.addSpacing(2)

        # ── Subtitle: family · year · compute tier ──
        self.subtitleInfoLabel = CaptionLabel("")
        self.subtitleInfoLabel.setWordWrap(True)
        self.subtitleInfoLabel.setTextColor(
            QColor(142, 142, 147), QColor(142, 142, 147)
        )
        setFont(self.subtitleInfoLabel, 12)
        self.subtitleInfoLabel.hide()
        layout.addWidget(self.subtitleInfoLabel)
        layout.addSpacing(8)

        # ── Short description ──
        self.shortDescLabel = BodyLabel(
            "Choose an architecture from the list to see its details."
        )
        self.shortDescLabel.setWordWrap(True)
        self.shortDescLabel.setTextColor(QColor(142, 142, 147), QColor(142, 142, 147))
        layout.addWidget(self.shortDescLabel)
        layout.addSpacing(16)

        # ── Detailed description ──
        self.descLabel = BodyLabel("")
        self.descLabel.setWordWrap(True)
        setFont(self.descLabel, 12)
        self.descLabel.hide()
        layout.addWidget(self.descLabel)
        layout.addSpacing(20)

        # ── Specs grid ──
        self.specsWidget = QWidget()
        self.specsLayout = QVBoxLayout(self.specsWidget)
        self.specsLayout.setContentsMargins(0, 0, 0, 0)
        self.specsLayout.setSpacing(0)
        layout.addWidget(self.specsWidget)
        layout.addSpacing(20)

        # ── Best for ──
        self.bestForHeader = StrongBodyLabel("Best for")
        setFont(self.bestForHeader, 13, QFont.Weight.DemiBold)
        self.bestForHeader.hide()
        layout.addWidget(self.bestForHeader)
        layout.addSpacing(6)

        self.bestForLabel = BodyLabel("")
        self.bestForLabel.setWordWrap(True)
        self.bestForLabel.hide()
        layout.addWidget(self.bestForLabel)
        layout.addSpacing(20)

        # ── Pros ──
        self.prosHeader = StrongBodyLabel("Strengths")
        setFont(self.prosHeader, 13, QFont.Weight.DemiBold)
        self.prosHeader.hide()
        layout.addWidget(self.prosHeader)
        layout.addSpacing(6)

        self.prosWidget = QWidget()
        self.prosLayout = QVBoxLayout(self.prosWidget)
        self.prosLayout.setContentsMargins(0, 0, 0, 0)
        self.prosLayout.setSpacing(3)
        self.prosWidget.hide()
        layout.addWidget(self.prosWidget)
        layout.addSpacing(16)

        # ── Cons ──
        self.consHeader = StrongBodyLabel("Limitations")
        setFont(self.consHeader, 13, QFont.Weight.DemiBold)
        self.consHeader.hide()
        layout.addWidget(self.consHeader)
        layout.addSpacing(6)

        self.consWidget = QWidget()
        self.consLayout = QVBoxLayout(self.consWidget)
        self.consLayout.setContentsMargins(0, 0, 0, 0)
        self.consLayout.setSpacing(3)
        self.consWidget.hide()
        layout.addWidget(self.consWidget)
        layout.addSpacing(16)

        # ── Tags ──
        self.tagsWidget = QWidget()
        self.tagsLayout = FlowLayout(self.tagsWidget, needAni=False)
        self.tagsLayout.setContentsMargins(0, 0, 0, 0)
        self.tagsLayout.setHorizontalSpacing(6)
        self.tagsLayout.setVerticalSpacing(6)
        self.tagsWidget.hide()
        layout.addWidget(self.tagsWidget)

        layout.addStretch()

        self._applyStyle()

    def setModel(self, model_key: str):
        """Update detail panel with model information."""
        info = MODEL_INFO.get(model_key, {})
        if not info:
            return

        # ── Title ──
        self.titleLabel.setText(info.get("display_name", model_key))

        # ── Subtitle line: Family · Author · Year ──
        subtitle_parts = []
        family = info.get("architecture_family", "")
        if family:
            subtitle_parts.append(family)
        author = info.get("author_institution", "")
        if author:
            subtitle_parts.append(author)
        year = info.get("year_published")
        if year:
            subtitle_parts.append(str(year))
        if subtitle_parts:
            self.subtitleInfoLabel.setText(" · ".join(subtitle_parts))
            self.subtitleInfoLabel.show()
        else:
            self.subtitleInfoLabel.hide()

        # ── Short description ──
        self.shortDescLabel.setText(info.get("short_description", ""))

        # ── Detailed description ──
        desc = info.get("description", "")
        if desc:
            self.descLabel.setText(desc)
            self.descLabel.show()
        else:
            self.descLabel.hide()

        # ── Specs ──
        self._clearSpecs()
        params = info.get("params_m")
        self._addSpec("Parameters", f"{params}M" if params is not None else "—")
        size = info.get("size_mb")
        self._addSpec("Model Size", f"{size:.0f} MB" if size is not None else "—")
        dims = info.get("supported_dims", [])
        self._addSpec("Dimensions", ", ".join(f"{d}D" for d in dims))
        compute = info.get("compute_tier", "")
        if compute:
            self._addSpec("Compute Tier", compute)
        self._addSpec(
            "Pretrained",
            "ImageNet weights available"
            if info.get("is_pretrained")
            else "Train from scratch",
        )
        self._addSpec("Registry Key", model_key)

        # ── Best for ──
        best_for = info.get("best_for", "")
        if best_for:
            self.bestForHeader.show()
            self.bestForLabel.show()
            self.bestForLabel.setText(best_for)
        else:
            self.bestForHeader.hide()
            self.bestForLabel.hide()

        # ── Pros ──
        self._populateBulletList(
            info.get("pros", []),
            self.prosLayout,
            self.prosWidget,
            self.prosHeader,
            "✓ ",
        )

        # ── Cons ──
        self._populateBulletList(
            info.get("cons", []),
            self.consLayout,
            self.consWidget,
            self.consHeader,
            "✗ ",
        )

        # ── Tags ──
        self._populateTags(info.get("tags", []))

        # Scroll to top
        self._scroll.verticalScrollBar().setValue(0)

    def _addSpec(self, label: str, value: str):
        """Add a spec row to the grid."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 10, 0, 10)

        keyLabel = CaptionLabel(label)
        keyLabel.setFixedWidth(110)
        keyLabel.setTextColor(QColor(142, 142, 147), QColor(142, 142, 147))
        setFont(keyLabel, 12)
        row.addWidget(keyLabel)

        valLabel = BodyLabel(value)
        valLabel.setWordWrap(True)
        setFont(valLabel, 13)
        row.addWidget(valLabel)

        self.specsLayout.addLayout(row)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(
            "background: rgba(120,120,128,0.08);"
            if not isDarkTheme()
            else "background: rgba(120,120,128,0.2);"
        )
        self.specsLayout.addWidget(sep)

    def _populateBulletList(
        self,
        items: list[str],
        layout: QVBoxLayout,
        widget: QWidget,
        header: QWidget,
        prefix: str,
    ):
        """Populate a bullet-point section (pros or cons)."""
        # Clear previous
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not items:
            widget.hide()
            header.hide()
            return

        header.show()
        widget.show()
        for text in items:
            lbl = CaptionLabel(f"{prefix}{text}")
            lbl.setWordWrap(True)
            lbl.setTextColor(QColor(142, 142, 147), QColor(142, 142, 147))
            setFont(lbl, 12)
            layout.addWidget(lbl)

    def _populateTags(self, tags: list[str]):
        """Populate the tag pills."""
        # Clear previous — FlowLayout.takeAt returns the widget directly
        self.tagsLayout.removeAllWidgets()
        for child in self.tagsWidget.findChildren(QLabel):
            child.deleteLater()

        if not tags:
            self.tagsWidget.hide()
            return

        self.tagsWidget.show()
        style = self._TAG_DARK if isDarkTheme() else self._TAG_LIGHT
        for tag in tags:
            pill = QLabel(tag)
            pill.setStyleSheet(f"QLabel {{ {style} }}")
            pill.setFixedHeight(20)
            self.tagsLayout.addWidget(pill)

    def _clearSpecs(self):
        """Remove all spec rows."""
        while self.specsLayout.count():
            item = self.specsLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clearLayout(item.layout())

    @staticmethod
    def _clearLayout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                ModelDetailPanel._clearLayout(item.layout())

    def _applyStyle(self):
        if isDarkTheme():
            self.setStyleSheet(
                "#modelDetailPanel {"
                "  background: rgba(44,44,46,1);"
                "  border-left: 1px solid rgba(120,120,128,0.2);"
                "  border-top-right-radius: 10px;"
                "  border-bottom-right-radius: 10px;"
                "}"
            )
        else:
            self.setStyleSheet(
                "#modelDetailPanel {"
                "  background: rgba(249,249,251,1);"
                "  border-left: 1px solid rgba(120,120,128,0.08);"
                "  border-top-right-radius: 10px;"
                "  border-bottom-right-radius: 10px;"
                "}"
            )


# =============================================================================
# Category Filter Bar
# =============================================================================
class CategoryFilterBar(QWidget):
    """Horizontal scrollable pill buttons for category filtering."""

    categoryChanged = Signal(str)  # "" means "All"

    def __init__(self, categories: list[str], parent=None):
        super().__init__(parent)
        self._current = ""

        mainLayout = QHBoxLayout(self)
        mainLayout.setContentsMargins(0, 0, 0, 0)

        # Horizontal scroll area so pills keep natural width
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.viewport().setStyleSheet("background: transparent;")
        scroll.setFixedHeight(50)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinAndMaxSize)

        # "All" button
        allBtn = PillPushButton(self)
        allBtn.setText("All")
        allBtn.setChecked(True)
        allBtn.setFixedHeight(28)
        allBtn.clicked.connect(lambda *_: self._onCategoryClicked(""))
        layout.addWidget(allBtn)
        self._buttons = {"": allBtn}

        for cat in categories:
            btn = PillPushButton(self)
            btn.setText(cat)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda *_, c=cat: self._onCategoryClicked(c))
            layout.addWidget(btn)
            self._buttons[cat] = btn

        scroll.setWidget(inner)
        mainLayout.addWidget(scroll)

    def _onCategoryClicked(self, category: str):
        if self._current == category:
            return
        # Uncheck previous
        if self._current in self._buttons:
            self._buttons[self._current].setChecked(False)
        self._current = category
        if category in self._buttons:
            self._buttons[category].setChecked(True)
        self.categoryChanged.emit(category)


# =============================================================================
# Model Browser Panel (shared between wizard step and dialog)
# =============================================================================
class ModelBrowserPanel(QWidget):
    """Reusable model browser: search + dim filter + category tabs + card
    list on the left, detail panel on the right.

    Signals
    -------
    modelSelected(str)
        Emitted whenever the user clicks a card.
    modelCountChanged(int, int)
        ``(shown, total)`` — emitted after each rebuild so the parent can
        update its own subtitle / status label.
    """

    modelSelected = Signal(str)
    modelCountChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: dict[str, ModelCard] = {}
        self._selected: str = ""
        self._dim_filter: int | None = None  # active filter value (None = all)
        self._suggested_dim: int | None = None  # suggested from data_info
        self._init_ui()
        self._populate()

    # ── UI construction ──────────────────────────────────────────────────

    def _init_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(0)
        main.setContentsMargins(0, 0, 0, 0)

        # Left panel
        left = QWidget(self)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 12, 0)
        left_lay.setSpacing(8)

        # Search row: dim-filter pill + search bar
        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        search_row.setContentsMargins(0, 0, 0, 0)

        from qfluentwidgets import FluentIcon as FIF

        self._dim_btn = DropDownPushButton(left)
        self._dim_btn.setFixedHeight(36)
        self._dim_btn.setIcon(FIF.FILTER)
        self._dim_btn.setText("All")

        dim_menu = RoundMenu(parent=self._dim_btn)
        self._dim_actions: dict[int | None, Action] = {}
        for key, label in [(None, "All"), (1, "1D"), (2, "2D"), (3, "3D")]:
            action = Action(label)
            action.setCheckable(True)
            action.triggered.connect(lambda *_, k=key: self._on_dim_action(k))
            dim_menu.addAction(action)
            self._dim_actions[key] = action
        self._dim_actions[None].setChecked(True)
        self._dim_btn.setMenu(dim_menu)
        search_row.addWidget(self._dim_btn)

        self._search = SearchLineEdit(left)
        self._search.setPlaceholderText("Search models...")
        self._search.setFixedHeight(36)
        self._search.textChanged.connect(self._on_search)
        search_row.addWidget(self._search, 1)

        left_lay.addLayout(search_row)

        # Category filter
        self._category_bar = CategoryFilterBar(list(MODEL_CATEGORIES.keys()), left)
        self._category_bar.categoryChanged.connect(self._on_category_changed)
        left_lay.addWidget(self._category_bar)

        # Scrollable card list
        self._card_scroll = ScrollArea(left)
        self._card_scroll.setWidgetResizable(True)
        self._card_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._card_scroll.enableTransparentBackground()

        self._card_container = QWidget()
        self._card_container.setStyleSheet("background: transparent;")
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 0, 8, 0)
        self._card_layout.setSpacing(6)
        self._card_scroll.setWidget(self._card_container)

        left_lay.addWidget(self._card_scroll, 1)
        main.addWidget(left, 3)

        # Right panel: detail
        self._detail = ModelDetailPanel(self)
        main.addWidget(self._detail, 2)

    # ── Public API ───────────────────────────────────────────────────────

    def set_dim_filter(self, dim: int | None):
        """Set/clear the dimensionality filter and rebuild the card list.

        When *dim* is not None, the suggested dimension is remembered so the
        dropdown can highlight it.
        """
        self._suggested_dim = dim
        self._dim_filter = dim
        # Update dropdown state
        self._sync_dim_menu()
        self._rebuild_cards()

    def select_model(self, model_key: str):
        """Programmatically select a model card."""
        self._select(model_key)

    def scroll_to(self, model_key: str):
        """Scroll the card list to make the given card visible."""
        if model_key in self._cards:
            self._card_scroll.ensureWidgetVisible(self._cards[model_key], 0, 50)

    @property
    def selected_key(self) -> str:
        return self._selected

    @property
    def detail_panel(self) -> ModelDetailPanel:
        return self._detail

    @property
    def model_count(self) -> int:
        return len(self._cards)

    # ── Card building ────────────────────────────────────────────────────

    def _populate(self):
        """Initial card build with no dim filter."""
        self._rebuild_cards()

    def _rebuild_cards(self):
        """(Re)build model cards, respecting the current dim filter."""
        # Clear existing
        for card in self._cards.values():
            card.deleteLater()
        self._cards.clear()

        while self._card_layout.count():
            item = self._card_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        active_dim = self._dim_filter
        total = 0
        shown = 0

        for cat_name, cat_data in MODEL_CATEGORIES.items():
            compatible: list[str] = []
            for model_id in cat_data["models"]:
                total += 1
                info = MODEL_INFO.get(model_id, {})
                supported = info.get("supported_dims", [])
                if active_dim is None or active_dim in supported:
                    compatible.append(model_id)
                    shown += 1

            if not compatible:
                continue

            # Category header
            header = self._make_category_header(cat_name, cat_data)
            self._card_layout.addWidget(header)

            for model_id in compatible:
                card = ModelCard(model_id, self._card_container)
                card.clicked.connect(self._select)
                self._card_layout.addWidget(card)
                self._cards[model_id] = card

            self._card_layout.addSpacing(8)

        self._card_layout.addStretch()
        self.modelCountChanged.emit(shown, total)

        # Auto-select first or keep previous
        if self._selected and self._selected in self._cards:
            self._select(self._selected)
        elif self._cards:
            self._select(next(iter(self._cards)))

        # Re-apply search
        if self._search.text().strip():
            self._on_search(self._search.text())

    @staticmethod
    def _make_category_header(cat_name: str, cat_data: dict) -> QWidget:
        widget = QWidget()
        widget.setObjectName(f"catHeader_{cat_name}")
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(4, 8, 0, 4)
        lay.setSpacing(2)

        name_lbl = StrongBodyLabel(cat_name, widget)
        setFont(name_lbl, 12, QFont.Weight.DemiBold)
        lay.addWidget(name_lbl)

        desc = cat_data.get("short_description", "")
        if desc:
            desc_lbl = CaptionLabel(desc, widget)
            desc_lbl.setTextColor(QColor(142, 142, 147), QColor(142, 142, 147))
            lay.addWidget(desc_lbl)

        return widget

    # ── Selection ────────────────────────────────────────────────────────

    def _select(self, model_key: str):
        if self._selected in self._cards:
            self._cards[self._selected].setSelected(False)
        self._selected = model_key
        if model_key in self._cards:
            self._cards[model_key].setSelected(True)
        self._detail.setModel(model_key)
        self.modelSelected.emit(model_key)

    # ── Filtering ────────────────────────────────────────────────────────

    def _on_dim_action(self, dim: int | None):
        """Handle dimension filter dropdown selection."""
        self._dim_filter = dim
        self._sync_dim_menu()
        self._rebuild_cards()

    def _sync_dim_menu(self):
        """Update the dropdown button text and checkmarks."""
        for key, action in self._dim_actions.items():
            action.setChecked(key == self._dim_filter)
        if self._dim_filter is None:
            self._dim_btn.setText("All")
        else:
            self._dim_btn.setText(f"{self._dim_filter}D")

    def _on_search(self, text: str):
        query = text.strip().lower()
        for cat_name, cat_data in MODEL_CATEGORIES.items():
            cat_visible = False
            for model_id in cat_data["models"]:
                if model_id not in self._cards:
                    continue
                card = self._cards[model_id]
                info = MODEL_INFO.get(model_id, {})
                name = info.get("display_name", model_id).lower()
                desc = info.get("short_description", "").lower()
                tags_str = " ".join(info.get("tags", [])).lower()
                best = info.get("best_for", "").lower()
                dims = " ".join(f"{d}D" for d in info.get("supported_dims", [])).lower()
                match = not query or any(
                    query in f for f in (model_id, name, desc, best, dims, tags_str)
                )
                card.setVisible(match)
                if match:
                    cat_visible = True

            header = self._card_container.findChild(QWidget, f"catHeader_{cat_name}")
            if header:
                header.setVisible(cat_visible)

    def _on_category_changed(self, category: str):
        for cat_name, cat_data in MODEL_CATEGORIES.items():
            visible = not category or cat_name == category
            for model_id in cat_data["models"]:
                if model_id in self._cards:
                    self._cards[model_id].setVisible(visible)
            header = self._card_container.findChild(QWidget, f"catHeader_{cat_name}")
            if header:
                header.setVisible(visible)

        if self._search.text().strip():
            self._on_search(self._search.text())


# =============================================================================
# Model Selector Dialog
# =============================================================================
class ModelSelectorDialog(MaskDialogBase):
    """Apple-quality model selection dialog.

    Embeds a :class:`ModelBrowserPanel` with dialog chrome:
    header, subtitle, cancel/select buttons.
    """

    modelSelected = Signal(str)  # model_key

    def __init__(
        self,
        current_model: str = "",
        parent=None,
        *,
        dim_filter: int | None = None,
    ):
        super().__init__(parent)
        self._initWidget()
        self._initLayout()
        self._connectSignals()

        # Apply dimension filter if provided
        if dim_filter is not None:
            self._browser.set_dim_filter(dim_filter)

        # Pre-select current model
        if current_model:
            self._browser.select_model(current_model)
            self._browser.scroll_to(current_model)

    def _initWidget(self):
        # Main container
        self.container = QFrame(self.widget)
        self.container.setObjectName("modelSelectorContainer")
        parent_size = self.parent().size() if self.parent() else QSize(1400, 900)
        w = max(960, int(parent_size.width() * 0.7))
        h = max(640, int(parent_size.height() * 0.7))
        self.container.setFixedSize(w, h)

        shadow = QGraphicsDropShadowEffect(self.container)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 60 if not isDarkTheme() else 100))
        shadow.setOffset(0, 8)
        self.container.setGraphicsEffect(shadow)

        if isDarkTheme():
            self.container.setStyleSheet(
                "#modelSelectorContainer {"
                "  background: rgb(39,39,41);"
                "  border-radius: 12px;"
                "  border: 1px solid rgba(120,120,128,0.3);"
                "}"
            )
        else:
            self.container.setStyleSheet(
                "#modelSelectorContainer {"
                "  background: white;"
                "  border-radius: 12px;"
                "  border: 1px solid rgba(120,120,128,0.12);"
                "}"
            )

        # Header
        self.headerLabel = SubtitleLabel("Choose Model")
        setFont(self.headerLabel, 22, QFont.Weight.Bold)

        self.subtitleLabel = CaptionLabel(f"{len(MODEL_INFO)} architectures available")
        self.subtitleLabel.setTextColor(QColor(142, 142, 147), QColor(142, 142, 147))
        setFont(self.subtitleLabel, 13)

        # Browser panel (the shared component)
        self._browser = ModelBrowserPanel()

        # Bottom buttons
        self.cancelBtn = PillPushButton()
        self.cancelBtn.setText("Cancel")

        self.selectBtn = PillPushButton()
        self.selectBtn.setText("Select")

    def _initLayout(self):
        dialogLayout = QHBoxLayout(self.widget)
        dialogLayout.setContentsMargins(0, 0, 0, 0)
        dialogLayout.addWidget(self.container, alignment=Qt.AlignmentFlag.AlignCenter)

        mainLayout = QVBoxLayout(self.container)
        mainLayout.setContentsMargins(24, 24, 0, 20)
        mainLayout.setSpacing(12)

        # Header
        mainLayout.addWidget(self.headerLabel)
        mainLayout.addWidget(self.subtitleLabel)
        mainLayout.addSpacing(4)

        # Browser fills remaining space
        mainLayout.addWidget(self._browser, 1)

        # Buttons bottom-left
        btnRow = QHBoxLayout()
        btnRow.setContentsMargins(0, 0, 24, 0)
        btnRow.addStretch()
        btnRow.addWidget(self.cancelBtn)
        btnRow.addWidget(self.selectBtn)
        mainLayout.addLayout(btnRow)

    def _connectSignals(self):
        self.cancelBtn.clicked.connect(lambda *_: self.reject())
        self.selectBtn.clicked.connect(lambda *_: self._onSelectClicked())
        self._browser.modelCountChanged.connect(self._onCountChanged)

    def _onCountChanged(self, shown: int, total: int):
        if shown < total:
            self.subtitleLabel.setText(f"Showing {shown} of {total} models")
        else:
            self.subtitleLabel.setText(f"{total} architectures available")

    def _onSelectClicked(self):
        key = self._browser.selected_key
        if key:
            self.modelSelected.emit(key)
            self.accept()

    def getSelectedModel(self) -> str:
        return self._browser.selected_key
