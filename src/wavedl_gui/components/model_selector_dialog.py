"""
WaveDL GUI - Model Selector Dialog

A polished, Apple-quality model selection dialog with:
- Left panel: searchable, filterable card list grouped by category
- Right panel: detailed information about the selected model
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
    BodyLabel,
    CaptionLabel,
    FlowLayout,
    InfoBadge,
    MaskDialogBase,
    PillPushButton,
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
# Model Selector Dialog
# =============================================================================
class ModelSelectorDialog(MaskDialogBase):
    """Apple-quality model selection dialog.

    Left panel: searchable card list with category filter pills.
    Right panel: detailed model information.
    """

    modelSelected = Signal(str)  # model_key

    def __init__(self, current_model: str = "", parent=None):
        super().__init__(parent)
        self._selectedKey = current_model
        self._cards: dict[str, ModelCard] = {}
        self._categoryCards: dict[str, list[str]] = {}

        self._initWidget()
        self._initLayout()
        self._connectSignals()
        self._populateModels()

        # Pre-select current model
        if current_model and current_model in self._cards:
            self._selectCard(current_model)
            self._scrollToCard(current_model)

    def _initWidget(self):
        # Main container
        self.container = QFrame(self.widget)
        self.container.setObjectName("modelSelectorContainer")
        # Size relative to parent window (70%)
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

        # --- Left Panel ---
        self.leftPanel = QWidget()

        # Header
        self.headerLabel = SubtitleLabel("Choose Model")
        setFont(self.headerLabel, 22, QFont.Weight.Bold)

        self.subtitleLabel = CaptionLabel(f"{len(MODEL_INFO)} architectures available")
        self.subtitleLabel.setTextColor(QColor(142, 142, 147), QColor(142, 142, 147))
        setFont(self.subtitleLabel, 13)

        # Search
        self.searchBox = SearchLineEdit()
        self.searchBox.setPlaceholderText("Search models...")
        self.searchBox.setFixedHeight(36)

        # Category filter
        self.categoryBar = CategoryFilterBar(list(MODEL_CATEGORIES.keys()))

        # Scrollable card list
        self.cardScroll = ScrollArea()
        self.cardScroll.setWidgetResizable(True)
        self.cardScroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.cardScroll.enableTransparentBackground()

        self.cardContainer = QWidget()
        self.cardContainer.setStyleSheet("background: transparent;")
        self.cardLayout = QVBoxLayout(self.cardContainer)
        self.cardLayout.setContentsMargins(0, 0, 8, 0)
        self.cardLayout.setSpacing(6)
        self.cardScroll.setWidget(self.cardContainer)

        # Bottom buttons
        self.cancelBtn = PillPushButton()
        self.cancelBtn.setText("Cancel")

        self.selectBtn = PillPushButton()
        self.selectBtn.setText("Select")

        # --- Right Panel ---
        self.detailPanel = ModelDetailPanel()

    def _initLayout(self):
        # Dialog centers the container
        dialogLayout = QHBoxLayout(self.widget)
        dialogLayout.setContentsMargins(0, 0, 0, 0)
        dialogLayout.addWidget(self.container, alignment=Qt.AlignmentFlag.AlignCenter)

        # Main horizontal split
        mainLayout = QHBoxLayout(self.container)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.setSpacing(0)

        # Left panel layout
        leftLayout = QVBoxLayout(self.leftPanel)
        leftLayout.setContentsMargins(24, 24, 16, 20)
        leftLayout.setSpacing(12)

        leftLayout.addWidget(self.headerLabel)
        leftLayout.addWidget(self.subtitleLabel)
        leftLayout.addSpacing(4)
        leftLayout.addWidget(self.searchBox)
        leftLayout.addWidget(self.categoryBar)
        leftLayout.addWidget(self.cardScroll, 1)

        # Button row
        btnRow = QHBoxLayout()
        btnRow.addStretch()
        btnRow.addWidget(self.cancelBtn)
        btnRow.addWidget(self.selectBtn)
        leftLayout.addLayout(btnRow)

        mainLayout.addWidget(self.leftPanel, 3)
        mainLayout.addWidget(self.detailPanel, 2)

    def _connectSignals(self):
        self.searchBox.textChanged.connect(self._onSearchChanged)
        self.categoryBar.categoryChanged.connect(self._onCategoryChanged)
        self.cancelBtn.clicked.connect(lambda *_: self.reject())
        self.selectBtn.clicked.connect(lambda *_: self._onSelectClicked())

    def _populateModels(self):
        """Build model cards grouped by category."""
        for category, cat_info in MODEL_CATEGORIES.items():
            # Category header
            header = self._createCategoryHeader(category)
            self.cardLayout.addWidget(header)
            self._categoryCards.setdefault(category, [])

            for key in cat_info["models"]:
                card = ModelCard(key)
                card.clicked.connect(self._selectCard)
                self.cardLayout.addWidget(card)
                self._cards[key] = card
                self._categoryCards[category].append(key)

            self.cardLayout.addSpacing(8)

        self.cardLayout.addStretch()

    def _createCategoryHeader(self, category: str) -> QWidget:
        """Create a category section header with description."""
        widget = QWidget()
        widget.setObjectName(f"catHeader_{category}")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 8, 0, 4)
        layout.setSpacing(2)

        nameLabel = StrongBodyLabel(category)
        setFont(nameLabel, 12, QFont.Weight.DemiBold)
        layout.addWidget(nameLabel)

        desc = MODEL_CATEGORIES.get(category, {}).get("description", "")
        if desc:
            descLabel = CaptionLabel(desc)
            descLabel.setTextColor(QColor(142, 142, 147), QColor(142, 142, 147))
            layout.addWidget(descLabel)

        return widget

    def _selectCard(self, model_key: str):
        """Handle card selection."""
        # Deselect previous
        if self._selectedKey in self._cards:
            self._cards[self._selectedKey].setSelected(False)

        self._selectedKey = model_key

        # Select new
        if model_key in self._cards:
            self._cards[model_key].setSelected(True)

        # Update detail panel
        self.detailPanel.setModel(model_key)

    def _scrollToCard(self, model_key: str):
        """Scroll the card list to make the given card visible."""
        if model_key in self._cards:
            card = self._cards[model_key]
            self.cardScroll.ensureWidgetVisible(card, 0, 50)

    def _onSearchChanged(self, text: str):
        """Filter cards by search text."""
        query = text.strip().lower()
        visible_count = 0

        for category, keys in self._categoryCards.items():
            category_visible = False
            for key in keys:
                card = self._cards[key]
                info = MODEL_INFO.get(key, {})
                name = info.get("display_name", key).lower()
                desc = info.get("short_description", "").lower()
                tags_str = " ".join(info.get("tags", [])).lower()
                best = info.get("best_for", "").lower()
                dims = " ".join(f"{d}D" for d in info.get("supported_dims", [])).lower()

                match = not query or any(
                    query in field for field in (key, name, desc, best, dims, tags_str)
                )
                card.setVisible(match)
                if match:
                    category_visible = True
                    visible_count += 1

            # Toggle category header
            header = self.container.findChild(QWidget, f"catHeader_{category}")
            if header:
                header.setVisible(category_visible)

        self.subtitleLabel.setText(
            f"{visible_count} model{'s' if visible_count != 1 else ''} found"
            if query
            else f"{len(MODEL_INFO)} architectures available"
        )

    def _onCategoryChanged(self, category: str):
        """Filter cards by category."""
        for cat, keys in self._categoryCards.items():
            visible = not category or cat == category
            for key in keys:
                card = self._cards[key]
                card.setVisible(visible)

            header = self.container.findChild(QWidget, f"catHeader_{cat}")
            if header:
                header.setVisible(visible)

        # Also apply search filter on top
        search_text = self.searchBox.text().strip()
        if search_text:
            self._onSearchChanged(search_text)

    def _onSelectClicked(self):
        """Confirm selection and close."""
        if self._selectedKey:
            self.modelSelected.emit(self._selectedKey)
            self.accept()

    def getSelectedModel(self) -> str:
        """Return the selected model key."""
        return self._selectedKey
