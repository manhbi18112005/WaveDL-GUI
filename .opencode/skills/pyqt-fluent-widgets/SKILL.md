---
name: pyqt-fluent-widgets
description: Full reference for the PyQt-Fluent-Widgets (qfluentwidgets) library — Microsoft Fluent Design widget toolkit for PyQt5/PyQt6/PySide2/PySide6. Covers theming, configuration, settings cards, navigation, icons, all widget classes, layouts, dialogs, acrylic material effects, window classes, multimedia, and Qt Designer integration. Use this skill when writing or modifying GUI code that uses qfluentwidgets.
license: MIT
compatibility: opencode
metadata:
  audience: developers
  framework: qt
  source: https://pyqt-fluent-widgets.readthedocs.io/en/latest/
---

# PyQt-Fluent-Widgets Full Reference

A fluent design widgets library based on PyQt5/PyQt6/PySide2/PySide6. Implements
Microsoft Fluent Design components for Qt applications. The package name is always
`qfluentwidgets` regardless of the Qt binding used.

Official docs: https://pyqt-fluent-widgets.readthedocs.io/en/latest/
Repository: https://github.com/zhiyiYo/PyQt-Fluent-Widgets
Website: https://qfluentwidgets.com

---

## 1. Installation & Setup

### Install (PyQt5 — default)

```bash
# Lite version (no AcrylicLabel)
pip install PyQt-Fluent-Widgets -i https://pypi.org/simple/

# Full version (AcrylicLabel available)
pip install "PyQt-Fluent-Widgets[full]" -i https://pypi.org/simple/
```

### Other Qt bindings

For PySide2, PySide6, or PyQt6, install the corresponding package from the
matching branch:

| Qt Binding | Package                       | Branch   |
|------------|-------------------------------|----------|
| PyQt5      | `PyQt-Fluent-Widgets`         | master   |
| PyQt6      | `PyQt6-Fluent-Widgets`        | PyQt6    |
| PySide2    | `PySide2-Fluent-Widgets`      | PySide2  |
| PySide6    | `PySide6-Fluent-Widgets`      | PySide6  |

**WARNING**: Never install more than one variant at the same time. They all
share the package name `qfluentwidgets` and will conflict.

### Running gallery example

```bash
cd examples/gallery
python demo.py
```

If you see `ImportError: cannot import name 'XXX' from 'qfluentwidgets'`, your
installed version is too old. Reinstall from `https://pypi.org/simple/`.

---

## 2. Theme System

### Theme Mode

Switch light/dark themes using `setTheme()`. Accepts three values:

```python
from qfluentwidgets import setTheme, Theme

setTheme(Theme.LIGHT)   # Light theme
setTheme(Theme.DARK)    # Dark theme
setTheme(Theme.AUTO)    # Follow system theme (falls back to light)
```

When the theme changes, the config instance managed by `qconfig` emits the
`themeChanged` signal.

### Theme Color

Change the accent/theme color using `setThemeColor()`:

```python
from qfluentwidgets import setThemeColor
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

setThemeColor(QColor(0, 101, 213))    # QColor
setThemeColor(Qt.blue)                 # Qt.GlobalColor
setThemeColor("#0065d5")               # Hex string
setThemeColor("red")                   # Color name string
```

When the theme color changes, `qconfig` emits the `themeColorChanged` signal.

### Automatic StyleSheet Switching

Inherit `StyleSheetBase` to automatically switch QSS files when theme changes.
Create separate QSS files for light and dark themes:

```
app/resource/qss/light/main_window.qss
app/resource/qss/dark/main_window.qss
```

```python
from enum import Enum
from qfluentwidgets import StyleSheetBase, Theme, isDarkTheme, qconfig


class StyleSheet(StyleSheetBase, Enum):
    """Style sheet"""

    MAIN_WINDOW = "main_window"
    DASHBOARD = "dashboard"

    def path(self, theme=Theme.AUTO):
        theme = qconfig.theme if theme == Theme.AUTO else theme
        return f"app/resource/qss/{theme.value.lower()}/{self.value}.qss"


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # Apply style sheet — auto-updates on theme change
        StyleSheet.MAIN_WINDOW.apply(self)
```

### Utility Functions

| Function        | Description                                      |
|-----------------|--------------------------------------------------|
| `isDarkTheme()` | Returns `True` if current theme is dark          |
| `getIconColor(theme)` | Returns icon color string for the given theme |

---

## 3. Configuration System (QConfig)

PyQt-Fluent-Widgets uses `ConfigItem` to represent configuration items and
`QConfig` to read/write values. Values are auto-persisted to a JSON file.

### ConfigItem Attributes

| Attribute    | Type              | Description                                          |
|-------------|-------------------|------------------------------------------------------|
| `group`     | `str`             | Group name for the config item                       |
| `name`      | `str`             | Name of the config item                              |
| `default`   | `Any`             | Default value (used when stored value is invalid)    |
| `validator` | `ConfigValidator` | Validates and corrects config values                 |
| `serializer`| `ConfigSerializer`| Serializes/deserializes non-JSON-native types        |
| `restart`   | `bool`            | Whether app restart is needed after value change     |

### ConfigItem Subclasses

| Class                | Value Type     | Validator              | Notes                              |
|---------------------|----------------|------------------------|------------------------------------|
| `ConfigItem`        | Any            | `ConfigValidator`      | Base class                         |
| `OptionsConfigItem` | Enum or str    | `OptionsValidator`     | Fixed set of choices               |
| `RangeConfigItem`   | int/float      | `RangeValidator`       | Min/max bounded numeric            |
| `ColorConfigItem`   | `QColor`/str   | `ColorValidator`       | Auto-uses `ColorSerializer`        |

### Validators

| Validator          | Description                           | Constructor                      |
|-------------------|---------------------------------------|----------------------------------|
| `ConfigValidator` | Base — always passes                  | `ConfigValidator()`              |
| `BoolValidator`   | Boolean values only                   | `BoolValidator()`                |
| `OptionsValidator`| Must be one of given options          | `OptionsValidator([opt1, opt2])` |
| `RangeValidator`  | Numeric range                         | `RangeValidator(min, max)`       |
| `ColorValidator`  | Valid QColor                          | `ColorValidator()`               |
| `FolderValidator` | Valid folder path                     | `FolderValidator()`              |
| `FolderListValidator` | List of valid folder paths        | `FolderListValidator()`          |

### Serializers

| Serializer        | Description                                    |
|-------------------|------------------------------------------------|
| `ConfigSerializer`| Base — identity transform                      |
| `EnumSerializer`  | Serializes Python Enum to/from JSON            |
| `ColorSerializer` | Serializes QColor to/from hex string           |

### Full Config Example

```python
from enum import Enum
from qfluentwidgets import (
    qconfig, QConfig, ConfigItem, OptionsConfigItem,
    RangeConfigItem, ColorConfigItem,
    BoolValidator, OptionsValidator, RangeValidator,
    EnumSerializer,
)


class MvQuality(Enum):
    FULL_HD = "Full HD"
    HD = "HD"
    SD = "SD"
    LD = "LD"

    @staticmethod
    def values():
        return [q.value for q in MvQuality]


class Config(QConfig):
    """Application config"""

    # MainWindow group
    enableAcrylic = ConfigItem(
        "MainWindow", "EnableAcrylic", False, BoolValidator()
    )
    playBarColor = ColorConfigItem("MainWindow", "PlayBarColor", "#225C7F")
    themeMode = OptionsConfigItem(
        "MainWindow", "ThemeMode", "Light",
        OptionsValidator(["Light", "Dark", "Auto"]),
        restart=True,
    )
    recentPlaysNumber = RangeConfigItem(
        "MainWindow", "RecentPlayNumbers", 300, RangeValidator(10, 300)
    )

    # Online group
    onlineMvQuality = OptionsConfigItem(
        "Online", "MvQuality", MvQuality.FULL_HD,
        OptionsValidator(MvQuality), EnumSerializer(MvQuality),
    )


# Create singleton and load from file
cfg = Config()
qconfig.load("config/config.json", cfg)
```

### Accessing Config Values

```python
# Read
value = cfg.get(cfg.enableAcrylic)

# Write (auto-saves to JSON)
cfg.set(cfg.enableAcrylic, True)
```

---

## 4. Setting Cards

Setting cards display configuration items as interactive UI elements. When the
user interacts with a card, the bound `ConfigItem` value updates and auto-saves.

### Available Setting Card Types

| Class                      | Widget Inside      | Description                                    |
|---------------------------|--------------------|------------------------------------------------|
| `SettingCard`             | (base class)       | Base card — subclass or use directly           |
| `HyperlinkCard`          | Hyperlink          | Card with a clickable URL                      |
| `ColorSettingCard`       | Color picker       | Pick a color                                   |
| `CustomColorSettingCard` | Button             | Button that opens a color chooser              |
| `ComboBoxSettingCard`    | ComboBox           | Dropdown selection                             |
| `RangeSettingCard`       | Slider             | Numeric range slider                           |
| `PushSettingCard`        | PushButton         | Card with a plain push button                  |
| `PrimaryPushSettingCard` | PrimaryPushButton  | Card with a primary-colored push button        |
| `SwitchSettingCard`      | SwitchButton       | Toggle switch                                  |
| `OptionsSettingCard`     | Radio buttons      | Group of radio buttons                         |
| `FolderListSettingCard`  | Folder list        | Display and manage a list of folder paths      |

### SettingCardGroup

Group related setting cards together. The group adjusts its layout
automatically based on the height of its child cards.

```python
from qfluentwidgets import SettingCardGroup, SwitchSettingCard, FluentIcon as FIF

group = SettingCardGroup("Appearance", self)
group.addSettingCard(
    SwitchSettingCard(
        FIF.BRUSH,
        "Enable acrylic effect",
        "Use acrylic blur effect on the navigation pane",
        cfg.enableAcrylic,
    )
)
```

### Constructing a Setting Card

Most setting cards follow this constructor pattern:

```python
SwitchSettingCard(
    icon,           # FluentIcon or FluentIconBase enum member
    title,          # str — card title
    content,        # str — description text
    configItem,     # ConfigItem — the config item to bind
    parent=None,    # QWidget
)
```

For `ComboBoxSettingCard`, pass the list of display texts:

```python
ComboBoxSettingCard(
    icon,
    title,
    content,
    texts=["Option A", "Option B"],  # display labels
    configItem=cfg.myOption,
    parent=self,
)
```

---

## 5. Navigation

### Architecture

`NavigationInterface` provides a side navigation panel. Use it with
`QStackWidget` inside a `QHBoxLayout`:

```
+---------------------------+--------------------+
|   NavigationInterface     |   QStackWidget     |
|   (NavigationPanel)       |   (sub-interfaces) |
+---------------------------+--------------------+
```

The `NavigationPanel` holds navigation menu items. All items must inherit from
`NavigationWidget`. The library provides `NavigationPushButton` as a default
implementation.

### Adding Navigation Items

**Simple approach** — `addItem()` adds a `NavigationPushButton`:

```python
self.navigationInterface.addItem(
    routeKey="home",
    icon=FIF.HOME,
    text="Home",
    onClick=lambda: self.stackWidget.setCurrentWidget(self.homeInterface),
    position=NavigationItemPosition.TOP,
)
```

**Custom widget approach** — `addWidget()`:

```python
def addWidget(
    self,
    routeKey: str,
    widget: NavigationWidget,
    onClick=None,
    position=NavigationItemPosition.TOP,
    tooltip: str = None,
    parentRouteKey: str = None,
)
```

### Parameters

| Parameter        | Type                       | Description                                                   |
|-----------------|----------------------------|---------------------------------------------------------------|
| `routeKey`      | `str`                      | Unique identifier for this navigation item                    |
| `widget`        | `NavigationWidget`         | The widget to add to the panel                                |
| `onClick`       | callable                   | Slot for the `clicked` signal                                 |
| `position`      | `NavigationItemPosition`   | Where to place the item                                       |
| `tooltip`       | `str`                      | Tooltip text                                                  |
| `parentRouteKey`| `str`                      | Route key of parent (for tree navigation)                     |

### NavigationItemPosition

| Value    | Description                                                |
|----------|------------------------------------------------------------|
| `TOP`    | Top section of the panel                                   |
| `SCROLL` | Scrollable middle section (for many items)                |
| `BOTTOM` | Bottom section of the panel                               |

### Display Modes

| Mode                            | Trigger                    | Description                           |
|--------------------------------|----------------------------|---------------------------------------|
| `NavigationDisplayMode.EXPAND` | Window width >= 1008px     | Full expanded side panel              |
| `NavigationDisplayMode.COMPACT`| Window width < 1008px      | Icon-only collapsed panel             |
| `NavigationDisplayMode.MENU`   | User clicks expand button  | Temporarily expanded overlay panel    |
| `NavigationDisplayMode.MINIMAL`| Manual setup required      | Only a hamburger menu button shown    |

Use `NavigationInterface.setExpandWidth()` to change the 1008px breakpoint.
Call `NavigationInterface.setDefaultRouteKey()` before running the app.

### Custom Navigation Widget Example

Subclass `NavigationWidget` and override `paintEvent()` and optionally
`setCompacted()`:

```python
from qfluentwidgets import NavigationWidget, isDarkTheme


class AvatarWidget(NavigationWidget):
    """Avatar navigation widget"""

    def __init__(self, parent=None):
        super().__init__(isSelectable=False, parent=parent)
        self.avatar = QImage("resource/avatar.png").scaled(
            24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.SmoothPixmapTransform | QPainter.Antialiasing
        )
        painter.setPen(Qt.NoPen)

        if self.isPressed:
            painter.setOpacity(0.7)

        # Draw hover background
        if self.isEnter:
            c = 255 if isDarkTheme() else 0
            painter.setBrush(QColor(c, c, c, 10))
            painter.drawRoundedRect(self.rect(), 5, 5)

        # Draw circular avatar
        painter.setBrush(QBrush(self.avatar))
        painter.translate(8, 6)
        painter.drawEllipse(0, 0, 24, 24)
        painter.translate(-8, -6)

        # Draw text when not compacted
        if not self.isCompacted:
            painter.setPen(Qt.white if isDarkTheme() else Qt.black)
            font = QFont("Segoe UI")
            font.setPixelSize(14)
            painter.setFont(font)
            painter.drawText(QRect(44, 0, 255, 36), Qt.AlignVCenter, "Username")
```

### NavigationTreeWidgetBase

For tree-structured (hierarchical) navigation menus, parent items must be
instances of `NavigationTreeWidgetBase`. Child items reference the parent via
`parentRouteKey`.

### MSFluentWindow and FluentWindow

These are convenience window classes that bundle `NavigationInterface` +
`QStackWidget` together. See Section 11 (Window Classes) for details.

When using `MSFluentWindow` or `FluentWindow`, add sub-interfaces using:

```python
self.addSubInterface(self.homeInterface, FIF.HOME, "Home")
self.addSubInterface(
    self.settingInterface, FIF.SETTING, "Settings",
    position=NavigationItemPosition.BOTTOM,
)
```

---

## 6. Icons

### Built-in FluentIcon

The `FluentIcon` enum (commonly aliased as `FIF`) provides hundreds of built-in
Microsoft Fluent icons that auto-switch between light/dark variants:

```python
from qfluentwidgets import FluentIcon as FIF

# Use in buttons, cards, navigation, etc.
button = PushButton(FIF.ADD, "Add Item")
```

### Custom Icon Enum

Create theme-aware custom icons by inheriting `FluentIconBase`:

```python
from enum import Enum
from qfluentwidgets import FluentIconBase, getIconColor, Theme


class MyFluentIcon(FluentIconBase, Enum):
    """Custom icons"""

    ADD = "Add"
    CUT = "Cut"
    COPY = "Copy"

    def path(self, theme=Theme.AUTO):
        # Return path to SVG file for the given theme
        return f":/icons/{self.value}_{getIconColor(theme)}.svg"
```

The `getIconColor(theme)` function returns `"white"` for dark theme and
`"black"` for light theme, matching the standard naming convention for icon
SVG files.

### Icon File Naming Convention

```
icons/
  Add_black.svg      # Light theme
  Add_white.svg      # Dark theme
  Cut_black.svg
  Cut_white.svg
```

### IconWidget

Display an icon as a standalone widget:

```python
from qfluentwidgets import IconWidget, FluentIcon as FIF

icon_widget = IconWidget(FIF.GITHUB, parent=self)
icon_widget.setFixedSize(24, 24)
```

---

## 7. Widgets Catalog

All widgets are importable from `qfluentwidgets`. They are drop-in replacements
for standard Qt widgets with Fluent Design styling and theme awareness.

### Basic Input Widgets

| Class               | Qt Equivalent           | Description                              |
|--------------------|-------------------------|------------------------------------------|
| `PushButton`       | `QPushButton`           | Standard push button                     |
| `PrimaryPushButton`| `QPushButton`           | Accent-colored push button               |
| `DropDownPushButton`| `QPushButton`          | Button with dropdown menu                |
| `SplitPushButton`  | `QPushButton`           | Split button (action + dropdown)         |
| `ToolButton`       | `QToolButton`           | Icon-only tool button                    |
| `PrimaryToolButton`| `QToolButton`           | Accent-colored tool button               |
| `DropDownToolButton`| `QToolButton`          | Tool button with dropdown                |
| `SplitToolButton`  | `QToolButton`           | Split tool button                        |
| `ToggleButton`     | `QPushButton`           | Toggleable push button                   |
| `ToggleToolButton` | `QToolButton`           | Toggleable tool button                   |
| `HyperlinkButton`  | `QPushButton`           | Button styled as a hyperlink             |
| `RadioButton`      | `QRadioButton`          | Fluent-styled radio button               |
| `CheckBox`         | `QCheckBox`             | Fluent-styled check box                  |
| `ComboBox`         | `QComboBox`             | Fluent-styled combo box (dropdown)       |
| `EditableComboBox` | `QComboBox`             | Combo box with editable text field       |
| `SwitchButton`     | N/A                     | iOS-style toggle switch                  |
| `Slider`           | `QSlider`               | Fluent-styled slider                     |
| `HollowHandleSlider`| `QSlider`              | Slider with hollow circular handle       |
| `SpinBox`          | `QSpinBox`              | Fluent-styled integer spin box           |
| `DoubleSpinBox`    | `QDoubleSpinBox`        | Fluent-styled float spin box             |
| `TimeEdit`         | `QTimeEdit`             | Fluent-styled time editor                |
| `DateEdit`         | `QDateEdit`             | Fluent-styled date editor                |
| `DateTimeEdit`     | `QDateTimeEdit`         | Fluent-styled date/time editor           |

### Text & Labels

| Class              | Qt Equivalent | Description                                    |
|-------------------|---------------|------------------------------------------------|
| `LineEdit`        | `QLineEdit`   | Single-line text input with clear button       |
| `SearchLineEdit`  | `QLineEdit`   | Line edit with search icon                     |
| `PasswordLineEdit`| `QLineEdit`   | Line edit with password visibility toggle      |
| `TextEdit`        | `QTextEdit`   | Multi-line text editor                         |
| `PlainTextEdit`   | `QPlainTextEdit` | Plain-text multi-line editor                |
| `TitleLabel`      | `QLabel`      | Large title text                               |
| `SubtitleLabel`   | `QLabel`      | Subtitle text                                  |
| `StrongBodyLabel` | `QLabel`      | Bold body text                                 |
| `BodyLabel`       | `QLabel`      | Standard body text                             |
| `CaptionLabel`    | `QLabel`      | Small caption text                             |
| `LargeTitleLabel` | `QLabel`      | Extra-large title text                         |
| `DisplayLabel`    | `QLabel`      | Display-size text                              |
| `ImageLabel`      | `QLabel`      | Image display label                            |
| `AvatarWidget`    | `QLabel`      | Circular avatar display                        |
| `AcrylicLabel`    | `QLabel`      | Label with acrylic blur (full install only)    |

### Data Display

| Class              | Qt Equivalent   | Description                                 |
|-------------------|-----------------|---------------------------------------------|
| `ListView`        | `QListView`     | Fluent-styled list view                     |
| `ListWidget`      | `QListWidget`   | Fluent-styled list widget                   |
| `TableView`       | `QTableView`    | Fluent-styled table view                    |
| `TableWidget`     | `QTableWidget`  | Fluent-styled table widget                  |
| `TreeView`        | `QTreeView`     | Fluent-styled tree view                     |
| `TreeWidget`      | `QTreeWidget`   | Fluent-styled tree widget                   |
| `FlipView`        | N/A             | Horizontal/vertical image carousel          |
| `PipsPager`       | N/A             | Dot-style page indicator                    |
| `TabView`         | `QTabWidget`    | Fluent-styled tab bar                       |

### Cards

| Class              | Description                                       |
|-------------------|---------------------------------------------------|
| `CardWidget`      | Base card container with rounded corners & shadow  |
| `SimpleCardWidget`| Simplified card without extra decoration           |
| `ElevatedCardWidget` | Card with elevation shadow                     |
| `HeaderCardWidget`| Card with a header section                         |

### Containers & Scroll

| Class              | Qt Equivalent    | Description                               |
|-------------------|------------------|-------------------------------------------|
| `ScrollArea`      | `QScrollArea`    | Smooth-scrolling scroll area              |
| `SingleDirectionScrollArea` | `QScrollArea` | Scroll area for one direction only |
| `SmoothScrollArea`| `QScrollArea`    | Enhanced smooth-scrolling scroll area     |
| `SmoothScrollBar` | `QScrollBar`     | Animated smooth scroll bar                |
| `StackedWidget`   | `QStackedWidget` | Stacked widget with slide animation       |
| `PopUpAniStackedWidget` | `QStackedWidget` | Stacked widget with popup animation  |
| `Separator`       | N/A              | Horizontal or vertical separator line     |

### Command Bar

| Class        | Description                                            |
|-------------|--------------------------------------------------------|
| `CommandBar`| Ribbon-style command bar with primary/secondary actions |
| `CommandButton` | Individual button inside a CommandBar              |

### Progress Indicators

| Class              | Qt Equivalent    | Description                              |
|-------------------|------------------|------------------------------------------|
| `ProgressBar`     | `QProgressBar`   | Horizontal progress bar                  |
| `IndeterminateProgressBar` | `QProgressBar` | Indeterminate (animated) progress bar |
| `ProgressRing`    | N/A              | Circular progress indicator              |
| `IndeterminateProgressRing` | N/A    | Spinning circular progress indicator     |

### Status & Notifications

| Class            | Description                                             |
|-----------------|---------------------------------------------------------|
| `InfoBar`       | Notification banner (success/warning/error/info/custom) |
| `InfoBadge`     | Small badge overlay (counts/dots/icons)                 |
| `StateToolTip`  | Progress notification tooltip                           |
| `ToolTip`       | Standard hover tooltip                                  |
| `ToolTipFilter` | Event filter that adds tooltips to any widget           |
| `TeachingTip`   | Instructional popup attached to a widget                |
| `FlyoutViewBase`| Base for flyout content                                 |
| `Flyout`        | Lightweight popup anchored to a widget                  |

### InfoBar Usage Pattern

```python
from qfluentwidgets import InfoBar, InfoBarPosition

# Success notification
InfoBar.success(
    title="Success",
    content="Operation completed",
    parent=self,
    position=InfoBarPosition.TOP_RIGHT,
    duration=3000,  # ms, -1 for persistent
)

# Warning
InfoBar.warning(title="Warning", content="...", parent=self)

# Error
InfoBar.error(title="Error", content="...", parent=self)

# Info
InfoBar.info(title="Info", content="...", parent=self)
```

### InfoBarPosition Values

| Position                 | Description              |
|-------------------------|--------------------------|
| `TOP`                   | Top center               |
| `BOTTOM`                | Bottom center            |
| `TOP_LEFT`              | Top left                 |
| `TOP_RIGHT`             | Top right                |
| `BOTTOM_LEFT`           | Bottom left              |
| `BOTTOM_RIGHT`          | Bottom right             |
| `NONE`                  | Manual positioning       |

### Menus

| Class              | Qt Equivalent | Description                              |
|-------------------|---------------|------------------------------------------|
| `RoundMenu`       | `QMenu`       | Rounded-corner context menu              |
| `CheckableMenu`   | `QMenu`       | Menu with checkable items                |
| `SystemTrayMenu`  | `QMenu`       | Menu for system tray icons               |
| `LineEditMenu`    | `QMenu`       | Context menu for line edits              |
| `MenuAnimationType` | Enum        | Menu animation styles                    |
| `Action`          | `QAction`     | Fluent-styled menu action                |

### CycleListWidget

| Class              | Description                                        |
|-------------------|----------------------------------------------------|
| `CycleListWidget` | Infinite-scroll list widget (for picker columns)   |

### ModelComboBox

| Class              | Description                                        |
|-------------------|----------------------------------------------------|
| `ModelComboBox`   | ComboBox backed by a QAbstractItemModel             |
| `MultiSelectComboBox` | ComboBox allowing multiple selections           |

---

## 8. Layouts

| Class           | Description                                                   |
|----------------|---------------------------------------------------------------|
| `FlowLayout`   | Wrapping flow layout (items wrap to next row when full)       |
| `ExpandLayout`  | Vertical layout that expands children to fill available width |
| `VBoxLayout`   | Enhanced QVBoxLayout with convenience methods                 |

### FlowLayout Example

```python
from qfluentwidgets import FlowLayout

layout = FlowLayout(parent_widget)
layout.setContentsMargins(10, 10, 10, 10)
layout.setHorizontalSpacing(10)
layout.setVerticalSpacing(10)

for item in items:
    layout.addWidget(item)
```

---

## 9. Dialogs & Message Boxes

### Dialog Classes

| Class               | Description                                              |
|--------------------|----------------------------------------------------------|
| `Dialog`           | Standard Fluent-styled dialog (title + content + buttons)|
| `MessageBox`       | Modal message box with fluent styling                    |
| `MessageBoxBase`   | Base class for custom message boxes                      |
| `MessageDialog`    | Win10-style message dialog                               |
| `MaskDialogBase`   | Base class for dialogs with background mask overlay      |
| `ColorDialog`      | Full color picker dialog                                 |
| `FolderListDialog` | Dialog for managing a list of folder paths               |

### MessageBox Usage

```python
from qfluentwidgets import MessageBox

dialog = MessageBox(
    "Title",
    "Are you sure you want to proceed?",
    parent=self,
)

if dialog.exec():
    # User clicked OK
    pass
else:
    # User clicked Cancel
    pass
```

### Custom Dialog (MessageBoxBase)

Subclass `MessageBoxBase` to create custom dialogs with arbitrary content:

```python
from qfluentwidgets import MessageBoxBase, SubtitleLabel, LineEdit


class CustomDialog(MessageBoxBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel("Enter your name")
        self.nameInput = LineEdit()
        self.nameInput.setPlaceholderText("Type your name here")

        # Add widgets to the dialog's view layout
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.nameInput)

        # Set min width
        self.widget.setMinimumWidth(350)
```

---

## 10. Material / Acrylic Effects

Acrylic effects provide a frosted-glass visual style. Most acrylic widgets
require the **full** installation (`pip install "PyQt-Fluent-Widgets[full]"`).

| Class              | Base Class       | Description                              |
|-------------------|------------------|------------------------------------------|
| `AcrylicLabel`    | `QLabel`         | Label with acrylic blur background       |
| `AcrylicWidget`   | `QWidget`        | Generic widget with acrylic background   |
| `AcrylicMenu`     | `RoundMenu`      | Context menu with acrylic backdrop       |
| `AcrylicComboBox` | `ComboBox`       | Combo box with acrylic dropdown          |
| `AcrylicLineEdit` | `LineEdit`       | Line edit with acrylic completer popup   |
| `AcrylicFlyout`   | `Flyout`         | Flyout with acrylic background           |
| `AcrylicToolTip`  | `ToolTip`        | Tooltip with acrylic background          |

### AcrylicLabel Usage

```python
from qfluentwidgets import AcrylicLabel

label = AcrylicLabel(parent=self)
label.setPixmap(QPixmap("background.jpg"))
label.setBlurRadius(40)  # Blur intensity
```

---

## 11. Window Classes

### FluentWindow

A navigation-ready window with a left sidebar. Bundles `NavigationInterface` +
`StackedWidget` in an `QHBoxLayout`.

```python
from qfluentwidgets import FluentWindow, NavigationItemPosition, FluentIcon as FIF


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.homeInterface = HomeWidget(self)
        self.settingInterface = SettingWidget(self)

        self.addSubInterface(self.homeInterface, FIF.HOME, "Home")
        self.addSubInterface(
            self.settingInterface, FIF.SETTING, "Settings",
            position=NavigationItemPosition.BOTTOM,
        )
```

### MSFluentWindow

Microsoft Store-style Fluent window with a top navigation bar instead of a
side panel. The navigation bar appears at the top of the window.

```python
from qfluentwidgets import MSFluentWindow, NavigationItemPosition, FluentIcon as FIF


class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()

        self.homeInterface = HomeWidget(self)
        self.homeInterface.setObjectName("homeInterface")

        self.addSubInterface(self.homeInterface, FIF.HOME, "Home")
```

**Key difference**: `FluentWindow` has a left sidebar; `MSFluentWindow` has a
top navigation bar.

### SplashScreen

Show a splash screen while the application loads:

```python
from qfluentwidgets import SplashScreen

splash = SplashScreen(self)
splash.setPixmap(QPixmap("splash_image.png"))

# Show splash
splash.show()

# ... do initialization work ...

# Close splash
splash.finish()
```

### FramelessWindow

Base class for a frameless (borderless) window with custom title bar:

```python
from qfluentwidgets import FramelessWindow


class MyWindow(FramelessWindow):
    def __init__(self):
        super().__init__()
        self.setTitleBar(StandardTitleBar(self))
```

---

## 12. Multimedia

Multimedia components for audio/video playback. Located in
`qfluentwidgets.multimedia`.

| Class            | Description                                          |
|-----------------|------------------------------------------------------|
| `MediaPlayBar`  | Playback control bar (play/pause, seek, volume)      |
| `SimpleMediaPlayBar` | Minimal playback bar                            |
| `StandardMediaPlayBar` | Full-featured playback bar with all controls  |
| `MediaPlayer`   | Media player wrapper                                 |
| `VideoWidget`   | Video playback display widget                        |

### MediaPlayBar Usage

```python
from qfluentwidgets.multimedia import StandardMediaPlayBar

playBar = StandardMediaPlayBar(parent=self)
playBar.player.setSource(QUrl.fromLocalFile("audio.mp3"))
```

---

## 13. Qt Designer Integration

### Widget Promotion (No Plugin Required)

1. In Qt Designer, right-click on a standard widget (e.g., `QPushButton`)
2. Select **Promote to...** from the context menu
3. In the dialog:
   - **Header file**: `qfluentwidgets`
   - **Promoted class name**: `PushButton` (or any qfluentwidgets class)
4. Click **Add**, then **Promote**
5. Save the `.ui` file and compile with `pyuic5` / `pyside6-uic`

The generated Python code will import the fluent widget from `qfluentwidgets`.

### Plugin-Based Integration

The [Fluent Client](https://www.youtube.com/watch?v=7UCmcsOlhTk) provides
Designer plugins for drag-and-drop usage of qfluentwidgets components directly
in the visual designer.

### Complex Example Pattern

For multi-interface windows with side navigation in Designer:

1. Create the main window with a `QHBoxLayout`
2. Add a plain `QWidget` for navigation (promote to `NavigationInterface`)
3. Add a `QStackedWidget` for content pages
4. Set up navigation items in code

---

## Appendix A: Common Import Patterns

```python
# Core theming and config
from qfluentwidgets import (
    setTheme, Theme, isDarkTheme,
    setThemeColor, qconfig,
    StyleSheetBase, FluentTranslator,
)

# Navigation
from qfluentwidgets import (
    NavigationInterface, NavigationItemPosition,
    NavigationWidget, NavigationPushButton,
    FluentWindow, MSFluentWindow,
)

# Icons
from qfluentwidgets import FluentIcon as FIF, FluentIconBase, getIconColor, IconWidget

# Widgets — basic input
from qfluentwidgets import (
    PushButton, PrimaryPushButton, ToolButton,
    DropDownPushButton, SplitPushButton,
    ToggleButton, HyperlinkButton,
    CheckBox, RadioButton, SwitchButton,
    ComboBox, EditableComboBox,
    Slider, SpinBox, DoubleSpinBox,
)

# Widgets — text and labels
from qfluentwidgets import (
    LineEdit, SearchLineEdit, PasswordLineEdit,
    TextEdit, PlainTextEdit,
    TitleLabel, SubtitleLabel, StrongBodyLabel,
    BodyLabel, CaptionLabel, LargeTitleLabel,
)

# Widgets — data display
from qfluentwidgets import (
    ListView, ListWidget, TableView, TableWidget,
    TreeView, TreeWidget, FlipView, TabView,
    CardWidget, SimpleCardWidget, ElevatedCardWidget,
)

# Widgets — status and notification
from qfluentwidgets import (
    InfoBar, InfoBarPosition, InfoBadge,
    StateToolTip, ToolTip, ToolTipFilter,
    TeachingTip, Flyout, FlyoutViewBase,
)

# Widgets — progress
from qfluentwidgets import (
    ProgressBar, IndeterminateProgressBar,
    ProgressRing, IndeterminateProgressRing,
)

# Widgets — containers
from qfluentwidgets import (
    ScrollArea, SmoothScrollArea, SmoothScrollBar,
    StackedWidget, PopUpAniStackedWidget,
    Separator, CommandBar,
)

# Dialogs
from qfluentwidgets import (
    Dialog, MessageBox, MessageBoxBase,
    ColorDialog, FolderListDialog, MaskDialogBase,
)

# Settings
from qfluentwidgets import (
    QConfig, ConfigItem, OptionsConfigItem,
    RangeConfigItem, ColorConfigItem,
    BoolValidator, OptionsValidator, RangeValidator,
    ConfigSerializer, EnumSerializer, ColorSerializer,
    SettingCard, SettingCardGroup,
    SwitchSettingCard, ComboBoxSettingCard,
    RangeSettingCard, PushSettingCard,
    PrimaryPushSettingCard, HyperlinkCard,
    ColorSettingCard, CustomColorSettingCard,
    OptionsSettingCard, FolderListSettingCard,
)

# Layouts
from qfluentwidgets import FlowLayout, ExpandLayout, VBoxLayout

# Menus
from qfluentwidgets import (
    RoundMenu, CheckableMenu, SystemTrayMenu,
    Action, MenuAnimationType,
)

# Material / Acrylic (full install only for AcrylicLabel)
from qfluentwidgets import (
    AcrylicLabel, AcrylicWidget, AcrylicMenu,
    AcrylicComboBox, AcrylicLineEdit,
    AcrylicFlyout, AcrylicToolTip,
)

# Window
from qfluentwidgets import FluentWindow, MSFluentWindow, SplashScreen

# Multimedia
from qfluentwidgets.multimedia import (
    MediaPlayBar, StandardMediaPlayBar, SimpleMediaPlayBar,
    MediaPlayer, VideoWidget,
)

# Fonts
from qfluentwidgets import setFont
```

## Appendix B: Signals Reference

### Config Signals (on qconfig instance)

| Signal               | Emitted When                          |
|---------------------|---------------------------------------|
| `themeChanged`      | Theme mode changes via `setTheme()`   |
| `themeColorChanged` | Theme color changes via `setThemeColor()` |
| `appRestartSig`     | A config item with `restart=True` changes |

### Navigation Signals

| Signal                         | On                    | Description                    |
|-------------------------------|-----------------------|--------------------------------|
| `clicked`                     | `NavigationWidget`    | Item was clicked               |

### Widget Signals

Standard Qt signals apply. Notable additions:

| Widget          | Signal           | Description                           |
|----------------|------------------|---------------------------------------|
| `SwitchButton` | `checkedChanged` | Switch toggled on/off                 |
| `InfoBar`      | `closedSignal`   | InfoBar was dismissed                 |
| `Flyout`       | `closed`         | Flyout was closed                     |

---

## Appendix C: WaveDL-GUI Project Usage

This project (`WaveDL-GUI`) uses qfluentwidgets in 15+ files. Key patterns:

| Pattern                                      | Used In                              |
|---------------------------------------------|--------------------------------------|
| `MSFluentWindow` base class                 | `view/main_window.py`               |
| `StyleSheetBase` + Enum with `path()`       | `common/style_sheet.py`             |
| `FluentIconBase` + Enum with `path()`       | `common/icon.py`                    |
| `QConfig` subclass with `ConfigItem` fields | `common/config.py`                  |
| `SettingCardGroup`, `SwitchSettingCard`      | `view/setting_interface.py`         |
| `NavigationItemPosition.TOP/BOTTOM`         | `view/main_window.py`               |
| `FluentTranslator`                          | `main.py`                           |
| `InfoBar`, `InfoBarPosition`                | `view/main_window.py` and others    |
| `MessageBox`                                | `view/main_window.py`               |
| `SplashScreen`                              | `view/main_window.py`               |
| `FluentIcon as FIF`                         | Multiple view files                  |
| `isDarkTheme()`                             | `components/statistic_widget.py`    |
| `SimpleCardWidget`                          | `components/empty_status_widget.py` |
| `SystemTrayMenu`, `Action`                  | `components/system_tray_icon.py`    |
| `setTheme`, `Theme`                         | `components/system_tray_icon.py`    |
| `ScrollArea`, `FlowLayout`                  | `components/interface.py`           |
| `StrongBodyLabel`, `CaptionLabel`, `setFont`| `components/statistic_widget.py`    |
