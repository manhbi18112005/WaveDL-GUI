"""WaveDL GUI - Styled SettingCardGroup."""

from PySide6.QtGui import QFont
from qfluentwidgets import SettingCardGroup as _CardGroup, setFont


class SettingCardGroup(_CardGroup):
    """A styled card group with a DemiBold 14pt title.

    Previously duplicated across ProjectInterface,
    TrainingInterface, and SettingInterface.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        setFont(self.titleLabel, 14, QFont.Weight.DemiBold)
