from qfluentwidgets import LineEdit, SettingCard


class TextSettingCard(SettingCard):
    def __init__(self, icon, title, content, parent=None):
        super().__init__(icon, title, content, parent)
        self.lineEdit = LineEdit()
        self.lineEdit.setPlaceholderText("Enter text...")

        self.hBoxLayout.addWidget(self.lineEdit)
        self.hBoxLayout.addSpacing(15)


# Example usage in a settings interface:
# card = TextSettingCard(FluentIcon.EDIT, "Username", "Change your username")
# card.lineEdit.setText(cfg.username.value)
# card.lineEdit.textChanged.connect(lambda text: setattr(cfg, 'username', text))
