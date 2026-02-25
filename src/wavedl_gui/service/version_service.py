import re

import requests
from PySide6.QtCore import QVersionNumber

from ..common.exception_handler import exceptionHandler
from ..common.setting import VERSION
from ..common.utils import getSystemProxy


class VersionService:
    """Version service"""

    def __init__(self):
        self.currentVersion = VERSION
        self.lastestVersion = VERSION
        self.versionPattern = re.compile(r"v(\d+)\.(\d+)\.(\d+)")

    @exceptionHandler("version", VERSION)
    def getLatestVersion(self):
        """get latest version"""
        url = "https://api.github.com/repos/manhbi18112005/WaveDL-GUI/releases/latest"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.64"
        }

        proxy = getSystemProxy()
        proxies = {"http": proxy, "https": proxy}

        response = requests.get(
            url, headers=headers, timeout=5, allow_redirects=True, proxies=proxies
        )
        response.raise_for_status()

        # parse version
        version = response.json()["tag_name"]  # type:str
        match = self.versionPattern.search(version)
        if not match:
            return VERSION

        self.lastestVersion = version[1:]
        return self.lastestVersion

    def hasNewVersion(self) -> bool:
        """check whether there is a new version"""
        version = QVersionNumber.fromString(self.getLatestVersion())
        currentVersion = QVersionNumber.fromString(self.currentVersion)
        return version > currentVersion
