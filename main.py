#!/usr/bin/env python3
"""
DataMatrix Quality Scanner - Modern Edition
Сканер качества печати DataMatrix согласно ГОСТ Р 57302-2016

Авторы: А. Свидович / А. Петляков для PROGRESS
Версия: 2.0.0 (Modern Rewrite)
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QFont

from src.main_window import MainWindow


def main():
    """Точка входа в приложение"""
    
    # Настройка High-DPI
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("DataMatrix Scanner")
    app.setOrganizationName("PROGRESS")
    app.setApplicationVersion("2.0.0")
    
    # Установка современного шрифта
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Создание и показ главного окна
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
