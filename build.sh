#!/bin/bash
# Скрипт сборки exe файла для Windows

set -e

echo "=== DataMatrix Scanner Build Script ==="
echo "Авторы: А. Свидович / А. Петляков для PROGRESS"
echo ""

# Проверка Python
if ! command -v python &> /dev/null; then
    echo "Ошибка: Python не установлен"
    exit 1
fi

echo "Версия Python: $(python --version)"

# Создание виртуального окружения (опционально)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Создание виртуального окружения..."
    python -m venv venv
    source venv/bin/activate
fi

# Установка зависимостей
echo "Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

# Установка OpenCV contrib для декодирования DataMatrix
pip install opencv-contrib-python

echo ""
echo "=== Сборка exe файла ==="

# Сборка PyInstaller
pyinstaller build.spec --clean

echo ""
echo "=== Сборка завершена ==="
echo "Файл: dist/DataMatrixScanner/DataMatrixScanner.exe"

# Создание архива
cd dist
ARCHIVE_NAME="DataMatrixScanner-$(date +%Y%m%d-%H%M%S)-win64.zip"
zip -r "$ARCHIVE_NAME" DataMatrixScanner
echo "Архив: dist/$ARCHIVE_NAME"

# Хеш-суммы
echo ""
echo "=== Хеш-суммы ==="
sha256sum "$ARCHIVE_NAME"
sha256sum DataMatrixScanner/DataMatrixScanner.exe
